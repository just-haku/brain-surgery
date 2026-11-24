"""Deterministic quality, emoji/emoticon policy, and NativeTools schema validator."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any

from surgery.vocab_filter import is_emoji_or_banned_symbol

from .formatter import (
    render_gemma4_orpo_pair,
    render_gemma4_training_sample,
    validate_gemma4_wire_format,
)
from .nc_schema import NC_TOOL_SPECS, get_nc_openai_tools, normalize_tool_call, validate_tool_call

_ASCII_EMOTICON_RE = re.compile(
    r"(?:(?<![A-Za-z0-9_:/=])(?::\)+|:\(+|:D+|;+\)+|:P\b)|(?<![A-Za-z0-9_])(?:=+\)+|\._\.|\.\-\.|\^_\^))(?=$|[\s,.!?;'\"\]}])"
)
_FENCED_CODE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_RAW_JSON_LEAK_RE = re.compile(r'\{\s*"(?:thought|action|tool)"\s*:')

VALID_MODES = frozenset({"nc_agent", "chat"})
VALID_ORPO_FAILURES = frozenset(
    {"context_bleed", "tool_schema_or_json_leak", "type_mutation", "repetition_or_emoji"}
)


def contains_ascii_emoticon(text: str) -> bool:
    """Detect ASCII emoticons (:), :))), ._., .-., =)), etc.) without code false-positives."""
    return bool(_ASCII_EMOTICON_RE.search(text))


def extract_code_blocks_and_prose(text: str) -> tuple[list[str], str]:
    """Separate fenced and inline code blocks from surrounding natural prose."""
    code_blocks: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        code_blocks.append(match.group(0))
        return " "

    without_fenced = _FENCED_CODE_RE.sub(_collect, text)
    prose_only = _INLINE_CODE_RE.sub(_collect, without_fenced)
    return code_blocks, prose_only


def has_repetition_loop(text: str, ngram_size: int = 8, max_repeats: int = 3) -> bool:
    """Detect repetitive word n-grams or consecutive duplicate lines typical of 2B loops."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 12]
    if lines:
        line_counts = Counter(lines)
        if max(line_counts.values()) >= max_repeats + 1:
            return True

    words = text.split()
    if len(words) < ngram_size * max_repeats:
        return False
    ngrams = Counter(tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1))
    return any(cnt >= max_repeats for cnt in ngrams.values())


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        out = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        out = tokenizer.encode(text)
    return list(out.ids) if hasattr(out, "ids") else list(out)


def _check_unicode_emoji_in_obj(obj: Any) -> bool:
    if isinstance(obj, str):
        return is_emoji_or_banned_symbol(obj)
    if isinstance(obj, dict):
        return any(_check_unicode_emoji_in_obj(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_check_unicode_emoji_in_obj(v) for v in obj)
    return False


def _validate_model_turn(
    msg: dict[str, Any],
    mode: str,
    role: str,
    enable_thinking: bool,
) -> list[str]:
    errors: list[str] = []
    thought = msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thought") or ""
    content = str(msg.get("content") or "")
    tool_calls = msg.get("tool_calls") or []

    if _check_unicode_emoji_in_obj(thought) or _check_unicode_emoji_in_obj(content) or _check_unicode_emoji_in_obj(tool_calls):
        errors.append("banned Unicode emoji / variation selector / ZWJ detected")

    if enable_thinking:
        if not isinstance(thought, str) or not thought.strip():
            errors.append("enable_thinking=True requires non-empty model thinking trace")
        else:
            if contains_ascii_emoticon(thought):
                errors.append("ASCII emoticon forbidden inside thinking block")
            if has_repetition_loop(thought):
                errors.append("repetition loop detected inside thinking block")

    if not tool_calls and not content.strip():
        errors.append("model turn must contain 'tool_calls' or non-empty 'content'")
        return errors

    if tool_calls:
        if mode != "nc_agent":
            errors.append("tool_calls only allowed in mode='nc_agent'")
        if not isinstance(tool_calls, list):
            errors.append("'tool_calls' must be a list")
        else:
            for idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    errors.append(f"tool_calls[{idx}] must be a dict")
                    continue
                try:
                    tname, targs = normalize_tool_call(tc)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"tool_calls[{idx}] invalid arguments JSON: {exc}")
                    continue
                if contains_ascii_emoticon(json.dumps(targs, ensure_ascii=False)):
                    errors.append(f"tool_calls[{idx}] contains forbidden ASCII emoticon")
                errors.extend(f"tool_calls[{idx}]: {e}" for e in validate_tool_call(tname, targs))

    if content.strip():
        if has_repetition_loop(content):
            errors.append("repetition loop detected inside model content")
        if mode == "nc_agent":
            if contains_ascii_emoticon(content):
                errors.append("ASCII emoticon forbidden inside nc_agent output")
            if _RAW_JSON_LEAK_RE.search(content):
                errors.append("raw JSON tool-call envelope leaked inside nc_agent plain content")
            if role == "planner":
                try:
                    plan = json.loads(content.strip())
                    if not isinstance(plan, dict) or "tasks" not in plan or not (1 <= len(plan["tasks"]) <= 4):
                        errors.append("planner content must be JSON with 1..4 tasks")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"planner content must be valid JSON: {exc}")
        elif mode == "chat":
            code_blocks, prose_only = extract_code_blocks_and_prose(content)
            for blk in code_blocks:
                if contains_ascii_emoticon(blk):
                    errors.append("ASCII emoticon forbidden inside code block")
            if _RAW_JSON_LEAK_RE.search(prose_only):
                errors.append("raw nc-agent JSON leaked into chat prose")

    return errors


def _resolve_tools_for_sample(sample: dict[str, Any]) -> list[dict[str, Any]] | None:
    if sample.get("tools"):
        return sample["tools"]
    if sample.get("mode") == "nc_agent":
        return get_nc_openai_tools()
    return None


def validate_sft_sample(
    sample: dict[str, Any],
    tokenizer: Any | None = None,
    max_tokens: int = 4096,
) -> list[str]:
    """Validate a NativeTools / bilingual chat SFT sample and render Gemma 4 wire format."""
    if not isinstance(sample, dict):
        return ["sample must be a dict"]

    errors: list[str] = []
    mode = sample.get("mode")
    if mode not in VALID_MODES:
        return [f"invalid mode {mode!r}; expected one of {sorted(VALID_MODES)}"]

    role = str(sample.get("role") or "coder")
    enable_thinking = bool(sample.get("enable_thinking", True))
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["'messages' must be a non-empty list"]

    has_model_turn = False
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"messages[{idx}] must be a dict")
            continue
        m_role = msg.get("role")
        if _check_unicode_emoji_in_obj(msg):
            errors.append(f"messages[{idx}]: banned Unicode emoji detected")
            continue
        if m_role in ("model", "assistant"):
            has_model_turn = True
            turn_errs = _validate_model_turn(
                msg=msg, mode=mode, role=role, enable_thinking=enable_thinking
            )
            errors.extend(f"messages[{idx}]: {e}" for e in turn_errs)
        elif m_role == "tool":
            tname = msg.get("name")
            if tname not in NC_TOOL_SPECS:
                errors.append(f"messages[{idx}]: unknown tool name {tname!r}")
            if not str(msg.get("content") or "").strip():
                errors.append(f"messages[{idx}]: empty tool response content")

    if not has_model_turn:
        errors.append("conversation must include at least one assistant/model turn")
        return errors

    try:
        rendered = render_gemma4_training_sample(
            messages,
            tools=_resolve_tools_for_sample(sample),
            enable_thinking=enable_thinking,
        )
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f"wire format error: {exc}")
        return errors

    errors.extend(validate_gemma4_wire_format(rendered["text"]))

    if tokenizer is not None:
        ids = _encode_ids(tokenizer, rendered["text"])
        if len(ids) > max_tokens:
            errors.append(f"token count {len(ids)} exceeds max_tokens={max_tokens}")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None and unk_id in ids:
            errors.append("<unk> token produced by pruned tokenizer")

    return errors


def validate_orpo_pair(
    pair: dict[str, Any],
    tokenizer: Any | None = None,
    max_tokens: int = 4096,
) -> list[str]:
    """Validate an ORPO preference pair ensuring chosen is spotless and distinct from rejected."""
    if not isinstance(pair, dict):
        return ["ORPO pair must be a dict"]

    errors: list[str] = []
    failure_type = pair.get("failure_type")
    if failure_type not in VALID_ORPO_FAILURES:
        errors.append(f"invalid failure_type {failure_type!r}")

    mode = pair.get("mode", "nc_agent")
    if mode not in VALID_MODES:
        return [f"invalid mode {mode!r}"]
    role = str(pair.get("role") or "coder")
    enable_thinking = bool(pair.get("enable_thinking", True))

    prompt_msgs = pair.get("prompt_messages")
    chosen = pair.get("chosen")
    rejected = pair.get("rejected")
    if not isinstance(prompt_msgs, list) or not prompt_msgs:
        return ["'prompt_messages' must be a non-empty list"]
    if not isinstance(chosen, dict) or not isinstance(rejected, dict):
        return ["'chosen' and 'rejected' must be dicts"]

    if chosen == rejected:
        errors.append("'chosen' and 'rejected' must differ")

    for idx, msg in enumerate(prompt_msgs):
        if _check_unicode_emoji_in_obj(msg):
            errors.append(f"prompt_messages[{idx}] contains banned Unicode emoji")

    chosen_errs = _validate_model_turn(
        msg=chosen, mode=mode, role=role, enable_thinking=enable_thinking
    )
    errors.extend(f"chosen: {e}" for e in chosen_errs)

    if not (rejected.get("tool_calls") or str(rejected.get("content") or "").strip()):
        errors.append("rejected must contain 'tool_calls' or non-empty 'content'")

    try:
        rendered = render_gemma4_orpo_pair(
            prompt_msgs,
            chosen,
            rejected,
            tools=_resolve_tools_for_sample(pair),
            enable_thinking=enable_thinking,
        )
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f"ORPO wire render error: {exc}")
        return errors

    errors.extend(
        validate_gemma4_wire_format(
            rendered["prompt"] + rendered["chosen"],
            allow_partial_tool_turn=bool(chosen.get("tool_calls")),
        )
    )

    if tokenizer is not None:
        chosen_ids = _encode_ids(tokenizer, rendered["prompt"] + rendered["chosen"])
        rejected_ids = _encode_ids(tokenizer, rendered["prompt"] + rendered["rejected"])
        if len(chosen_ids) > max_tokens or len(rejected_ids) > max_tokens:
            errors.append(
                f"ORPO pair exceeds max_tokens={max_tokens}: chosen={len(chosen_ids)}, rejected={len(rejected_ids)}"
            )

    return errors


def deduplicate_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically deduplicate samples by normalized message content hash."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in samples:
        payload = item.get("messages") or item.get("prompt_messages") or item
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(item)
    return unique

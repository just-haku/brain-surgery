"""Deterministic quality, Adaptive Thinking (85/15), emoji/emoticon, and NativeTools validator."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from surgery.vocab_filter import is_emoji_or_banned_symbol

from .formatter import (
    render_gemma4_orpo_pair,
    render_gemma4_training_sample,
    validate_gemma4_wire_format,
)
from .nc_schema import (
    NC_TOOL_SPECS,
    get_nc_openai_tools,
    normalize_tool_call,
    validate_tool_call,
)

MAX_REASONING_CHARS = 240
READ_ONLY_TOOLS = frozenset(
    {"list_files", "search_text", "read_outline", "read_symbol", "read_slice"}
)
MUTATING_TOOLS = frozenset(
    {"edit_file", "write_file", "run_command", "spawn_subagent"}
)

_ASCII_EMOTICON_RE = re.compile(
    r"(?:(?<![A-Za-z0-9_:/=])(?::\)+|:\(+|:D+|;+\)+|:P\b)|(?<![A-Za-z0-9_])(?:=+\)+|\._\.|\.\-\.|\^_\^))(?=$|[\s,.!?;'\"\]}])"
)
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_RAW_JSON_LEAK_RE = re.compile(r'\{\s*"(?:thought|action|tool)"\s*:')
_ROLE_LINE_RE = re.compile(r"\bRole:\s*(PLANNER|CODER|REVIEWER|EXPLORER)\b", re.IGNORECASE)

VALID_MODES = frozenset({"nc_agent", "chat"})
PER_STAGES = frozenset({"initial_plan", "micro_execute", "replan"})
RECEIPT_FIELDS = ("STATUS:", "CHANGES:", "FILES:", "ISSUES:")
VALID_ORPO_FAILURES = frozenset(
    {
        # Legacy Stage 2B failure buckets
        "context_bleed",
        "tool_schema_or_json_leak",
        "type_mutation",
        "repetition_or_emoji",
        # Prompt B NativeTools + CodeGraph + spawn_subagent buckets
        "edit_exact_vs_hallucinated_args",
        "graph_read_symbol_vs_blind_dump",
        "no_thinking_tool_vs_runaway_cot",
        "subagent_report_vs_nested_spawn",
        "stop_and_cite_vs_duplicate_or_nested_json",
        "ascii_vs_unicode_emoji",
        # Wave 4 PER Loop preference buckets
        "per_granular_plan_vs_monolithic",
        "per_ledger_replan_vs_repeat_failed",
        "per_compact_receipt_vs_verbose",
        "per_worker_no_thinking_vs_thinking_on",
    }
)


def _est_tokens(text: str) -> int:
    return (len(text or "") + 3) // 4


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
    """Detect repetitive word n-grams or consecutive duplicate lines typical of 2B/4B loops."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 12]
    if lines:
        line_counts = Counter(lines)
        if max(line_counts.values()) >= max_repeats + 1:
            return True

    words = text.split()
    if len(words) < ngram_size * max_repeats:
        return False
    ngrams = Counter(
        tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)
    )
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
        return any(
            _check_unicode_emoji_in_obj(k) or _check_unicode_emoji_in_obj(v)
            for k, v in obj.items()
        )
    if isinstance(obj, list):
        return any(_check_unicode_emoji_in_obj(v) for v in obj)
    return False


def _infer_mode(sample: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    explicit = sample.get("mode")
    if explicit is not None:
        return str(explicit)
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool" or m.get("tool_calls"):
            return "nc_agent"
        txt = str(m.get("content") or "")
        if "## REPO_MAP" in txt or "## TASK" in txt or _ROLE_LINE_RE.search(txt):
            return "nc_agent"
    return "chat"


def _infer_role_and_worker(
    sample: dict[str, Any],
    messages: list[dict[str, Any]],
    mode: str,
) -> tuple[str, bool]:
    sys_txt = ""
    if messages and isinstance(messages[0], dict) and messages[0].get("role") in ("system", "developer"):
        sys_txt = str(messages[0].get("content") or "")

    per_stage = sample.get("per_stage")
    role_val = sample.get("role")
    if role_val:
        role = str(role_val).lower()
    elif per_stage in ("initial_plan", "replan"):
        role = "planner"
    else:
        m = _ROLE_LINE_RE.search(sys_txt)
        if m:
            role = m.group(1).lower()
        else:
            role = "coder" if mode == "nc_agent" else "chat"

    depth_val = int(sample.get("depth", 0) or 0)
    is_worker = bool(
        depth_val >= 1
        or role == "explorer"
        or sample.get("bucket") == "per_subagent_worker"
        or "depth=1" in sys_txt
        or "depth-1 worker" in sys_txt.lower()
        or (
            per_stage is None
            and role in ("coder", "reviewer")
            and "Role: CODER" in sys_txt
            and "spawn_subagent" not in sys_txt
        )
    )
    return role, is_worker


def _infer_thinking_flag(sample: dict[str, Any], *, default: bool = True) -> bool:
    if "thinking" in sample and sample["thinking"] is not None:
        return bool(sample["thinking"])
    if "enable_thinking" in sample and sample["enable_thinking"] is not None:
        return bool(sample["enable_thinking"])
    return default


def _resolve_tools_for_sample(
    sample: dict[str, Any],
    *,
    mode: str,
    role: str = "coder",
    is_worker: bool = False,
    messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]] | None:
    if sample.get("tools"):
        return sample["tools"]
    if (
        role == "planner"
        or sample.get("grammar") == "planner"
        or sample.get("per_stage") in ("initial_plan", "replan")
    ):
        return None
    if mode == "nc_agent":
        sys_omits_spawn = bool(
            messages
            and isinstance(messages[0], dict)
            and sample.get("per_stage") == "micro_execute"
            and "spawn_subagent" not in str(messages[0].get("content") or "")
        )
        return get_nc_openai_tools(depth=1 if (is_worker or sys_omits_spawn) else 0)
    return None


def _check_planner_tasks_detail(comp_text: str) -> list[str]:
    errs: list[str] = []
    try:
        parsed = json.loads(comp_text)
    except Exception:
        return errs
    if not isinstance(parsed, dict):
        return ["planner content must be JSON object"]
    if list(parsed.keys()) != ["goal", "decisions", "tasks"]:
        errs.append(f"planner keys must be ['goal', 'decisions', 'tasks'], got {list(parsed.keys())}")
    tasks = parsed.get("tasks")
    if not isinstance(tasks, list) or not (1 <= len(tasks) <= 4):
        return errs
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            errs.append(f"tasks[{i}] must be object")
            continue
        if not str(task.get("id") or "").strip():
            errs.append(f"tasks[{i}] missing non-empty 'id'")
        if not str(task.get("title") or "").strip():
            errs.append(f"tasks[{i}] missing non-empty 'title'")
        tfiles = task.get("target_files")
        if not isinstance(tfiles, list) or not all(isinstance(x, str) for x in tfiles):
            errs.append(f"tasks[{i}].target_files must be list[str]")
    return errs


def _check_subtask_receipt_text(content: str) -> list[str]:
    errs: list[str] = []
    upper_lines = [ln.strip().upper() for ln in content.splitlines() if ln.strip()]
    for prefix in RECEIPT_FIELDS:
        if not any(ln.startswith(prefix) for ln in upper_lines):
            errs.append(f"micro_execute SubTaskReceipt missing '{prefix}' line")
    status_lines = [ln for ln in upper_lines if ln.startswith("STATUS:")]
    if status_lines and not any(tok in status_lines[0] for tok in ("DONE", "BLOCKED")):
        errs.append("SubTaskReceipt STATUS must be DONE or BLOCKED")
    return errs


def _check_native_initial_user(
    messages: list[dict[str, Any]],
    sample: dict[str, Any],
    prefix: str,
) -> list[str]:
    if sample.get("grammar") in ("tool_call", "final_only"):
        return []
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            txt = msg.get("content")
            if not isinstance(txt, str) or not txt.strip():
                return [f"{prefix} messages[{idx}] user prompt has empty content"]
            if "## SCRATCHPAD" in txt or "## NEXT" in txt:
                return [f"{prefix} NativeTools user prompt must omit ## SCRATCHPAD and ## NEXT"]
            return []
    return []


def _validate_model_turn(
    msg: dict[str, Any],
    mode: str,
    role: str,
    enable_thinking: bool,
    *,
    is_worker: bool = False,
    seen_read_sigs: list[tuple[str, str]] | None = None,
) -> list[str]:
    errors: list[str] = []
    thought = msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thought") or ""
    content = str(msg.get("content") or "")
    tool_calls = msg.get("tool_calls") or []

    if (
        _check_unicode_emoji_in_obj(thought)
        or _check_unicode_emoji_in_obj(content)
        or _check_unicode_emoji_in_obj(tool_calls)
    ):
        errors.append("banned Unicode emoji / variation selector / ZWJ detected")

    if isinstance(thought, str) and thought.strip():
        if not enable_thinking:
            errors.append("thinking=False forbids non-empty reasoning_content")
        if len(thought) > MAX_REASONING_CHARS:
            errors.append(
                f"reasoning_content exceeds {MAX_REASONING_CHARS} chars ({len(thought)})"
            )
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
                if is_worker and tname == "spawn_subagent":
                    errors.append(f"tool_calls[{idx}]: depth >= 1 worker cannot call spawn_subagent")
                if role == "explorer" and tname in MUTATING_TOOLS:
                    errors.append(f"tool_calls[{idx}]: explorer role cannot call mutating tool {tname!r}")
                errors.extend(f"tool_calls[{idx}]: {e}" for e in validate_tool_call(tname, targs))
                if seen_read_sigs is not None and tname in READ_ONLY_TOOLS:
                    sig = (tname, json.dumps(targs, sort_keys=True, ensure_ascii=False))
                    if seen_read_sigs and seen_read_sigs[-1] == sig:
                        errors.append(f"tool_calls[{idx}]: duplicate consecutive read call {sig}")
                    seen_read_sigs.append(sig)

    if content.strip():
        if has_repetition_loop(content):
            errors.append("repetition loop detected inside model content")
        code_blocks, prose_only = extract_code_blocks_and_prose(content)
        for blk in code_blocks:
            if contains_ascii_emoticon(blk):
                errors.append("ASCII emoticon forbidden inside code block")

        if mode == "nc_agent":
            if _RAW_JSON_LEAK_RE.search(content):
                errors.append("raw JSON tool-call envelope leaked inside nc_agent plain content")
            if role == "planner":
                if contains_ascii_emoticon(content):
                    errors.append("ASCII emoticon forbidden inside planner JSON")
                try:
                    plan = json.loads(content.strip())
                    if (
                        not isinstance(plan, dict)
                        or "tasks" not in plan
                        or not (1 <= len(plan["tasks"]) <= 4)
                    ):
                        errors.append("planner content must be JSON with 1..4 tasks")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"planner content must be valid JSON: {exc}")
        elif mode == "chat":
            if _RAW_JSON_LEAK_RE.search(prose_only):
                errors.append("raw nc-agent JSON leaked into chat prose")

    return errors


def validate_sft_sample(
    sample: dict[str, Any],
    tokenizer: Any | None = None,
    max_tokens: int = 8192,
) -> list[str]:
    """Validate an OpenAI NativeTools / bilingual SFT trajectory and render Gemma 4 wire format."""
    if not isinstance(sample, dict):
        return ["sample must be a dict"]

    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["'messages' must be a non-empty list"]

    mode = _infer_mode(sample, messages)
    if mode not in VALID_MODES:
        return [f"invalid mode {mode!r}; expected one of {sorted(VALID_MODES)}"]

    role, is_worker = _infer_role_and_worker(sample, messages, mode)
    enable_thinking = _infer_thinking_flag(sample, default=True)
    if is_worker and "thinking" in sample and enable_thinking:
        return ["depth >= 1 worker/explorer/reviewer should keep thinking=False"]

    errors: list[str] = []
    per_stage = sample.get("per_stage")
    if per_stage is not None and per_stage not in PER_STAGES:
        errors.append(f"unknown per_stage {per_stage!r}; expected one of {sorted(PER_STAGES)}")
    if mode == "nc_agent":
        errors.extend(_check_native_initial_user(messages, sample, "SFT"))

    first_user_idx = next(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if per_stage == "replan" and first_user_idx is not None:
        init_txt = str(messages[first_user_idx].get("content") or "")
        if "## LEDGER" not in init_txt or "REPLAN" not in init_txt:
            errors.append("per_stage 'replan' initial user prompt must include ## LEDGER and REPLAN")

    has_model_turn = False
    has_any_thought = False
    assistant_turns = 0
    seen_read_sigs: list[tuple[str, str]] = []

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
            assistant_turns += 1
            thought_val = msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thought") or ""
            if isinstance(thought_val, str) and thought_val.strip():
                has_any_thought = True
            turn_errs = _validate_model_turn(
                msg=msg,
                mode=mode,
                role=role,
                enable_thinking=enable_thinking,
                is_worker=is_worker,
                seen_read_sigs=seen_read_sigs,
            )
            errors.extend(f"messages[{idx}]: {e}" for e in turn_errs)
        elif m_role == "user" and first_user_idx is not None and idx > first_user_idx:
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                errors.append(f"messages[{idx}]: empty mid-trajectory user prompt")
        elif m_role == "tool":
            tname = msg.get("name")
            if tname not in NC_TOOL_SPECS:
                errors.append(f"messages[{idx}]: unknown tool name {tname!r}")
            if not str(msg.get("tool_call_id") or "").strip():
                errors.append(f"messages[{idx}]: role 'tool' requires non-empty tool_call_id")
            if not isinstance(msg.get("content"), str):
                errors.append(f"messages[{idx}]: tool response content must be a string")

    if not has_model_turn:
        errors.append("conversation must include at least one assistant/model turn")
        return errors

    if enable_thinking and not has_any_thought:
        errors.append("enable_thinking=True requires non-empty reasoning_content in at least one model turn")

    if per_stage in ("initial_plan", "replan"):
        errors.extend(_check_planner_tasks_detail(str(messages[-1].get("content") or "").strip()))
    elif per_stage == "micro_execute":
        if not (1 <= assistant_turns <= 5):
            errors.append(f"per_stage 'micro_execute' has {assistant_turns} assistant turns (max 5)")
        final_msg = messages[-1]
        if isinstance(final_msg.get("content"), str) and not final_msg.get("tool_calls"):
            errors.extend(_check_subtask_receipt_text(final_msg["content"]))
        else:
            errors.append("per_stage 'micro_execute' must end with plain-text SubTaskReceipt")

    total_est = sum(
        _est_tokens(m.get("content") or "") + _est_tokens(m.get("reasoning_content") or "")
        for m in messages
        if isinstance(m, dict)
    )
    ceiling = 14000 if sample.get("ctx_profile") == "16k" else 7600
    if total_est > ceiling:
        errors.append(
            f"estimated token count ~{total_est} exceeds ctx_profile ceiling ({ceiling})"
        )

    try:
        rendered = render_gemma4_training_sample(
            messages,
            tools=_resolve_tools_for_sample(
                sample, mode=mode, role=role, is_worker=is_worker, messages=messages
            ),
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
    """Validate an ORPO preference pair (both legacy schema and Prompt B schema)."""
    if not isinstance(pair, dict):
        return ["ORPO pair must be a dict"]

    prompt_msgs = pair.get("prompt_messages") or pair.get("prompt")
    chosen = pair.get("chosen")
    rejected = pair.get("rejected")
    if not isinstance(prompt_msgs, list) or not prompt_msgs:
        return ["'prompt' / 'prompt_messages' must be a non-empty list"]
    if not isinstance(chosen, dict) or not isinstance(rejected, dict):
        return ["'chosen' and 'rejected' must be dicts"]

    errors: list[str] = []
    failure_type = pair.get("failure_type") or pair.get("bucket")
    if failure_type not in VALID_ORPO_FAILURES:
        errors.append(f"invalid ORPO bucket/failure_type {failure_type!r}")

    mode = _infer_mode(pair, prompt_msgs)
    if mode not in VALID_MODES:
        return [f"invalid mode {mode!r}"]
    role, is_worker = _infer_role_and_worker(pair, prompt_msgs, mode)

    chosen_thought = str(
        chosen.get("reasoning_content") or chosen.get("reasoning") or chosen.get("thought") or ""
    ).strip()
    enable_thinking = _infer_thinking_flag(pair, default=bool(chosen_thought))

    if chosen == rejected:
        errors.append("'chosen' and 'rejected' must differ")

    for idx, msg in enumerate(prompt_msgs):
        if _check_unicode_emoji_in_obj(msg):
            errors.append(f"prompt[{idx}] contains banned Unicode emoji")
    if mode == "nc_agent":
        errors.extend(_check_native_initial_user(prompt_msgs, pair, "ORPO"))

    # Seed duplicate read check with the most recent read tool call in prompt if present.
    prior_reads: list[tuple[str, str]] = []
    for m in prompt_msgs:
        if isinstance(m, dict) and m.get("role") in ("assistant", "model"):
            for tc in m.get("tool_calls") or []:
                try:
                    tname, targs = normalize_tool_call(tc)
                except Exception:  # noqa: BLE001
                    continue
                if tname in READ_ONLY_TOOLS:
                    prior_reads.append((tname, json.dumps(targs, sort_keys=True, ensure_ascii=False)))

    chosen_errs = _validate_model_turn(
        msg=chosen,
        mode=mode,
        role=role,
        enable_thinking=enable_thinking,
        is_worker=is_worker,
        seen_read_sigs=prior_reads,
    )
    errors.extend(f"chosen: {e}" for e in chosen_errs)
    if failure_type == "per_worker_no_thinking_vs_thinking_on" and chosen_thought:
        errors.append("chosen: depth=1 worker must keep thinking=False / empty reasoning_content")
    if enable_thinking and not chosen_thought:
        errors.append("chosen: enable_thinking=True requires non-empty reasoning_content")

    has_rejected_signal = bool(
        rejected.get("tool_calls")
        or str(rejected.get("content") or "").strip()
        or str(rejected.get("reasoning_content") or "").strip()
    )
    if not has_rejected_signal:
        errors.append("rejected must contain 'tool_calls', 'content', or 'reasoning_content'")

    try:
        rendered = render_gemma4_orpo_pair(
            prompt_msgs,
            chosen,
            rejected,
            tools=_resolve_tools_for_sample(
                pair, mode=mode, role=role, is_worker=is_worker, messages=prompt_msgs
            ),
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
        payload = item.get("messages") or item.get("prompt_messages") or item.get("prompt") or item
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(item)
    return unique


def convert_and_validate_openai_jsonl(
    raw_jsonl_paths: list[str | Path],
    out_sft_path: str | Path,
    out_orpo_path: str | Path,
    out_rejected_path: str | Path,
    tokenizer: Any | None = None,
    max_tokens: int = 8192,
    max_thinking_ratio: float = 0.20,
) -> dict[str, Any]:
    """Ingest AGY_DISTILL_PROMPTS OpenAI JSONLs, validate 85/15 thinking + schema, and render Gemma 4."""
    valid_sft: list[dict[str, Any]] = []
    valid_orpo: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    total_lines = 0

    for raw_path in raw_jsonl_paths:
        for line_no, raw_line in enumerate(Path(raw_path).read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            total_lines += 1
            try:
                obj = json.loads(raw_line)
                if not isinstance(obj, dict):
                    raise ValueError("JSONL line is not an object")
            except Exception as exc:  # noqa: BLE001
                msg = f"parse_error ({Path(raw_path).name}:L{line_no}): {exc}"
                rejection_reasons[msg] += 1
                rejected_rows.append({"line": raw_line[:300], "errors": [msg]})
                continue

            is_orpo = "chosen" in obj and "rejected" in obj and (
                "prompt" in obj or "prompt_messages" in obj
            )
            if is_orpo:
                errs = validate_orpo_pair(obj, tokenizer=tokenizer, max_tokens=min(max_tokens, 4096))
                if errs:
                    for e in errs:
                        rejection_reasons[e] += 1
                    rejected_rows.append({"id": obj.get("id") or f"L{line_no}", "errors": errs})
                else:
                    valid_orpo.append(obj)
            else:
                errs = validate_sft_sample(obj, tokenizer=tokenizer, max_tokens=max_tokens)
                if errs:
                    for e in errs:
                        rejection_reasons[e] += 1
                    rejected_rows.append({"id": obj.get("id") or f"L{line_no}", "errors": errs})
                else:
                    valid_sft.append(obj)

    dedup_sft = deduplicate_samples(valid_sft)
    dedup_orpo = deduplicate_samples(valid_orpo)

    for p in (Path(out_sft_path), Path(out_orpo_path), Path(out_rejected_path)):
        p.parent.mkdir(parents=True, exist_ok=True)

    sft_buckets: Counter[str] = Counter()
    orpo_buckets: Counter[str] = Counter()
    thinking_true_count = 0
    thinking_false_count = 0
    max_reasoning_chars = 0

    with Path(out_sft_path).open("w", encoding="utf-8") as f:
        for sample in dedup_sft:
            msgs = sample["messages"]
            mode = _infer_mode(sample, msgs)
            role, is_worker = _infer_role_and_worker(sample, msgs, mode)
            thinking = _infer_thinking_flag(sample, default=True)
            if thinking:
                thinking_true_count += 1
            else:
                thinking_false_count += 1

            for m in msgs:
                r_txt = str(m.get("reasoning_content") or m.get("reasoning") or "")
                if len(r_txt) > max_reasoning_chars:
                    max_reasoning_chars = len(r_txt)

            bucket = str(sample.get("bucket", "unknown"))
            sft_buckets[bucket] += 1
            rendered = render_gemma4_training_sample(
                msgs,
                tools=_resolve_tools_for_sample(
                    sample, mode=mode, role=role, is_worker=is_worker, messages=msgs
                ),
                enable_thinking=thinking,
            )
            record = {
                "id": sample.get("id"),
                "bucket": bucket,
                "per_stage": sample.get("per_stage"),
                "mode": mode,
                "role": role,
                "thinking": thinking,
                "ctx_profile": sample.get("ctx_profile", "8k"),
                "text": rendered["text"],
                "assistant_spans": rendered["assistant_spans"],
                "messages": msgs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with Path(out_orpo_path).open("w", encoding="utf-8") as f:
        for pair in dedup_orpo:
            prompt_msgs = pair.get("prompt_messages") or pair["prompt"]
            mode = _infer_mode(pair, prompt_msgs)
            role, is_worker = _infer_role_and_worker(pair, prompt_msgs, mode)
            chosen_thought = str(pair["chosen"].get("reasoning_content") or "").strip()
            if len(chosen_thought) > max_reasoning_chars:
                max_reasoning_chars = len(chosen_thought)
            thinking = _infer_thinking_flag(pair, default=bool(chosen_thought))
            bucket = str(pair.get("bucket") or pair.get("failure_type") or "unknown")
            orpo_buckets[bucket] += 1

            rendered = render_gemma4_orpo_pair(
                prompt_msgs,
                pair["chosen"],
                pair["rejected"],
                tools=_resolve_tools_for_sample(
                    pair, mode=mode, role=role, is_worker=is_worker, messages=prompt_msgs
                ),
                enable_thinking=thinking,
            )
            record = {
                "id": pair.get("id"),
                "bucket": bucket,
                "failure_type": pair.get("failure_type") or bucket,
                "mode": mode,
                "thinking": thinking,
                "prompt": rendered["prompt"],
                "chosen": rendered["chosen"],
                "rejected": rendered["rejected"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with Path(out_rejected_path).open("w", encoding="utf-8") as f:
        for rej in rejected_rows:
            f.write(json.dumps(rej, ensure_ascii=False) + "\n")

    total_sft = thinking_true_count + thinking_false_count
    thinking_ratio = round(thinking_true_count / total_sft, 4) if total_sft else 0.0
    return {
        "total_lines": total_lines,
        "valid_sft_before_dedup": len(valid_sft),
        "kept_sft": len(dedup_sft),
        "valid_orpo_before_dedup": len(valid_orpo),
        "kept_orpo": len(dedup_orpo),
        "rejected_count": len(rejected_rows),
        "thinking_true_count": thinking_true_count,
        "thinking_false_count": thinking_false_count,
        "thinking_true_ratio": thinking_ratio,
        "thinking_ratio_ok": thinking_ratio <= max_thinking_ratio,
        "max_reasoning_chars": max_reasoning_chars,
        "sft_bucket_counts": dict(sft_buckets),
        "orpo_bucket_counts": dict(orpo_buckets),
        "top_rejection_reasons": dict(rejection_reasons.most_common(10)),
    }

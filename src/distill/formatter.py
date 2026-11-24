"""Exact Gemma 4 training serializer matching HF chat_template.jinja + NativeTools.

Mirrors the official Gemma 4 Jinja template (haku-nekooo/gemma4-e2b-nuclear-pruned /
google/gemma-4-E2B-it) including <|tool>, <|tool_call>, <|tool_response>, and
continue_same_model_turn while preserving <|channel>thought...\n<channel|> labels.
"""

from __future__ import annotations

import json
from typing import Any

BOS_TOKEN = "<bos>"
TURN_START = "<|turn>"
TURN_END = "<turn|>"
THINK_TOKEN = "<|think|>\n"
THOUGHT_OPEN = "<|channel>thought\n"
THOUGHT_CLOSE = "<channel|>"
TOOL_DECL_OPEN = "<|tool>"
TOOL_DECL_CLOSE = "<tool|>"
TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"
TOOL_RESP_OPEN = "<|tool_response>"
TOOL_RESP_CLOSE = "<tool_response|>"
STR_QUOTE = '<|"|>'


def format_gemma4_argument(argument: Any, escape_keys: bool = True) -> str:
    """Mirror Gemma 4 Jinja `format_argument(argument, escape_keys)`."""
    if argument is None:
        return "null"
    if isinstance(argument, bool):
        return "true" if argument else "false"
    if isinstance(argument, str):
        return f"{STR_QUOTE}{argument}{STR_QUOTE}"
    if isinstance(argument, dict):
        parts = []
        for k, v in sorted(argument.items()):
            key_str = f"{STR_QUOTE}{k}{STR_QUOTE}" if escape_keys else str(k)
            parts.append(f"{key_str}:{format_gemma4_argument(v, escape_keys=escape_keys)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(argument, (list, tuple)):
        return "[" + ",".join(format_gemma4_argument(item, escape_keys=escape_keys) for item in argument) + "]"
    return str(argument)


def format_gemma4_parameters(
    properties: dict[str, Any],
    required: list[str] | None = None,
    filter_keys: bool = False,
) -> str:
    """Mirror Gemma 4 Jinja `format_parameters(properties, required)`."""
    standard_keys = {"description", "type", "properties", "required", "nullable"}
    out: list[str] = []
    for key, value in sorted(properties.items()):
        if filter_keys and key in standard_keys:
            continue
        fields: list[str] = []
        if value.get("description"):
            fields.append(f"description:{STR_QUOTE}{value['description']}{STR_QUOTE}")
        vtype = str(value.get("type", "")).upper()
        if vtype == "STRING" and value.get("enum"):
            fields.append(f"enum:{format_gemma4_argument(value['enum'])}")
        elif vtype == "ARRAY" and isinstance(value.get("items"), dict) and value["items"]:
            item_parts: list[str] = []
            items = value["items"]
            for ik, iv in sorted(items.items()):
                if iv is None:
                    continue
                if ik == "properties":
                    inner = format_gemma4_parameters(iv, items.get("required") or []) if isinstance(iv, dict) else ""
                    item_parts.append(f"properties:{{{inner}}}")
                elif ik == "required":
                    reqs = ",".join(f"{STR_QUOTE}{r}{STR_QUOTE}" for r in iv)
                    item_parts.append(f"required:[{reqs}]")
                elif ik == "type":
                    tval = iv.upper() if isinstance(iv, str) else [str(x).upper() for x in iv]
                    item_parts.append(f"type:{format_gemma4_argument(tval)}")
                else:
                    item_parts.append(f"{ik}:{format_gemma4_argument(iv)}")
            fields.append("items:{" + ",".join(item_parts) + "}")
        if value.get("nullable"):
            fields.append("nullable:true")
        if vtype == "OBJECT":
            if isinstance(value.get("properties"), dict):
                inner = format_gemma4_parameters(value["properties"], value.get("required") or [])
                fields.append(f"properties:{{{inner}}}")
            elif isinstance(value, dict):
                inner = format_gemma4_parameters(value, value.get("required") or [], filter_keys=True)
                fields.append(f"properties:{{{inner}}}")
            if value.get("required"):
                reqs = ",".join(f"{STR_QUOTE}{r}{STR_QUOTE}" for r in value["required"])
                fields.append(f"required:[{reqs}]")
        fields.append(f"type:{STR_QUOTE}{vtype}{STR_QUOTE}")
        out.append(f"{key}:{{{','.join(fields)}}}")
    return ",".join(out)


def format_gemma4_tool_declaration(tool_data: dict[str, Any]) -> str:
    """Mirror Gemma 4 Jinja `format_function_declaration(tool_data)`."""
    fn = tool_data["function"]
    buf = [f"declaration:{fn['name']}{{description:{STR_QUOTE}{fn.get('description', '')}{STR_QUOTE}"]
    params = fn.get("parameters")
    if params:
        p_parts: list[str] = []
        if params.get("properties"):
            p_parts.append(
                f"properties:{{{format_gemma4_parameters(params['properties'], params.get('required'))}}}"
            )
        if params.get("required"):
            reqs = ",".join(f"{STR_QUOTE}{item}{STR_QUOTE}" for item in params["required"])
            p_parts.append(f"required:[{reqs}]")
        if params.get("type"):
            p_parts.append(f"type:{STR_QUOTE}{str(params['type']).upper()}{STR_QUOTE}")
        buf.append(f",parameters:{{{','.join(p_parts)}}}")
    if "response" in fn:
        resp_decl = fn["response"]
        r_parts: list[str] = []
        if resp_decl.get("description"):
            r_parts.append(f"description:{STR_QUOTE}{resp_decl['description']}{STR_QUOTE}")
        if str(resp_decl.get("type", "")).upper() == "OBJECT":
            r_parts.append(f"type:{STR_QUOTE}OBJECT{STR_QUOTE}")
        buf.append(f",response:{{{','.join(r_parts)}}}")
    buf.append("}")
    return "".join(buf)


def format_gemma4_tool_call(tool_call: dict[str, Any]) -> str:
    """Mirror Gemma 4 Jinja `<|tool_call>call:name{...}<tool_call|>` serialization."""
    fn = tool_call["function"] if isinstance(tool_call.get("function"), dict) else tool_call
    name = fn["name"]
    raw_args = fn.get("arguments")
    if isinstance(raw_args, str):
        stripped = raw_args.strip()
        try:
            raw_args = json.loads(stripped) if stripped else {}
        except json.JSONDecodeError:
            inner = stripped[1:-1] if stripped.startswith("{") and stripped.endswith("}") else stripped
            return f"{TOOL_CALL_OPEN}call:{name}{{{inner}}}{TOOL_CALL_CLOSE}"
    if isinstance(raw_args, dict):
        inner = ",".join(
            f"{k}:{format_gemma4_argument(v, escape_keys=False)}" for k, v in sorted(raw_args.items())
        )
    else:
        inner = ""
    return f"{TOOL_CALL_OPEN}call:{name}{{{inner}}}{TOOL_CALL_CLOSE}"


def format_gemma4_tool_response(tool_name: str, response: Any) -> str:
    """Mirror Gemma 4 Jinja `format_tool_response_block(tool_name, response)`."""
    if isinstance(response, dict):
        inner = ",".join(
            f"{k}:{format_gemma4_argument(v, escape_keys=False)}" for k, v in sorted(response.items())
        )
        return f"{TOOL_RESP_OPEN}response:{tool_name}{{{inner}}}{TOOL_RESP_CLOSE}"
    return f"{TOOL_RESP_OPEN}response:{tool_name}{{value:{format_gemma4_argument(str(response), escape_keys=False)}}}{TOOL_RESP_CLOSE}"


def _extract_thought(msg: dict[str, Any]) -> str:
    val = msg.get("reasoning") or msg.get("reasoning_content") or msg.get("thought") or ""
    return str(val).strip()


def _extract_tool_call_name(tc: dict[str, Any]) -> str:
    if isinstance(tc.get("function"), dict):
        return str(tc["function"].get("name", "unknown"))
    return str(tc.get("name", "unknown"))


def render_gemma4_training_sample(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = True,
    include_bos: bool = True,
    add_generation_prompt: bool = False,
) -> dict[str, Any]:
    """Render Gemma 4 trajectory preserving thoughts + assistant loss spans."""
    if not messages:
        raise ValueError("messages must not be empty")

    buf: list[str] = []
    pos = 0

    def emit(chunk: str) -> None:
        nonlocal pos
        buf.append(chunk)
        pos += len(chunk)

    if include_bos:
        emit(BOS_TOKEN)

    loop_messages = list(messages)
    if enable_thinking or tools or messages[0].get("role") in ("system", "developer"):
        emit(f"{TURN_START}system\n")
        if enable_thinking:
            emit(THINK_TOKEN)
        if messages[0].get("role") in ("system", "developer"):
            emit(str(messages[0].get("content", "")).strip())
            loop_messages = loop_messages[1:]
        if tools:
            for tool in tools:
                emit(f"{TOOL_DECL_OPEN}{format_gemma4_tool_declaration(tool).strip()}{TOOL_DECL_CLOSE}")
        emit(f"{TURN_END}\n")

    if not loop_messages:
        raise ValueError("conversation must include at least one user/model turn")

    assistant_spans: list[tuple[int, int]] = []
    prev_non_tool_role: str | None = None
    prev_message_type: str | None = None

    for idx, msg in enumerate(loop_messages):
        raw_role = msg.get("role")
        if raw_role == "tool":
            continue

        prev_message_type = None
        role = "model" if raw_role in ("assistant", "model") else raw_role
        if role not in ("user", "model"):
            raise ValueError(f"Unsupported role in training sample: {raw_role!r}")

        continue_same_model_turn = role == "model" and prev_non_tool_role in ("assistant", "model")
        if not continue_same_model_turn:
            emit(f"{TURN_START}{role}\n")

        if role == "user":
            emit(f"{str(msg.get('content', '')).strip()}{TURN_END}\n")
            prev_non_tool_role = raw_role
            continue

        # Model/assistant turn: collect model-generated span separately from <|tool_response>.
        span_start: int | None = pos

        thought = _extract_thought(msg)
        if enable_thinking and thought:
            emit(f"{THOUGHT_OPEN}{thought}\n{THOUGHT_CLOSE}")

        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            for tc in tool_calls:
                emit(format_gemma4_tool_call(tc))
            prev_message_type = "tool_call"

        # Fold following consecutive role:tool messages (exactly as Jinja template does).
        tr_out = False
        if tool_calls:
            k = idx + 1
            while k < len(loop_messages) and loop_messages[k].get("role") == "tool":
                if span_start is not None and pos > span_start:
                    assistant_spans.append((span_start, pos))
                    span_start = None
                follow = loop_messages[k]
                tname = str(follow.get("name") or "unknown")
                for tc in tool_calls:
                    if tc.get("id") and tc.get("id") == follow.get("tool_call_id"):
                        tname = _extract_tool_call_name(tc)
                        break
                emit(format_gemma4_tool_response(tname, follow.get("content", "")))
                tr_out = True
                prev_message_type = "tool_response"
                k += 1

        content_str = str(msg.get("content") or "").strip()
        if content_str:
            if span_start is None:
                span_start = pos
            emit(content_str)

        # Forward-scan next non-tool role.
        next_nt_role: str | None = None
        for j in range(idx + 1, len(loop_messages)):
            if loop_messages[j].get("role") != "tool":
                next_nt_role = loop_messages[j].get("role")
                break

        continues_into_next = (
            next_nt_role in ("assistant", "model") and (not tool_calls or tr_out)
        )

        if prev_message_type == "tool_call" and not tr_out:
            if span_start is not None and pos > span_start:
                assistant_spans.append((span_start, pos))
                span_start = None
            emit(TOOL_RESP_OPEN)
        elif continues_into_next:
            if span_start is not None and pos > span_start:
                assistant_spans.append((span_start, pos))
                span_start = None
        elif not (tr_out and not content_str and next_nt_role is None):
            if span_start is None:
                span_start = pos
            emit(TURN_END)
            assistant_spans.append((span_start, pos))
            span_start = None
            emit("\n")
        elif span_start is not None and pos > span_start:
            assistant_spans.append((span_start, pos))

        prev_non_tool_role = raw_role

    if add_generation_prompt:
        if prev_message_type not in ("tool_response", "tool_call"):
            emit(f"{TURN_START}model\n")
        elif prev_message_type == "tool_response" and enable_thinking:
            emit(THOUGHT_OPEN)

    return {"text": "".join(buf), "assistant_spans": assistant_spans}


def render_gemma4_orpo_pair(
    prompt_messages: list[dict[str, Any]],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = True,
    include_bos: bool = True,
) -> dict[str, str]:
    """Render (prompt, chosen, rejected) strings ready for TRL ORPOTrainer."""
    if not prompt_messages:
        raise ValueError("prompt_messages must not be empty")

    prompt_res = render_gemma4_training_sample(
        prompt_messages,
        tools=tools,
        enable_thinking=enable_thinking,
        include_bos=include_bos,
        add_generation_prompt=False,
    )
    prompt_prefix = prompt_res["text"]
    last_non_tool = next(
        (m for m in reversed(prompt_messages) if m.get("role") != "tool"),
        None,
    )
    last_raw = prompt_messages[-1].get("role")
    if last_raw != "tool" and not (last_non_tool and last_non_tool.get("role") in ("assistant", "model")):
        prompt_prefix += f"{TURN_START}model\n"

    def _render_candidate(cand: dict[str, Any]) -> str:
        parts: list[str] = []
        thought = _extract_thought(cand)
        if enable_thinking and thought:
            parts.append(f"{THOUGHT_OPEN}{thought}\n{THOUGHT_CLOSE}")
        tool_calls = cand.get("tool_calls") or []
        for tc in tool_calls:
            parts.append(format_gemma4_tool_call(tc))
        body = str(cand.get("content") or "").strip()
        if body:
            parts.append(body)
        if not tool_calls:
            parts.append(TURN_END + "\n")
        return "".join(parts)

    return {
        "prompt": prompt_prefix,
        "chosen": _render_candidate(chosen),
        "rejected": _render_candidate(rejected),
    }


def validate_gemma4_wire_format(
    text: str,
    allow_partial_tool_turn: bool = False,
) -> list[str]:
    """Verify balanced Gemma 4 turn, thought, tool, and response tags + zero XML <thinking>."""
    errors: list[str] = []
    if "<thinking>" in text or "</thinking>" in text:
        errors.append("forbidden custom XML <thinking> tag found; use <|channel>thought...<channel|>")

    checks = (
        ("turn", TURN_START, TURN_END, True),
        ("thought", "<|channel>thought", THOUGHT_CLOSE, False),
        ("tool_decl", TOOL_DECL_OPEN, TOOL_DECL_CLOSE, False),
        ("tool_call", TOOL_CALL_OPEN, TOOL_CALL_CLOSE, False),
        ("tool_resp", TOOL_RESP_OPEN, TOOL_RESP_CLOSE, False),
    )
    for label, open_tok, close_tok, require_nonzero in checks:
        n_open = text.count(open_tok)
        n_close = text.count(close_tok)
        if require_nonzero and n_open == 0:
            errors.append(f"missing {label} markers ({open_tok})")
        if label == "turn" and allow_partial_tool_turn and n_open == n_close + 1:
            continue
        if n_open != n_close:
            errors.append(f"unbalanced {label} markers: {open_tok}={n_open}, {close_tok}={n_close}")
    return errors

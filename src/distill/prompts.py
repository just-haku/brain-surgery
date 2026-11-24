"""Gemini prompt builders and deterministic bucket/model routing for NativeTools."""

from __future__ import annotations

import json
import random
from typing import Any

from .nc_schema import NC_CAVEMAN_BASE, NC_ROLE_GUIDES, NC_TOOL_SPECS

MODEL_PRO = "gemini-2.5-pro"
MODEL_FLASH = "gemini-2.5-flash"

BUCKET_SPECS: dict[str, dict[str, Any]] = {
    "A_nc_agent": {
        "ratio": 0.35,
        "model": MODEL_PRO,
        "temperature": 0.4,
        "mode": "nc_agent",
    },
    "B_bilingual_debug": {
        "ratio": 0.25,
        "model": MODEL_FLASH,
        "temperature": 0.5,
        "mode": "chat",
    },
    "C_direct_code": {
        "ratio": 0.15,
        "model": MODEL_PRO,
        "temperature": 0.35,
        "mode": "chat",
    },
    "D_clarify": {
        "ratio": 0.15,
        "model": MODEL_FLASH,
        "temperature": 0.45,
        "mode": "mixed",
    },
    "E_dev_chat": {
        "ratio": 0.10,
        "model": MODEL_FLASH,
        "temperature": 0.7,
        "mode": "chat",
    },
}

_LANGUAGES = ("Rust", "Python", "C++", "TypeScript", "SQL", "HTML/CSS")

_NC_SCENARIOS = (
    "context_isolation_switch_file: Turn 1 inspects file A; new ## TASK switches to symbol in file B. Model must call read_symbol/search_text on file B and never bleed file A state.",
    "scratchpad_eviction_respect: ## SCRATCHPAD already has [step 1 read_symbol -> sym ...] or cached snippet. Model must NOT re-read cached symbol and must proceed to edit_file or plain-text answer.",
    "edit_file_strict_args: Bug fix via native tool_calls edit_file with exact path, target, replacement (NEVER start_line/end_line/content/old_str), preserving Rust i32 / C++ int32_t / TS strict types.",
    "multi_turn_native_trajectory: Full 2-3 hop trajectory: assistant(tool_calls: search_text) -> tool -> assistant(tool_calls: read_symbol) -> tool -> assistant(tool_calls: edit_file) -> tool -> assistant(final plain text answer with zero raw JSON leak).",
    "planner_decomposition: Role PLANNER outputting crisp 2-4 micro-tasks JSON plan without unnecessary chatter.",
)

_ORPO_FAILURES = (
    ("context_bleed", "rejected mixes symbol/variable from previous file or repeats read already present in ## SCRATCHPAD"),
    ("tool_schema_or_json_leak", "rejected passes start_line/content to edit_file, wraps tool call as raw JSON string inside content, or nests JSON in prose"),
    ("type_mutation", "rejected silently mutates Rust i32 to f64 or weakens TypeScript strict type instead of fixing logic"),
    ("repetition_or_emoji", "rejected repeats sentences/n-grams in reasoning_content/content or leaks Unicode emoji"),
)

_VI_TOPICS = (
    "borrow checker và lifetimes trong Rust",
    "con trỏ thông minh (unique_ptr/shared_ptr) và memory safety trong C++",
    "asyncio event loop, deadlock và race condition trong Python",
    "tối ưu truy vấn SQL index và EXPLAIN ANALYZE",
    "TypeScript discriminated unions và strict type narrowing",
    "quản lý VRAM KV cache và quantization cho LLM chạy local",
)

_POLICY_RULES = """STRICT GLOBAL RULES:
1. Return ONLY a valid JSON object matching the exact output schema below (no markdown wrappers).
2. ZERO Unicode emoji, pictographs, dingbats, ZWJ (U+200D), or variation selectors (U+FE0F) anywhere in ANY field.
3. ASCII emoticons (:), :))), ._., .-., =))) are FORBIDDEN inside 'reasoning_content', FORBIDDEN inside code blocks, FORBIDDEN inside tool_calls, and FORBIDDEN in nc_agent outputs. Only Bucket E chat prose may use 0-2 ASCII emoticons naturally.
4. Keep 'reasoning_content' concise (2-5 short bullet lines, <= 180 words) with zero repetitive phrasing.
5. In Vietnamese explanations, write natural Vietnamese and keep English technical terms verbatim (borrow checker, lifetime, mutex, KV cache, schema, etc.)."""


def _nc_schema_summary() -> str:
    compact = {
        name: {
            "required": list(spec["required"]),
            "optional": list(spec["optional"]),
        }
        for name, spec in NC_TOOL_SPECS.items()
    }
    return json.dumps(compact, ensure_ascii=False)


def _build_sft_prompt(
    sample_id: str,
    bucket: str,
    language: str,
    scenario: str,
    enable_thinking: bool,
    role: str,
) -> str:
    if bucket == "A_nc_agent" or (bucket == "D_clarify" and role == "coder"):
        mode = "nc_agent"
        sys_prompt = f"{NC_CAVEMAN_BASE}{NC_ROLE_GUIDES[role]}"
        schema_msg = """OUTPUT JSON SCHEMA (NativeTools multi-turn trajectory):
{
  "id": "__SAMPLE_ID__",
  "bucket": "__BUCKET__",
  "mode": "nc_agent",
  "role": "__ROLE__",
  "enable_thinking": __THINK_BOOL__,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "## REPO_MAP\\n...\\n\\n## TASK\\n..."},
    {
      "role": "assistant",
      "reasoning_content": "__REASON_HINT__",
      "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "read_symbol", "arguments": {"path": "...", "symbol": "..."}}}
      ],
      "content": ""
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "read_symbol", "content": "L10-L15:\\n..."},
    {
      "role": "assistant",
      "reasoning_content": "__REASON_HINT__",
      "content": "Concise plain-text engineering answer (NO JSON wrapper)."
    }
  ]
}"""
        extra = (
            f"Use Native OpenAI tool_calls + role='tool' responses (NEVER wrap tool calls inside JSON strings in 'content').\n"
            f"Use nc-agent user prompt anchors (## REPO_MAP, ## LEDGER, ## TASK, optionally ## SCRATCHPAD and ## NEXT).\n"
            f"Allowed tools & exact args: {_nc_schema_summary()}.\n"
            f"If Role=='planner', omit tool_calls and output clean micro-task JSON in 'content'.\n"
            f"Scenario focus: {scenario}."
        )
    else:
        mode = "chat"
        sys_prompt = (
            "Bạn là trợ lý lập trình song ngữ Anh–Việt cho nuclear-code. "
            "Giải thích súc tích, giữ nguyên thuật ngữ tiếng Anh chuyên ngành và giữ nguyên kiểu dữ liệu gốc khi sửa code."
        )
        schema_msg = """OUTPUT JSON SCHEMA:
{
  "id": "__SAMPLE_ID__",
  "bucket": "__BUCKET__",
  "mode": "chat",
  "role": "__ROLE__",
  "enable_thinking": __THINK_BOOL__,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "reasoning_content": "__REASON_HINT__", "content": "..."}
  ]
}"""
        if bucket == "E_dev_chat":
            extra = (
                "Natural Vietnamese/English developer chat about coding workflow or debugging. "
                "May include 1-2 simple ASCII emoticons (e.g. ':)', ':)))', '._.', '.-.', '=))') ONLY in plain conversational prose outside code blocks."
            )
        elif bucket == "D_clarify":
            extra = "User request omits file path or ambiguous signature. Assistant must refuse to guess and ask 1 crisp clarifying question."
        elif bucket == "C_direct_code":
            extra = f"Single-turn direct {language} code fix/refactor preserving original static types (e.g. i32, int32_t, strict TS) + concise edge-case check."
        else:
            extra = f"Bilingual EN-VI debugging/explanation on {language}: {scenario}."

    schema_rendered = (
        schema_msg.replace("__SAMPLE_ID__", sample_id)
        .replace("__BUCKET__", bucket)
        .replace("__ROLE__", role)
        .replace("__THINK_BOOL__", str(enable_thinking).lower())
        .replace("__REASON_HINT__", "..." if enable_thinking else "")
    )

    return f"""Generate 1 high-quality synthetic training example for a compact 2B coding model.

{_POLICY_RULES}

TASK PARAMETERS:
- id: "{sample_id}"
- bucket: "{bucket}"
- mode: "{mode}"
- role: "{role}"
- enable_thinking: {str(enable_thinking).lower()}
- language: "{language}"
- system_content_hint: {json.dumps(sys_prompt, ensure_ascii=False)}
- guidance: {extra}

{schema_rendered}"""


def _build_orpo_prompt(
    pair_id: str,
    failure_type: str,
    failure_desc: str,
    language: str,
    enable_thinking: bool,
) -> str:
    mode = "chat" if failure_type == "repetition_or_emoji" and hash(pair_id) % 2 == 0 else "nc_agent"
    role = "coder"
    return f"""Generate 1 high-signal ORPO preference pair (chosen vs rejected) for a compact 2B coding assistant using NativeTools.

{_POLICY_RULES}

TASK PARAMETERS:
- id: "{pair_id}"
- failure_type: "{failure_type}"
- mode: "{mode}"
- role: "{role}"
- enable_thinking: {str(enable_thinking).lower()}
- language: "{language}"
- allowed_nc_tools: {_nc_schema_summary()}
- chosen requirement: spotless NativeTools assistant turn (`tool_calls` with exact schema args OR concise plain text `content`), preserves exact types, zero repetition, zero Unicode emoji.
- rejected requirement: exhibits realistic mistake -> {failure_desc}.

OUTPUT JSON SCHEMA:
{{
  "id": "{pair_id}",
  "bucket": "ORPO_pairs",
  "failure_type": "{failure_type}",
  "mode": "{mode}",
  "role": "{role}",
  "enable_thinking": {str(enable_thinking).lower()},
  "prompt_messages": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}}
  ],
  "chosen": {{"reasoning_content": {"\"...\"" if enable_thinking else "\"\""}, "tool_calls": [], "content": "..."}},
  "rejected": {{"reasoning_content": "...", "tool_calls": [], "content": "..."}}
}}"""


def generate_seed_tasks(
    total_sft: int = 4000,
    total_orpo: int = 600,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build deterministic Gemini Batch task specs routed across Pro and Flash."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []

    bucket_names = list(BUCKET_SPECS.keys())
    counts: dict[str, int] = {}
    assigned = 0
    for name in bucket_names[:-1]:
        c = int(round(total_sft * BUCKET_SPECS[name]["ratio"]))
        counts[name] = c
        assigned += c
    counts[bucket_names[-1]] = max(0, total_sft - assigned)

    seq = 0
    for bucket, count in counts.items():
        spec = BUCKET_SPECS[bucket]
        for i in range(count):
            seq += 1
            lang = _LANGUAGES[(seq + i) % len(_LANGUAGES)]
            enable_thinking = (i % 10) < 7  # ~70% thinking on, ~30% thinking off
            if bucket == "A_nc_agent":
                role = "planner" if (i % 10 == 9) else "coder"
                scenario = _NC_SCENARIOS[i % len(_NC_SCENARIOS)]
            elif bucket == "D_clarify":
                role = "coder" if (i % 2 == 0) else "chat"
                scenario = "missing_symbol_or_ambiguous_spec"
            else:
                role = "chat"
                scenario = _VI_TOPICS[(seq + i) % len(_VI_TOPICS)]

            key = f"sft__{bucket}__{seq:05d}"
            tasks.append(
                {
                    "key": key,
                    "kind": "sft",
                    "bucket": bucket,
                    "model": spec["model"],
                    "temperature": spec["temperature"],
                    "prompt_text": _build_sft_prompt(
                        sample_id=key,
                        bucket=bucket,
                        language=lang,
                        scenario=scenario,
                        enable_thinking=enable_thinking,
                        role=role,
                    ),
                }
            )

    for j in range(total_orpo):
        failure_type, failure_desc = _ORPO_FAILURES[j % len(_ORPO_FAILURES)]
        lang = _LANGUAGES[j % len(_LANGUAGES)]
        enable_thinking = (j % 5) != 0
        key = f"orpo__{failure_type}__{j + 1:05d}"
        tasks.append(
            {
                "key": key,
                "kind": "orpo",
                "bucket": "ORPO_pairs",
                "model": MODEL_PRO,
                "temperature": 0.45,
                "prompt_text": _build_orpo_prompt(
                    pair_id=key,
                    failure_type=failure_type,
                    failure_desc=failure_desc,
                    language=lang,
                    enable_thinking=enable_thinking,
                ),
            }
        )

    rng.shuffle(tasks)
    return tasks

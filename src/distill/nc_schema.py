"""Exact nc-agent tool schemas (NativeTools OpenAI format + prompt layout)."""

from __future__ import annotations

import json
from typing import Any

# Mirror of D:\Coding\nuclear_code\crates\nc-agent\src\{tools,orchestrator,graph,context}.rs.
NC_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "read_symbol": {
        "description": "Read exact source body of a function, struct, impl, or class symbol (<= 90 lines).",
        "required": ("path", "symbol"),
        "optional": (),
        "types": {"path": str, "symbol": str},
        "param_desc": {
            "path": "Workspace-relative file path.",
            "symbol": "Exact symbol name to inspect.",
        },
        "non_empty": ("path", "symbol"),
        "forbidden": ("start_line", "end_line", "content"),
    },
    "search_text": {
        "description": "Search workspace text or symbols across files (case-insensitive UTF-8 scan).",
        "required": ("query",),
        "optional": ("ext", "max_hits"),
        "types": {"query": str, "ext": str, "max_hits": int},
        "param_desc": {
            "query": "Exact text or identifier to search for.",
            "ext": "Optional file extension filter without dot, e.g. rs or py.",
            "max_hits": "Maximum number of search matches to return (>= 1, default 15, cap 100).",
        },
        "non_empty": ("query",),
        "positive_ints": ("max_hits",),
        "forbidden": ("path", "symbol"),
    },
    "edit_file": {
        "description": (
            "Replace an exact unique substring in a file (matches == 1). "
            "Always read exact text first; never pass start_line, end_line, or full content."
        ),
        "required": ("path", "target", "replacement"),
        "optional": (),
        "types": {"path": str, "target": str, "replacement": str},
        "param_desc": {
            "path": "Workspace-relative file path.",
            "target": "Exact existing verbatim code snippet to replace.",
            "replacement": "Replacement snippet preserving existing types unless explicitly asked.",
        },
        "non_empty": ("path", "target"),
        "forbidden": ("start_line", "end_line", "line", "content", "new_content", "old_str", "new_str"),
    },
    "read_outline": {
        "description": "List top-level symbols and line ranges in a source file (up to 45 symbols).",
        "required": ("path",),
        "optional": (),
        "types": {"path": str},
        "param_desc": {"path": "Workspace-relative file path."},
        "non_empty": ("path",),
        "forbidden": (),
    },
    "read_slice": {
        "description": "Read a bounded 1-indexed inclusive line range (<= 120 lines) from a file.",
        "required": ("path", "start_line", "end_line"),
        "optional": (),
        "types": {"path": str, "start_line": int, "end_line": int},
        "param_desc": {
            "path": "Workspace-relative file path.",
            "start_line": "Required 1-based start line number.",
            "end_line": "Required 1-based inclusive end line number (>= start_line).",
        },
        "non_empty": ("path",),
        "positive_ints": ("start_line", "end_line"),
        "forbidden": (),
    },
    "list_files": {
        "description": "List workspace files optionally filtered to a directory.",
        "required": (),
        "optional": ("dir", "max_entries"),
        "types": {"dir": str, "max_entries": int},
        "param_desc": {
            "dir": "Optional subdirectory path relative to workspace root.",
            "max_entries": "Maximum number of entries to list (>= 1, default 60).",
        },
        "non_empty": (),
        "positive_ints": ("max_entries",),
        "forbidden": (),
    },
    "write_file": {
        "description": "Create a new file or overwrite an entire file inside the workspace.",
        "required": ("path", "content"),
        "optional": (),
        "types": {"path": str, "content": str},
        "param_desc": {
            "path": "Workspace-relative file path.",
            "content": "Full file contents to write.",
        },
        "non_empty": ("path",),
        "forbidden": ("target", "replacement"),
    },
    "run_command": {
        "description": "Run a workspace build, check, or test command (output capped at 4,096 bytes).",
        "required": ("cmd",),
        "optional": (),
        "types": {"cmd": str},
        "param_desc": {"cmd": "Command string to execute."},
        "non_empty": ("cmd",),
        "forbidden": (),
    },
    "spawn_subagent": {
        "description": "Spawn a depth-1 sequential worker subagent (Lead only) and receive a compact SubagentReport.",
        "required": ("role", "task", "files"),
        "optional": (),
        "types": {"role": str, "task": str, "files": list},
        "param_desc": {
            "role": "Worker role: explorer (read-only), coder, or reviewer.",
            "task": "Self-contained micro-task description for the worker.",
            "files": "Workspace-relative file paths for focused context (pass [] if none).",
        },
        "non_empty": ("role", "task"),
        "enum": {"role": ("explorer", "coder", "reviewer")},
        "forbidden": (),
    },
}

FINAL_STEP_PROMPT = "[FINAL STEP] Answer TASK concisely in plain text now; do not call tools."

DEFAULT_CODER_TOOLS: tuple[str, ...] = tuple(NC_TOOL_SPECS.keys())
WORKER_CODER_TOOLS: tuple[str, ...] = tuple(
    name for name in DEFAULT_CODER_TOOLS if name != "spawn_subagent"
)
EXPLORER_TOOLS: tuple[str, ...] = (
    "list_files",
    "search_text",
    "read_outline",
    "read_symbol",
    "read_slice",
)

NC_CAVEMAN_BASE = (
    "# nuclear_code Low-CtxWin Agent System Prompt (Caveman Mode)\n\n"
    "You are `nuclear_code`, an offline local coding agent running on bounded context (`8K`-`64K`).\n"
    "Work fast, keep answers terse, preserve every technical fact.\n\n"
    "## Communication Rules (Caveman Compression)\n"
    "1. Answer first, then reason, then next step. Zero greetings, filler, hedging, or recap.\n"
    "2. Short direct words. One idea per sentence. Never drop negations (`not`, `never`, `no`, `only`).\n"
    "3. Code, commands, file paths, symbol names, numbers, and error lines stay **byte-exact**.\n"
    "4. Auto-clarity exception: switch to full clear sentences for security warnings, "
    "irreversible deletes/migrations, or when asking the user for confirmation.\n\n"
    "## Low-Context Discipline\n"
    "1. Trust `Repo Map` and `State Ledger` first; never ask to dump whole files when line ranges work.\n"
    "2. Request bounded slices (`start_line..end_line`). Expect older tool outputs to be evicted to 1-line receipts.\n"
    "3. Prefer smallest complete diff (YAGNI). Do not rewrite untouched code or remove existing comments.\n"
    "4. Emit tool calls strictly matching the active JSON/GBNF tool schema. Never wrap tool JSON in prose.\n"
    "5. Verify edits with the project check/test command before marking a micro-task `done`."
)

_CODER_DELEGATE_CLAUSE = (
    " Delegate multi-file exploration/edits via spawn_subagent(role,task,files) "
    "(depth-1 sequential queue) when helpful."
)

NC_ROLE_GUIDES: dict[str, str] = {
    "planner": (
        '\n\nRole: PLANNER. Break goal into <= 4 micro-tasks. JSON ONLY: '
        '{"goal":"...","decisions":[...],"tasks":[{"id":"T1","title":"...","target_files":["..."],"verify_cmd":null}]}'
    ),
    "coder": (
        "\n\nRole: CODER. Use native tools (read_outline/read_symbol before read_slice). "
        "Always read exact text before edit_file(path,target,replacement); keep original types/comments. "
        f"Never repeat reads.{_CODER_DELEGATE_CLAUSE} Once facts are ready, reply concisely in plain text."
    ),
    "reviewer": (
        "\n\nRole: REVIEWER. Inspect diffs via read_symbol/read_slice/run_command if needed, "
        "else reply concisely starting with `VERIFIED: ...`."
    ),
    "explorer": (
        "\n\nRole: EXPLORER. Read-only inspection via read_outline/read_symbol/read_slice/search_text/list_files. "
        "Never edit files or spawn subagents; return concise findings with verbatim paths and lines."
    ),
}

_JSON_TYPE_NAME = {
    str: "string",
    int: "integer",
    bool: "boolean",
    float: "number",
    list: "array",
}


def get_nc_openai_tools(
    names: tuple[str, ...] | list[str] | None = None,
    *,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """Return OpenAI-compatible function tool declarations for llama-server --jinja / HF template."""
    if names is not None:
        selected = tuple(names)
    elif depth >= 1:
        selected = WORKER_CODER_TOOLS
    else:
        selected = DEFAULT_CODER_TOOLS

    tools: list[dict[str, Any]] = []
    for name in selected:
        if name not in NC_TOOL_SPECS:
            raise KeyError(f"Unknown nc-agent tool: {name!r}")
        spec = NC_TOOL_SPECS[name]
        props: dict[str, Any] = {}
        enums = spec.get("enum", {})
        for pname, ptype in spec["types"].items():
            prop_schema: dict[str, Any] = {
                "type": _JSON_TYPE_NAME[ptype],
                "description": spec["param_desc"][pname],
            }
            if ptype is list:
                prop_schema["items"] = {"type": "string"}
            if pname in enums:
                prop_schema["enum"] = list(enums[pname])
            props[pname] = prop_schema
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": list(spec["required"]),
                    },
                },
            }
        )
    return tools


def build_nc_system_prompt(
    role: str = "coder",
    base: str = NC_CAVEMAN_BASE,
    depth: int = 0,
) -> str:
    """Return clean NativeTools system prompt for a given nc-agent role and subagent depth."""
    if role not in NC_ROLE_GUIDES:
        raise ValueError(f"Unknown nc-agent role: {role!r}")
    suffix = NC_ROLE_GUIDES[role]
    if role == "coder" and depth >= 1:
        suffix = suffix.replace(_CODER_DELEGATE_CLAUSE, "")
    return f"{base}{suffix}"


def format_repo_map_file(
    path: str,
    lines: int,
    symbols: list[tuple[str, int]] | tuple[tuple[str, int], ...] = (),
) -> str:
    """Format a file entry matching crates/nc-agent/src/repomap.rs."""
    if symbols:
        sym_str = ", ".join(f"{name}@{ln}" for name, ln in symbols)
        return f"{path}: {sym_str} [{lines}L]"
    return f"{path}: [{lines}L]"


def format_repo_map_dir(
    dir_path: str,
    file_count: int,
    total_lines: int,
    sym_count: int,
) -> str:
    """Format a cold directory rollup matching crates/nc-agent/src/repomap.rs."""
    clean = dir_path.rstrip("/")
    return f"{clean}/ [{file_count}f, {total_lines}L, {sym_count} syms]"


def build_nc_user_prompt(
    repo_map: str = "",
    ledger: str = "",
    task: str = "",
    scratchpad_entries: list[dict[str, Any]] | None = None,
    code_graph: str = "",
    *,
    legacy_gbnf: bool = False,
) -> str:
    """Render nc-agent user prompt: NativeTools (REPO_MAP/LEDGER/TASK) or legacy GBNF fallback."""
    sections: list[str] = []
    graph_str = code_graph.strip()
    map_str = repo_map.strip()
    if graph_str and map_str:
        if "[FILES]" in map_str:
            sections.append(f"## REPO_MAP\n[CODE_GRAPH 1-HOP]\n{graph_str}\n{map_str}")
        else:
            sections.append(f"## REPO_MAP\n[CODE_GRAPH 1-HOP]\n{graph_str}\n[FILES]\n{map_str}")
    elif graph_str:
        sections.append(f"## REPO_MAP\n[CODE_GRAPH 1-HOP]\n{graph_str}")
    elif map_str:
        sections.append(f"## REPO_MAP\n{map_str}")

    if ledger.strip():
        sections.append(f"## LEDGER\n{ledger.strip()}")
    if task.strip():
        sections.append(f"## TASK\n{task.strip()}")
    if (legacy_gbnf or scratchpad_entries) and scratchpad_entries:
        lines: list[str] = []
        for entry in scratchpad_entries:
            step = entry["step"]
            tool = entry["tool"]
            if entry.get("evicted"):
                lines.append(f"[step {step} {tool} -> {entry['receipt']}]")
            else:
                lines.append(f"[step {step} {tool}]\n{str(entry['raw_output']).rstrip()}")
        sections.append("## SCRATCHPAD\n" + "\n".join(lines))
        sections.append(
            "## NEXT\nUse SCRATCHPAD above. Call tools only for missing facts; do not repeat reads."
        )
    return "\n\n".join(sections)


def normalize_tool_call(raw_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract (tool_name, arguments_dict) from OpenAI or flat tool_call shapes."""
    if "function" in raw_call and isinstance(raw_call["function"], dict):
        fn = raw_call["function"]
        name = fn.get("name", "")
        args = fn.get("arguments", {})
    else:
        name = raw_call.get("name", "")
        args = raw_call.get("arguments", {})

    if isinstance(args, str):
        args = json.loads(args)
    if args is None:
        args = {}
    return str(name), args


def validate_tool_call(tool_name: str, args: dict[str, Any]) -> list[str]:
    """Validate tool name and arguments strictly against nc-agent's Rust schemas."""
    if tool_name not in NC_TOOL_SPECS:
        return [f"unknown tool {tool_name!r}"]
    if not isinstance(args, dict):
        return [f"tool {tool_name!r} arguments must be a dict"]

    errors: list[str] = []
    spec = NC_TOOL_SPECS[tool_name]
    allowed = set(spec["required"]) | set(spec["optional"])

    for key in args:
        if key in spec["forbidden"] or key not in allowed:
            errors.append(f"tool {tool_name!r} received disallowed arg {key!r}")

    for req in spec["required"]:
        if req not in args:
            errors.append(f"tool {tool_name!r} missing required arg {req!r}")

    enums = spec.get("enum", {})
    for key, val in args.items():
        expected_type = spec["types"].get(key)
        if expected_type is int:
            if not isinstance(val, int) or isinstance(val, bool):
                errors.append(f"tool {tool_name!r} arg {key!r} must be int")
            elif key in spec.get("positive_ints", ()) and val < 1:
                errors.append(f"tool {tool_name!r} arg {key!r} must be >= 1")
        elif expected_type is str:
            if not isinstance(val, str):
                errors.append(f"tool {tool_name!r} arg {key!r} must be str")
            else:
                if key in spec["non_empty"] and not val.strip():
                    errors.append(f"tool {tool_name!r} arg {key!r} must be non-empty")
                if key in enums and val not in enums[key]:
                    errors.append(
                        f"tool {tool_name!r} arg {key!r}={val!r} must be one of {enums[key]}"
                    )
        elif expected_type is list:
            if not isinstance(val, list) or any(
                not isinstance(item, str) or not item.strip() for item in val
            ):
                errors.append(f"tool {tool_name!r} arg {key!r} must be a list of non-empty strings")

    if (
        tool_name == "read_slice"
        and isinstance(args.get("start_line"), int)
        and not isinstance(args.get("start_line"), bool)
        and isinstance(args.get("end_line"), int)
        and not isinstance(args.get("end_line"), bool)
        and args["end_line"] < args["start_line"]
    ):
        errors.append("tool 'read_slice' end_line must be >= start_line")

    if (
        tool_name == "edit_file"
        and isinstance(args.get("target"), str)
        and isinstance(args.get("replacement"), str)
        and args["target"] == args["replacement"]
    ):
        errors.append("tool 'edit_file' is a no-op (target == replacement)")

    return errors

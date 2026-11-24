"""Exact nc-agent tool schemas (NativeTools OpenAI format + prompt layout)."""

from __future__ import annotations

from typing import Any

# Mirror of D:\Coding\nuclear_code\crates\nc-agent\src\orchestrator.rs:580-664.
NC_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "read_symbol": {
        "description": "Read exact source body of a function, struct, impl, or class symbol.",
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
        "description": "Search workspace text or symbols across files.",
        "required": ("query",),
        "optional": ("ext", "max_hits"),
        "types": {"query": str, "ext": str, "max_hits": int},
        "param_desc": {
            "query": "Exact text or identifier to search for.",
            "ext": "Optional file extension filter without dot, e.g. rs or py.",
            "max_hits": "Maximum number of search matches to return (>= 1).",
        },
        "non_empty": ("query",),
        "positive_ints": ("max_hits",),
        "forbidden": ("path", "symbol"),
    },
    "edit_file": {
        "description": "Replace an exact existing text snippet in a file. Always read before editing; never pass line numbers or full-file content.",
        "required": ("path", "target", "replacement"),
        "optional": (),
        "types": {"path": str, "target": str, "replacement": str},
        "param_desc": {
            "path": "Workspace-relative file path.",
            "target": "Exact existing verbatim code snippet to replace.",
            "replacement": "Replacement snippet preserving existing types unless explicitly asked.",
        },
        "non_empty": ("path", "target"),
        "forbidden": ("start_line", "end_line", "content", "new_content", "old_str", "new_str"),
    },
    "read_outline": {
        "description": "List top-level symbols and line ranges in a source file.",
        "required": ("path",),
        "optional": (),
        "types": {"path": str},
        "param_desc": {"path": "Workspace-relative file path."},
        "non_empty": ("path",),
        "forbidden": (),
    },
    "read_slice": {
        "description": "Read a bounded inclusive line range from a file.",
        "required": ("path", "start_line", "end_line"),
        "optional": (),
        "types": {"path": str, "start_line": int, "end_line": int},
        "param_desc": {
            "path": "Workspace-relative file path.",
            "start_line": "1-based start line number.",
            "end_line": "1-based inclusive end line number.",
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
            "max_entries": "Maximum number of entries to list (>= 1).",
        },
        "non_empty": (),
        "positive_ints": ("max_entries",),
        "forbidden": (),
    },
    "write_file": {
        "description": "Create a new file or overwrite an entire file.",
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
        "description": "Run a shell check, build, or test command in the workspace.",
        "required": ("cmd",),
        "optional": (),
        "types": {"cmd": str},
        "param_desc": {"cmd": "Command string to execute."},
        "non_empty": ("cmd",),
        "forbidden": (),
    },
}

DEFAULT_CODER_TOOLS: tuple[str, ...] = tuple(NC_TOOL_SPECS.keys())

NC_CAVEMAN_BASE = (
    "You are nuclear_code, a terse local coding agent. "
    "Answer first, zero fluff. Code, paths, commands, symbols, and errors stay verbatim."
)

NC_ROLE_GUIDES: dict[str, str] = {
    "planner": (
        "\n\nRole: PLANNER. Break goal into <= 4 micro-tasks with target files and verify commands."
    ),
    "coder": (
        "\n\nRole: CODER. Always read before edit_file; target must match exact existing text "
        "and preserve existing static types unless explicitly asked. "
        "Never repeat reads already in SCRATCHPAD; once facts are ready, respond concisely in plain text."
    ),
    "reviewer": (
        "\n\nRole: REVIEWER. Inspect diffs via read_symbol or read_slice if needed, "
        "then conclude concisely with VERIFIED or issues found."
    ),
}

_JSON_TYPE_NAME = {str: "string", int: "integer", bool: "boolean", float: "number"}


def get_nc_openai_tools(names: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    """Return OpenAI-compatible function tool declarations for llama-server --jinja / HF template."""
    selected = tuple(names) if names is not None else DEFAULT_CODER_TOOLS
    tools: list[dict[str, Any]] = []
    for name in selected:
        if name not in NC_TOOL_SPECS:
            raise KeyError(f"Unknown nc-agent tool: {name!r}")
        spec = NC_TOOL_SPECS[name]
        props: dict[str, Any] = {}
        for pname, ptype in spec["types"].items():
            props[pname] = {
                "type": _JSON_TYPE_NAME[ptype],
                "description": spec["param_desc"][pname],
            }
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


def build_nc_system_prompt(role: str = "coder", base: str = NC_CAVEMAN_BASE) -> str:
    """Return clean NativeTools system prompt for a given nc-agent role."""
    if role not in NC_ROLE_GUIDES:
        raise ValueError(f"Unknown nc-agent role: {role!r}")
    return f"{base}{NC_ROLE_GUIDES[role]}"


def build_nc_user_prompt(
    repo_map: str = "",
    ledger: str = "",
    task: str = "",
    scratchpad_entries: list[dict[str, Any]] | None = None,
) -> str:
    """Render nc-agent's ## REPO_MAP / ## LEDGER / ## TASK / ## SCRATCHPAD / ## NEXT layout."""
    sections: list[str] = []
    if repo_map.strip():
        sections.append(f"## REPO_MAP\n{repo_map.strip()}")
    if ledger.strip():
        sections.append(f"## LEDGER\n{ledger.strip()}")
    if task.strip():
        sections.append(f"## TASK\n{task.strip()}")
    if scratchpad_entries:
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
    import json

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
            elif key in spec["non_empty"] and not val.strip():
                errors.append(f"tool {tool_name!r} arg {key!r} must be non-empty")

    return errors

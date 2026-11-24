"""Deterministic benchmark probes, real code execution checks, and multimodal asset generation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from distill.nc_schema import (
    NC_TOOL_SPECS,
    build_nc_system_prompt,
    build_nc_user_prompt,
    format_repo_map_dir,
    format_repo_map_file,
    get_nc_openai_tools,
    normalize_tool_call,
    validate_tool_call,
)
from distill.validator import (
    MAX_REASONING_CHARS,
    _RAW_JSON_LEAK_RE,
    _check_planner_tasks_detail,
    _check_subtask_receipt_text,
    has_repetition_loop,
    is_emoji_or_banned_symbol,
)

from .metrics import (
    DEFAULT_PLE_CPU_OFFLOAD,
    CompletionTelemetry,
    LlamaServerSession,
    VramMonitor,
    stream_chat_completion,
)

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", re.DOTALL)


def _no_think_kwargs() -> dict[str, Any]:
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _think_kwargs() -> dict[str, Any]:
    return {"chat_template_kwargs": {"enable_thinking": True}}


def _log_probe(
    tag: str,
    probe_id: str,
    passed: bool,
    tel: CompletionTelemetry | None = None,
    note: str = "",
) -> None:
    status = "PASS" if passed else "FAIL"
    if tel is not None:
        pp = f"{tel.pp_tps:.1f}" if tel.pp_tps is not None else "-"
        tg = f"{tel.tg_tps:.1f}" if tel.tg_tps is not None else "-"
        ttfb = f"{tel.ttfb_ms:.0f}" if tel.ttfb_ms is not None else "-"
        extra = f" | {note}" if note else ""
        print(
            f"  [{tag}] {probe_id:<22} -> {status} | "
            f"ttfb={ttfb}ms total={tel.total_ms:.0f}ms pp={pp} tg={tg} "
            f"tok={tel.prompt_tokens}/{tel.completion_tokens}/{tel.reasoning_tokens}/c{tel.cached_tokens}{extra}",
            flush=True,
        )
    else:
        extra = f" | {note}" if note else ""
        print(f"  [{tag}] {probe_id:<22} -> {status}{extra}", flush=True)


def _repo_map_fixture() -> tuple[str, str]:
    files = "\n".join(
        [
            format_repo_map_file(
                "Cargo.toml",
                33,
                [("workspace", 1)],
            ),
            format_repo_map_file(
                "src/math.rs",
                48,
                [("apply_tax", 4), ("discount", 18), ("test_apply_tax", 30)],
            ),
            format_repo_map_file(
                "crates/nc-agent/src/orchestrator.rs",
                2593,
                [("AgentOrchestrator::run_turn", 120), ("TokenUsageStats::reset", 412)],
            ),
            format_repo_map_file(
                "crates/nc-agent/src/context.rs",
                480,
                [("final_only_gbnf", 88), ("planner_gbnf", 120), ("assemble_context", 210)],
            ),
            format_repo_map_dir("crates/nc-studio/src", 6, 940, 28),
        ]
    )
    graph = (
        "src/math.rs::apply_tax -> crates/nc-core/src/billing.rs::compute_total\n"
        "crates/nc-agent/src/orchestrator.rs::run_turn -> crates/nc-agent/src/context.rs::final_only_gbnf"
    )
    return files, graph


# ---------------------------------------------------------------------------
# Multimodal asset generation (Windows System.Drawing + System.Speech + ffmpeg)
# ---------------------------------------------------------------------------


def _run_ps(script: str, timeout_s: float = 45.0) -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=True,
    )


def ensure_multimodal_assets(assets_dir: Path, ffmpeg_exe: str = "ffmpeg") -> dict[str, Path]:
    """Create reproducible vision PNGs and 16 kHz mono WAV probes without extra pip libs."""
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    code_png = assets_dir / "code_bug.png"
    ui_png = assets_dir / "ui_console.png"
    audio_math = assets_dir / "audio_math.wav"
    audio_bug = assets_dir / "audio_bug_note.wav"

    if not code_png.is_file() or code_png.stat().st_size == 0:
        png_esc = str(code_png).replace("'", "''")
        ps_code = f"""
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap(920, 380)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::FromArgb(20, 24, 32))
$hdrFont = New-Object System.Drawing.Font('Consolas', 20, [System.Drawing.FontStyle]::Bold)
$codeFont = New-Object System.Drawing.Font('Consolas', 19, [System.Drawing.FontStyle]::Regular)
$errBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 110, 110))
$fgBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(235, 240, 248))
$accBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(120, 210, 255))
$g.DrawString('FILE: src/math.rs', $hdrFont, $accBrush, 28, 24)
$g.DrawString('pub fn apply_tax(price: i32) -> i32 {{', $codeFont, $fgBrush, 28, 86)
$g.DrawString('    price + 1', $codeFont, $fgBrush, 28, 128)
$g.DrawString('}}', $codeFont, $fgBrush, 28, 170)
$g.DrawString('FAILED test_apply_tax:', $hdrFont, $errBrush, 28, 245)
$g.DrawString('expected 110, got 101 for price=100', $codeFont, $errBrush, 28, 290)
$bmp.Save('{png_esc}', [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
"""
        _run_ps(ps_code)

    if not ui_png.is_file() or ui_png.stat().st_size == 0:
        ui_esc = str(ui_png).replace("'", "''")
        ps_ui = f"""
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap(960, 380)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::FromArgb(16, 19, 25))
$titleFont = New-Object System.Drawing.Font('Consolas', 20, [System.Drawing.FontStyle]::Bold)
$rowFont = New-Object System.Drawing.Font('Consolas', 18, [System.Drawing.FontStyle]::Regular)
$okBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(120, 240, 160))
$warnBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 195, 95))
$fgBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(235, 240, 248))
$g.DrawString('NUCLEAR CODE IDE STATUS', $titleFont, $okBrush, 28, 24)
$g.DrawString('brain=gemma4-e4b-nc | ctx=16384 | kv=q8_0', $rowFont, $fgBrush, 28, 82)
$g.DrawString('DIAGNOSTICS:', $titleFont, $warnBrush, 28, 152)
$g.DrawString('ERROR src/auth.rs:42 - missing bearer token', $rowFont, $warnBrush, 28, 206)
$g.DrawString('TOTAL ERRORS: 3', $titleFont, $warnBrush, 28, 272)
$bmp.Save('{ui_esc}', [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
"""
        _run_ps(ps_ui)

    _synth_wav(
        audio_math,
        "Calculate twenty seven times fourteen, then subtract eighteen.",
        ffmpeg_exe=ffmpeg_exe,
    )
    _synth_wav(
        audio_bug,
        "Bug note. File src slash login dot rs, line forty two throws missing header error.",
        ffmpeg_exe=ffmpeg_exe,
    )
    return {
        "code_bug_png": code_png,
        "ui_console_png": ui_png,
        "audio_math_wav": audio_math,
        "audio_bug_wav": audio_bug,
    }


def _synth_wav(out_wav: Path, spoken_text: str, *, ffmpeg_exe: str = "ffmpeg") -> None:
    if out_wav.is_file() and out_wav.stat().st_size > 1000:
        return
    raw_wav = out_wav.with_suffix(".raw.wav")
    raw_esc = str(raw_wav).replace("'", "''")
    txt_esc = spoken_text.replace("'", "''")
    ps_tts = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = -1
$s.SetOutputToWaveFile('{raw_esc}')
$s.Speak('{txt_esc}')
$s.Dispose()
"""
    _run_ps(ps_tts)
    try:
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(raw_wav),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(out_wav),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    finally:
        raw_wav.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Category 2: Vocabulary Surgery & Tokenizer Diagnostics
# ---------------------------------------------------------------------------


_TOKENIZE_PROBES: tuple[tuple[str, str], ...] = (
    (
        "en_prose",
        "Nuclear Code local coding agent preserves exact line ranges and state ledger facts.",
    ),
    (
        "vi_diacritics",
        "Kiem tra phau thuat tu dien tieng Viet: ă â đ ê ô ơ ư ệ ộ ự ả ẫ ỡ ỹ ị.",
    ),
    (
        "rust_code",
        "pub fn apply_tax(price: i32) -> Result<i32, String> { price.checked_mul(110).map(|v| v / 100).ok_or_else(|| \"overflow\".into()) }",
    ),
    (
        "python_code",
        "def dijkstra(adj: dict[int, list[tuple[int, int]]], src: int) -> dict[int, int]:\n    return {src: 0}",
    ),
    (
        "web_code",
        "<aside class=\"nav\"></aside><main style=\"display:grid;grid-template-columns:240px 1fr;\"></main>",
    ),
)


def _tokenize_with_llama(tokenize_exe: Path, gguf_path: Path, text: str) -> list[int]:
    proc = subprocess.run(
        [str(Path(tokenize_exe).resolve()), "-m", str(Path(gguf_path).resolve()), "--stdin", "--ids"],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=30,
        check=True,
    )
    raw = proc.stdout.decode("utf-8", errors="replace")
    for line in reversed(raw.splitlines()):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return [int(x) for x in json.loads(stripped)]
    raise RuntimeError(f"Could not parse token IDs from llama-tokenize stdout: {raw[-200:]}")


def run_vocab_diagnostics(tokenize_exe: Path, gguf_path: Path) -> dict[str, Any]:
    """Run Category 2 tokenizer audit across EN, VI diacritics, Rust, Python, and Web."""
    unk_probe = _tokenize_with_llama(tokenize_exe, gguf_path, "<unk>")
    non_bos_unk = [tid for tid in unk_probe if tid != 2]
    unk_id = non_bos_unk[0] if len(non_bos_unk) == 1 else 3

    details: list[dict[str, Any]] = []
    total_tokens = 0
    total_chars = 0
    total_unks = 0
    for name, sample_text in _TOKENIZE_PROBES:
        ids = _tokenize_with_llama(tokenize_exe, gguf_path, sample_text)
        body_ids = ids[1:] if ids and ids[0] == 2 else ids
        unk_hits = sum(1 for tid in body_ids if tid == unk_id)
        total_unks += unk_hits
        total_tokens += len(body_ids)
        total_chars += len(sample_text)
        details.append(
            {
                "name": name,
                "chars": len(sample_text),
                "tokens": len(body_ids),
                "chars_per_token": round(len(sample_text) / max(len(body_ids), 1), 3),
                "unk_count": unk_hits,
            }
        )

    print(
        f"  [Cat2] vocab unk={total_unks} tokens={total_tokens} avg_chars_per_tok={round(total_chars / max(total_tokens, 1), 3)}",
        flush=True,
    )
    return {
        "unk_token_id": unk_id,
        "total_unks": total_unks,
        "total_tokens": total_tokens,
        "avg_chars_per_token": round(total_chars / max(total_tokens, 1), 3),
        "zero_unk_pass": total_unks == 0,
        "probes": details,
    }


# ---------------------------------------------------------------------------
# Category 3: Native Tool Calling (nc-agent 9-tool contract)
# ---------------------------------------------------------------------------


def run_native_tool_probes(
    host: str, port: int
) -> tuple[dict[str, Any], list[CompletionTelemetry]]:
    """Run 6 NativeTools schema & behavioral probes against llama-server."""
    repo_map, graph = _repo_map_fixture()
    tools_d0 = get_nc_openai_tools(depth=0)
    sys_coder = build_nc_system_prompt("coder", depth=0)

    specs: list[dict[str, Any]] = [
        {
            "id": "nt_read_symbol",
            "expected_tool": "read_symbol",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_coder},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            code_graph=graph,
                            task="Inspect `TokenUsageStats::reset` in `crates/nc-agent/src/orchestrator.rs`.",
                        ),
                    },
                ],
                "tools": tools_d0,
                "tool_choice": "auto",
                "max_tokens": 180,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
            "check_args": lambda a: "TokenUsageStats" in a.get("symbol", "")
            or "reset" in a.get("symbol", ""),
        },
        {
            "id": "nt_read_slice",
            "expected_tool": "read_slice",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_coder},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            task="Call `read_slice` on `crates/nc-agent/src/context.rs` lines 85 to 115.",
                        ),
                    },
                ],
                "tools": tools_d0,
                "tool_choice": "auto",
                "max_tokens": 180,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
            "check_args": lambda a: a.get("start_line") == 85
            and a.get("end_line") == 115
            and "context.rs" in a.get("path", ""),
        },
        {
            "id": "nt_edit_file",
            "expected_tool": "edit_file",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_coder},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            task=(
                                "Fix `apply_tax` in `src/math.rs` so `price + 1` becomes "
                                "`price * 110 / 100` preserving `i32`."
                            ),
                        ),
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_read_1",
                                "type": "function",
                                "function": {
                                    "name": "read_symbol",
                                    "arguments": json.dumps(
                                        {"path": "src/math.rs", "symbol": "apply_tax"}
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_1",
                        "name": "read_symbol",
                        "content": "pub fn apply_tax(price: i32) -> i32 {\n    price + 1\n}",
                    },
                ],
                "tools": tools_d0,
                "tool_choice": "auto",
                "max_tokens": 220,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
            "check_args": lambda a: bool(
                re.search(r"\(?\s*price\s*\*\s*110\s*\)?\s*/\s*100", a.get("replacement", ""))
            )
            and "price + 1" in a.get("target", "")
            and "f64" not in a.get("replacement", ""),
        },
        {
            "id": "nt_search_text",
            "expected_tool": "search_text",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_coder},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            task="Use `search_text` to find all occurrences of `final_only_gbnf` in `.rs` files.",
                        ),
                    },
                ],
                "tools": tools_d0,
                "tool_choice": "auto",
                "max_tokens": 160,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
            "check_args": lambda a: a.get("query") == "final_only_gbnf",
        },
        {
            "id": "nt_spawn_subagent",
            "expected_tool": "spawn_subagent",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_coder},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            code_graph=graph,
                            task=(
                                "Delegate a read-only `explorer` worker via `spawn_subagent` to trace callers of "
                                "`apply_tax` across `src/math.rs` and `crates/nc-core/src/billing.rs`."
                            ),
                        ),
                    },
                ],
                "tools": tools_d0,
                "tool_choice": "auto",
                "max_tokens": 220,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
            "check_args": lambda a: a.get("role") == "explorer"
            and isinstance(a.get("files"), list)
            and bool(a.get("task")),
        },
        {
            "id": "nt_final_step",
            "expected_tool": None,
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_coder},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            task="Where is `final_only_gbnf` defined?",
                        ),
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_s1",
                                "type": "function",
                                "function": {
                                    "name": "search_text",
                                    "arguments": json.dumps({"query": "final_only_gbnf", "ext": "rs"}),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_s1",
                        "name": "search_text",
                        "content": "crates/nc-agent/src/context.rs:88: pub fn final_only_gbnf() -> &'static str",
                    },
                    {
                        "role": "user",
                        "content": "[FINAL STEP] Answer TASK concisely in plain text now; do not call tools.",
                    },
                ],
                "tools": tools_d0,
                "tool_choice": "auto",
                "max_tokens": 160,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
            "check_text": lambda t: "context.rs" in t and "88" in t,
        },
    ]

    results: list[dict[str, Any]] = []
    telems: list[CompletionTelemetry] = []
    for sp in specs:
        tel = stream_chat_completion(host, port, sp["payload"])
        telems.append(tel)
        score = score_native_tool_response(
            tel,
            expected_tool=sp["expected_tool"],
            check_args=sp.get("check_args"),
            check_text=sp.get("check_text"),
        )
        score["id"] = sp["id"]
        score["telemetry"] = tel.to_dict()
        _log_probe("Cat3", sp["id"], score["passed"], tel, "; ".join(score["errors"])[:80])
        results.append(score)

    passed = sum(1 for r in results if r["passed"])
    summary = {
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / max(len(results), 1), 3),
        "items": results,
    }
    return summary, telems


def score_native_tool_response(
    tel: CompletionTelemetry,
    *,
    expected_tool: str | None,
    check_args: Any = None,
    check_text: Any = None,
) -> dict[str, Any]:
    """Validate NativeTools response against nc-agent contract and semantic expectations."""
    errors: list[str] = []
    if not tel.ok:
        return {"passed": False, "errors": [tel.error or "request failed"]}
    if is_emoji_or_banned_symbol(tel.content) or is_emoji_or_banned_symbol(tel.reasoning_content):
        errors.append("unicode emoji detected")
    if _RAW_JSON_LEAK_RE.search(tel.content):
        errors.append("raw JSON tool/thought leak in content")
    if has_repetition_loop(tel.content):
        errors.append("repetition loop detected")

    if expected_tool is None:
        if tel.tool_calls:
            errors.append("unexpected tool_call on [FINAL STEP]")
        if not tel.content.strip():
            errors.append("empty plain text answer")
        elif check_text is not None and not check_text(tel.content):
            errors.append("final text missed required citation")
    else:
        if not tel.tool_calls:
            errors.append(f"expected tool_call {expected_tool!r}, got none")
        else:
            try:
                tname, targs = normalize_tool_call(tel.tool_calls[0])
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid tool JSON: {exc}")
                tname, targs = "", {}
            if tname and tname != expected_tool:
                errors.append(f"expected tool {expected_tool!r}, got {tname!r}")
            if tname in NC_TOOL_SPECS:
                errors.extend(validate_tool_call(tname, targs))
                if not errors and check_args is not None and not check_args(targs):
                    errors.append("tool args failed semantic check")

    return {"passed": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# Category 4: PER Loop (Plan -> Micro-Execute -> Replan)
# ---------------------------------------------------------------------------


def _extract_json_object(raw: str) -> str:
    txt = raw.strip()
    m = _FENCE_RE.search(txt)
    if m:
        txt = m.group(1).strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end > start:
        return txt[start : end + 1]
    return txt


def run_per_loop_probes(
    host: str, port: int
) -> tuple[dict[str, Any], list[CompletionTelemetry]]:
    """Test PER Loop initial_plan JSON, micro_execute 4-line SubTaskReceipt, and replan."""
    repo_map, graph = _repo_map_fixture()
    sys_planner = build_nc_system_prompt("planner", depth=0)
    sys_worker = build_nc_system_prompt("coder", depth=1)

    probes = [
        {
            "id": "per_initial_plan",
            "stage": "initial_plan",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_planner},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            code_graph=graph,
                            task="Plan tax bug fix in src/math.rs and verify caller billing behavior.",
                        ),
                    },
                ],
                "max_tokens": 320,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
        },
        {
            "id": "per_micro_receipt",
            "stage": "micro_execute",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_worker},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            task=(
                                "Complete micro-task T1 cleanly. Tool edit_file succeeded on src/math.rs (applied price * 110 / 100) "
                                "and `cargo test` passed. Emit ONLY the exact 4-line SubTaskReceipt "
                                "(STATUS:, CHANGES:, FILES:, ISSUES:)."
                            ),
                        ),
                    },
                ],
                "tools": get_nc_openai_tools(depth=1),
                "tool_choice": "auto",
                "max_tokens": 160,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
        },
        {
            "id": "per_replan",
            "stage": "replan",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_planner},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            ledger=(
                                "STATUS: BLOCKED\n"
                                "CHANGES: None\n"
                                "FILES: crates/nc-core/src/billing.rs\n"
                                "ISSUES: compute_total expects cents (u64) instead of i32 dollars\n"
                                "ACTION: REPLAN remaining tasks to bridge billing.rs units."
                            ),
                            task="REPLAN next micro-tasks strictly as JSON.",
                        ),
                    },
                ],
                "max_tokens": 320,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
        },
    ]

    items: list[dict[str, Any]] = []
    telems: list[CompletionTelemetry] = []
    for p in probes:
        tel = stream_chat_completion(host, port, p["payload"])
        telems.append(tel)
        sc = score_per_loop_response(tel, p["stage"])
        sc["id"] = p["id"]
        sc["telemetry"] = tel.to_dict()
        _log_probe("Cat4", p["id"], sc["passed"], tel, "; ".join(sc["errors"])[:80])
        items.append(sc)

    passed = sum(1 for r in items if r["passed"])
    return {
        "passed": passed,
        "total": len(items),
        "pass_rate": round(passed / max(len(items), 1), 3),
        "items": items,
    }, telems


def score_per_loop_response(tel: CompletionTelemetry, stage: str) -> dict[str, Any]:
    """Validate PER Loop outputs reusing distill.validator planner and receipt checks."""
    if not tel.ok:
        return {"passed": False, "errors": [tel.error or "request failed"]}
    errs: list[str] = []
    if is_emoji_or_banned_symbol(tel.content):
        errs.append("unicode emoji detected")
    if stage in ("initial_plan", "replan"):
        json_candidate = _extract_json_object(tel.content)
        try:
            parsed = json.loads(json_candidate)
            if not isinstance(parsed, dict) or not (1 <= len(parsed.get("tasks", [])) <= 4):
                errs.append("planner tasks must contain 1..4 items")
            else:
                errs.extend(_check_planner_tasks_detail(json_candidate))
        except Exception as exc:  # noqa: BLE001
            errs.append(f"invalid planner JSON: {exc}")
    elif stage == "micro_execute":
        if tel.tool_calls:
            errs.append("unexpected tool_call instead of final SubTaskReceipt")
        errs.extend(_check_subtask_receipt_text(tel.content))
    return {"passed": not errs, "errors": errs}


# ---------------------------------------------------------------------------
# Category 5: Adaptive Thinking & Subagent depth=1 Discipline
# ---------------------------------------------------------------------------


def run_adaptive_thinking_probes(
    host: str, port: int
) -> tuple[dict[str, Any], list[CompletionTelemetry]]:
    """Verify thinking=false suppression, thinking=true brevity, and depth=1 no-spawn."""
    repo_map, graph = _repo_map_fixture()
    sys_lead = build_nc_system_prompt("coder", depth=0)
    sys_worker = build_nc_system_prompt("coder", depth=1)

    probes = [
        {
            "id": "adapt_think_off_tool",
            "mode": "think_off",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_lead},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            task="Read outline of `crates/nc-agent/src/context.rs`.",
                        ),
                    },
                ],
                "tools": get_nc_openai_tools(depth=0),
                "tool_choice": "auto",
                "max_tokens": 160,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
        },
        {
            "id": "adapt_think_on_lead",
            "mode": "think_on",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_lead},
                    {
                        "role": "user",
                        "content": (
                            "Which bug causes apply_tax(100) to return 101 instead of 110, "
                            "and why must the return type stay i32 rather than f64? Answer in <= 2 sentences."
                        ),
                    },
                ],
                "max_tokens": 640,
                "temperature": 0.1,
                **_think_kwargs(),
            },
        },
        {
            "id": "adapt_worker_depth1",
            "mode": "worker_d1",
            "payload": {
                "messages": [
                    {"role": "system", "content": sys_worker},
                    {
                        "role": "user",
                        "content": build_nc_user_prompt(
                            repo_map=repo_map,
                            code_graph=graph,
                            task="You are a depth-1 worker. Inspect `src/math.rs` using a single read tool call (do not spawn subagents).",
                        ),
                    },
                ],
                "tools": get_nc_openai_tools(depth=1),
                "tool_choice": "auto",
                "max_tokens": 160,
                "temperature": 0.1,
                **_no_think_kwargs(),
            },
        },
    ]

    items: list[dict[str, Any]] = []
    telems: list[CompletionTelemetry] = []
    for p in probes:
        tel = stream_chat_completion(host, port, p["payload"])
        telems.append(tel)
        sc = score_adaptive_response(tel, p["mode"])
        sc["id"] = p["id"]
        sc["telemetry"] = tel.to_dict()
        _log_probe("Cat5", p["id"], sc["passed"], tel, "; ".join(sc["errors"])[:80])
        items.append(sc)

    passed = sum(1 for r in items if r["passed"])
    return {
        "passed": passed,
        "total": len(items),
        "pass_rate": round(passed / max(len(items), 1), 3),
        "items": items,
    }, telems


def score_adaptive_response(tel: CompletionTelemetry, mode: str) -> dict[str, Any]:
    """Check adaptive thinking discipline and worker spawn prohibition."""
    if not tel.ok:
        return {"passed": False, "errors": [tel.error or "request failed"]}
    errs: list[str] = []
    if mode == "think_off":
        if tel.reasoning_tokens > 0 or tel.reasoning_content.strip():
            errs.append(f"reasoning leaked when disabled ({tel.reasoning_tokens} tok)")
        if not tel.tool_calls:
            errs.append("missing expected tool call")
    elif mode == "think_on":
        if not tel.content.strip():
            errs.append("empty answer content after thinking")
        if len(tel.reasoning_content) > MAX_REASONING_CHARS * 2:
            errs.append(f"runaway thinking ({len(tel.reasoning_content)} chars)")
        if "110" not in tel.content and "i32" not in tel.content:
            errs.append("answer missed key fact (110 / i32)")
    elif mode == "worker_d1":
        if tel.reasoning_tokens > 0 or tel.reasoning_content.strip():
            errs.append(f"depth-1 worker leaked thinking ({tel.reasoning_tokens} tok)")
        for tc in tel.tool_calls:
            try:
                tname, _ = normalize_tool_call(tc)
            except Exception:
                tname = ""
            if tname == "spawn_subagent":
                errs.append("depth-1 worker called forbidden spawn_subagent")
        if not tel.tool_calls:
            errs.append("worker failed to issue inspection tool call")
    return {
        "passed": not errs,
        "errors": errs,
        "reasoning_chars": len(tel.reasoning_content),
        "reasoning_tokens": tel.reasoning_tokens,
    }


# ---------------------------------------------------------------------------
# Category 6: Multimodal Eyes & Ears (Image + Audio)
# ---------------------------------------------------------------------------


def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _run_mtmd_cli_probe(
    *,
    mtmd_exe: Path,
    model_path: Path,
    mmproj_path: Path,
    media_flag: str,
    media_path: Path,
    prompt: str,
) -> tuple[bool, str, float, int]:
    """Run llama-mtmd-cli.exe safely with DEVNULL stdin and VRAM peak tracking."""
    t0 = time.perf_counter()
    with VramMonitor(interval_s=0.2) as mon:
        for offload_args in ([], ["--no-mmproj-offload"]):
            cmd = [
                str(Path(mtmd_exe).resolve()),
                "-m",
                str(Path(model_path).resolve()),
                "--mmproj",
                str(Path(mmproj_path).resolve()),
                "-ot",
                DEFAULT_PLE_CPU_OFFLOAD,
                "--jinja",
                media_flag,
                str(Path(media_path).resolve()),
                "-p",
                prompt,
                "-n",
                "160",
                "--temp",
                "0.1",
                "--log-disable",
                *offload_args,
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(mtmd_exe.parent),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return True, proc.stdout.strip(), elapsed_ms, mon.peak_mib
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return False, (proc.stderr or proc.stdout or "mtmd failed")[-300:], elapsed_ms, mon.peak_mib


def run_multimodal_probes(
    *,
    server_exe: Path,
    mtmd_exe: Path,
    model_path: Path,
    mmproj_path: Path,
    assets: dict[str, Path],
    log_dir: Path,
) -> dict[str, Any]:
    """Benchmark Eyes (PNG) + Ears (WAV) via llama-server --mmproj + mtmd-cli fallback."""
    vision_audio_specs = [
        {
            "id": "mm_code_bug_png",
            "kind": "image",
            "path": assets["code_bug_png"],
            "prompt": "Read this code screenshot. State the buggy expression and the expected number in the failure banner.",
            "check": lambda t: ("price + 1" in t or "apply_tax" in t) and "110" in t,
        },
        {
            "id": "mm_ui_console_png",
            "kind": "image",
            "path": assets["ui_console_png"],
            "prompt": "Read the terminal screenshot. What file and line have an ERROR, and what is the TOTAL ERRORS count?",
            "check": lambda t: "auth.rs" in t and "42" in t and "3" in t,
        },
        {
            "id": "mm_audio_math_wav",
            "kind": "audio",
            "path": assets["audio_math_wav"],
            "prompt": "Listen carefully to the audio question and answer with the final integer result.",
            "check": lambda t: "360" in t,
        },
        {
            "id": "mm_audio_bug_wav",
            "kind": "audio",
            "path": assets["audio_bug_wav"],
            "prompt": "Transcribe the spoken bug note: state the exact filename and line number mentioned.",
            "check": lambda t: "login" in t.lower() and "42" in t,
        },
    ]

    mm_info: dict[str, Any] = {
        "mmproj_path": str(mmproj_path),
        "mmproj_size_mb": round(mmproj_path.stat().st_size / (1024 * 1024), 1),
        "server_boot_ok": False,
        "loaded_vram_mib": None,
        "peak_vram_mib": None,
        "ngl_used": None,
        "mmproj_offload_used": None,
        "items": [],
        "telemetries": [],
    }

    with LlamaServerSession(
        server_exe=server_exe,
        model_path=model_path,
        ctx_size=8192,
        mmproj_path=mmproj_path,
        ple_cpu_offload=True,
        log_dir=log_dir,
    ) as srv:
        mm_info["server_boot_ok"] = True
        mm_info["boot_ms"] = srv.boot_ms
        mm_info["loaded_vram_mib"] = srv.loaded_vram_mib
        mm_info["ngl_used"] = srv.ngl_used
        mm_info["mmproj_offload_used"] = srv.mmproj_offload_used

        for spec in vision_audio_specs:
            b64 = _b64_file(spec["path"])
            if spec["kind"] == "image":
                media_part: dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            else:
                media_part = {
                    "type": "input_audio",
                    "input_audio": {"data": b64, "format": "wav"},
                }

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            media_part,
                            {"type": "text", "text": spec["prompt"]},
                        ],
                    }
                ],
                "max_tokens": 180,
                "temperature": 0.1,
                **_no_think_kwargs(),
            }
            tel = stream_chat_completion(srv.host, srv.port, payload, timeout_s=180.0)
            if tel.ok:
                mm_info["telemetries"].append(tel.to_dict())
                passed = bool(spec["check"](tel.content)) and not is_emoji_or_banned_symbol(
                    tel.content
                )
                _log_probe("Cat6", spec["id"], passed, tel, f"mmproj={srv.mmproj_offload_used}")
                mm_info["items"].append(
                    {
                        "id": spec["id"],
                        "kind": spec["kind"],
                        "backend": "llama-server",
                        "passed": passed,
                        "answer_preview": tel.content.strip()[:220],
                        "ttfb_ms": tel.ttfb_ms,
                        "total_ms": tel.total_ms,
                        "prompt_tokens": tel.prompt_tokens,
                        "completion_tokens": tel.completion_tokens,
                    }
                )
            else:
                print(f"  [Cat6] {spec['id']:<22} -> server fallback ({tel.error})", flush=True)
                mm_info["items"].append(
                    {
                        "id": spec["id"],
                        "kind": spec["kind"],
                        "backend": "needs_mtmd_fallback",
                        "server_error": tel.error,
                    }
                )
        mm_info["peak_vram_mib"] = srv.current_peak_vram_mib()

    # Fallback via llama-mtmd-cli ONLY after llama-server exits cleanly so 4 GB VRAM is free
    for idx, item in enumerate(mm_info["items"]):
        if item.get("backend") != "needs_mtmd_fallback":
            continue
        spec = vision_audio_specs[idx]
        flag = "--image" if spec["kind"] == "image" else "--audio"
        ok, out_txt, elapsed_ms, peak_mib = _run_mtmd_cli_probe(
            mtmd_exe=mtmd_exe,
            model_path=model_path,
            mmproj_path=mmproj_path,
            media_flag=flag,
            media_path=spec["path"],
            prompt=spec["prompt"],
        )
        passed = ok and bool(spec["check"](out_txt)) and not is_emoji_or_banned_symbol(out_txt)
        _log_probe("Cat6", spec["id"], passed, None, f"mtmd-cli {elapsed_ms:.0f}ms vram={peak_mib}MiB")
        mm_info["items"][idx] = {
            "id": spec["id"],
            "kind": spec["kind"],
            "backend": "llama-mtmd-cli",
            "passed": passed,
            "answer_preview": out_txt[:220],
            "total_ms": elapsed_ms,
            "mtmd_peak_vram_mib": peak_mib,
        }
        if peak_mib > (mm_info["peak_vram_mib"] or 0):
            mm_info["peak_vram_mib"] = peak_mib

    passed_cnt = sum(1 for it in mm_info["items"] if it.get("passed"))
    mm_info["passed"] = passed_cnt
    mm_info["total"] = len(mm_info["items"])
    mm_info["pass_rate"] = round(passed_cnt / max(len(mm_info["items"]), 1), 3)
    return mm_info


# ---------------------------------------------------------------------------
# Category 7: Real Compiled / Executed Coding & Algorithms
# ---------------------------------------------------------------------------


def extract_code_block(text: str) -> str:
    """Extract the longest fenced code block from a model response, or raw text."""
    matches = _FENCE_RE.findall(text or "")
    if matches:
        return max(matches, key=len).strip()
    return (text or "").strip()


CODING_PROBES: tuple[dict[str, Any], ...] = (
    {
        "id": "code_py_dijkstra",
        "lang": "python",
        "prompt": (
            "Write a Python function `dijkstra(n: int, edges: list[tuple[int, int, int]], src: int) -> list[int]` "
            "returning shortest path distances from `src` to nodes `0..n-1` (-1 if unreachable). "
            "Output ONLY one ```python``` block."
        ),
        "harness": """
assert dijkstra(4, [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1)], 0) == [0, 3, 1, 4]
assert dijkstra(3, [(0, 1, 5)], 0) == [0, 5, -1]
assert dijkstra(1, [], 0) == [0]
print("PASS_PY_DIJKSTRA")
""",
    },
    {
        "id": "code_py_lru",
        "lang": "python",
        "prompt": (
            "Implement an O(1) `class LRUCache:` in Python with `__init__(self, capacity: int)`, "
            "`get(self, key: int) -> int` (-1 if absent), and `put(self, key: int, value: int) -> None`. "
            "Output ONLY one ```python``` block."
        ),
        "harness": """
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
c.put(3, 30)
assert c.get(3) == 30
print("PASS_PY_LRU")
""",
    },
    {
        "id": "code_rs_intervals",
        "lang": "rust",
        "prompt": (
            "Write a Rust function `pub fn merge_intervals(mut intervals: Vec<(i32, i32)>) -> Vec<(i32, i32)>` "
            "that merges all overlapping intervals and returns them sorted by start. "
            "Do NOT include `fn main()`. Output ONLY one ```rust``` block."
        ),
        "harness": """
fn main() {
    assert_eq!(merge_intervals(vec![(1,3),(2,6),(8,10),(15,18)]), vec![(1,6),(8,10),(15,18)]);
    assert_eq!(merge_intervals(vec![(4,5),(1,4)]), vec![(1,5)]);
    assert_eq!(merge_intervals(vec![]), vec![]);
    println!("PASS_RS_INTERVALS");
}
""",
    },
    {
        "id": "code_rs_parse_sum",
        "lang": "rust",
        "prompt": (
            "Write a safe Rust function `pub fn parse_positive_sum(input: &str) -> Result<i64, String>` "
            "splitting `input` by whitespace, parsing positive i64 integers (> 0), using checked_add, "
            "and returning `Err(...)` on empty input, non-positive integers (<= 0), invalid tokens, or overflow. "
            "Never call `unwrap()` or `panic!()`. Do NOT include `fn main()`. Output ONLY one ```rust``` block."
        ),
        "harness": """
fn main() {
    assert_eq!(parse_positive_sum("10 20 5"), Ok(35));
    assert!(parse_positive_sum("").is_err());
    assert!(parse_positive_sum("1 -2 3").is_err());
    assert!(parse_positive_sum("10 abc").is_err());
    println!("PASS_RS_PARSE_SUM");
}
""",
    },
    {
        "id": "code_js_deepclone",
        "lang": "javascript",
        "prompt": (
            "Write a JavaScript function `function deepClone(value, seen = new WeakMap())` "
            "that deeply clones primitives, Arrays, plain Objects, Dates, and handles circular references. "
            "Output ONLY one ```javascript``` block."
        ),
        "harness": """
const obj = { a: 1, d: new Date(1700000000000), nested: [1, { b: 2 }] };
obj.self = obj;
const cloned = deepClone(obj);
if (cloned === obj || cloned.nested === obj.nested) throw new Error('shallow');
if (cloned.self !== cloned) throw new Error('circular');
if (!(cloned.d instanceof Date) || cloned.d.getTime() !== 1700000000000) throw new Error('date');
if (cloned.a !== 1 || cloned.nested[1].b !== 2) throw new Error('vals');
console.log("PASS_JS_DEEPCLONE");
""",
    },
    {
        "id": "code_html_css_layout",
        "lang": "html",
        "prompt": (
            "Write a single HTML snippet with `<style>` CSS implementing a responsive 2-column dashboard "
            "layout having a 240px `<aside>` sidebar and fluid `<main>` pane using CSS Grid or Flexbox, "
            "collapsing to 1 column below `@media (max-width: 768px)`. Output ONLY one ```html``` block."
        ),
        "harness": "",
    },
)


def verify_code_solution(probe: dict[str, Any], raw_response: str) -> dict[str, Any]:
    """Execute generated code in a temp sandbox (Python/Rust/Node) or validate HTML/CSS."""
    code = extract_code_block(raw_response)
    lang = probe["lang"]
    if not code:
        return {"passed": False, "error": "empty code output"}

    if lang == "html":
        low = code.lower()
        checks = {
            "has_style": "<style" in low,
            "has_aside": "<aside" in low,
            "has_main": "<main" in low,
            "has_grid_or_flex": "grid" in low or "flex" in low,
            "has_240px": "240px" in low,
            "has_media_query": "@media" in low,
        }
        passed = all(checks.values())
        return {"passed": passed, "checks": checks}

    with tempfile.TemporaryDirectory(prefix="nc_code_eval_") as tmp:
        tmp_dir = Path(tmp)
        try:
            if lang == "python":
                src = tmp_dir / "sol.py"
                src.write_text(f"{code}\n\n{probe['harness']}\n", encoding="utf-8")
                proc = subprocess.run(
                    [sys.executable, str(src)],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            elif lang == "javascript":
                src = tmp_dir / "sol.js"
                src.write_text(f"{code}\n\n{probe['harness']}\n", encoding="utf-8")
                proc = subprocess.run(
                    ["node", str(src)],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            elif lang == "rust":
                if "unwrap(" in code or "panic!(" in code:
                    return {"passed": False, "error": "forbidden unwrap()/panic! in Rust solution"}
                src = tmp_dir / "sol.rs"
                exe = tmp_dir / "sol.exe"
                clean_code = re.sub(r"fn\s+main\s*\([^)]*\)\s*\{[\s\S]*\}\s*$", "", code).strip()
                src.write_text(f"{clean_code}\n\n{probe['harness']}\n", encoding="utf-8")
                comp = subprocess.run(
                    ["rustc", "--edition", "2021", str(src), "-o", str(exe)],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if comp.returncode != 0:
                    return {
                        "passed": False,
                        "error": f"rustc failed: {(comp.stderr or comp.stdout)[-300:]}",
                    }
                proc = subprocess.run(
                    [str(exe)],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            else:
                return {"passed": False, "error": f"unsupported lang {lang}"}
        except Exception as exc:  # noqa: BLE001
            return {"passed": False, "error": str(exc)}

        if proc.returncode == 0:
            return {"passed": True, "stdout": proc.stdout.strip()}
        return {
            "passed": False,
            "error": (proc.stderr or proc.stdout or f"exit {proc.returncode}")[-300:],
        }


def run_coding_probes(
    host: str, port: int
) -> tuple[dict[str, Any], list[CompletionTelemetry]]:
    """Run all 6 coding & algorithm challenges with real compiler/interpreter verification."""
    items: list[dict[str, Any]] = []
    telems: list[CompletionTelemetry] = []
    for probe in CODING_PROBES:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a senior systems engineer. Return clean, minimal, bug-free code only.",
                },
                {"role": "user", "content": probe["prompt"]},
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
            **_think_kwargs(),
        }
        tel = stream_chat_completion(host, port, payload, timeout_s=180.0)
        telems.append(tel)
        if not tel.ok:
            res = {"passed": False, "error": tel.error}
        else:
            res = verify_code_solution(probe, tel.content)
        _log_probe("Cat7", probe["id"], bool(res.get("passed")), tel, str(res.get("error") or "")[:70])
        items.append(
            {
                "id": probe["id"],
                "lang": probe["lang"],
                **res,
                "telemetry": tel.to_dict(),
            }
        )
    passed = sum(1 for it in items if it.get("passed"))
    return {
        "passed": passed,
        "total": len(items),
        "pass_rate": round(passed / max(len(items), 1), 3),
        "items": items,
    }, telems


# ---------------------------------------------------------------------------
# Category 8: Math & Computation Accuracy (Thinking ON vs OFF)
# ---------------------------------------------------------------------------


MATH_PROBES: tuple[dict[str, Any], ...] = (
    {
        "id": "math_arith",
        "prompt": "Compute `1738 * 429 - 95412`. Reply with `ANSWER: <integer>`.",
        "expected": 650190,
        "compare_no_think": False,
    },
    {
        "id": "math_mod_pow",
        "prompt": "Compute `7^13 mod 100`. Reply with `ANSWER: <integer>`.",
        "expected": 7,
        "compare_no_think": True,
    },
    {
        "id": "math_bitwise",
        "prompt": "Evaluate `(0xCAFE & 0x0FF0) >> 4` in decimal. Reply with `ANSWER: <integer>`.",
        "expected": 175,
        "compare_no_think": False,
    },
    {
        "id": "math_kv_vram",
        "prompt": (
            "Compute FP16 KV cache size in MiB for `2` (K+V) tensors * `26` layers * `8` KV heads * "
            "`256` head_dim * `16384` ctx tokens * `2` bytes per fp16 element, divided by `1048576` bytes/MiB. "
            "Reply with `ANSWER: <integer>`."
        ),
        "expected": 3328,
        "compare_no_think": True,
    },
    {
        "id": "math_comb",
        "prompt": "Compute binomial coefficient `C(12, 4)`. Reply with `ANSWER: <integer>`.",
        "expected": 495,
        "compare_no_think": False,
    },
    {
        "id": "math_linear",
        "prompt": "Solve `3x + 2y = 29` and `5x - y = 31`, then compute `x * y`. Reply with `ANSWER: <integer>`.",
        "expected": 28,
        "compare_no_think": True,
    },
    {
        "id": "math_dp_stairs",
        "prompt": "How many distinct ways can you climb 10 stairs taking 1 or 2 steps at a time? Reply with `ANSWER: <integer>`.",
        "expected": 89,
        "compare_no_think": True,
    },
    {
        "id": "math_percent",
        "prompt": "A codebase metric starts at `100000`, increases by 10%, then decreases by 10%. What is the final integer value? Reply with `ANSWER: <integer>`.",
        "expected": 99000,
        "compare_no_think": False,
    },
)


def parse_integer_answer(text: str) -> int | None:
    """Extract the final integer from `ANSWER: N` or the last integer token."""
    cleaned = text.replace(",", "")
    m = re.findall(r"ANSWER\s*:\s*(-?\d+)", cleaned, flags=re.IGNORECASE)
    if m:
        return int(m[-1])
    all_nums = re.findall(r"-?\d+", cleaned)
    return int(all_nums[-1]) if all_nums else None


def run_math_probes(
    host: str, port: int
) -> tuple[dict[str, Any], list[CompletionTelemetry]]:
    """Evaluate 8 exact computation probes with thinking=true plus thinking=false ablation."""
    think_items: list[dict[str, Any]] = []
    nothink_items: list[dict[str, Any]] = []
    telems: list[CompletionTelemetry] = []

    for probe in MATH_PROBES:
        payload_on = {
            "messages": [
                {"role": "system", "content": "Solve accurately. End with `ANSWER: <integer>`."},
                {"role": "user", "content": probe["prompt"]},
            ],
            "max_tokens": 1536,
            "temperature": 0.0,
            **_think_kwargs(),
        }
        tel_on = stream_chat_completion(host, port, payload_on, timeout_s=180.0)
        telems.append(tel_on)
        got_on = parse_integer_answer(tel_on.content) if tel_on.ok else None
        pass_on = tel_on.ok and got_on == probe["expected"]
        _log_probe("Cat8_ON", probe["id"], pass_on, tel_on, f"got={got_on} exp={probe['expected']}")
        think_items.append(
            {
                "id": probe["id"],
                "expected": probe["expected"],
                "got": got_on,
                "passed": pass_on,
                "reasoning_tokens": tel_on.reasoning_tokens,
                "completion_tokens": tel_on.completion_tokens,
                "total_ms": tel_on.total_ms,
            }
        )

        if probe["compare_no_think"]:
            payload_off = {
                "messages": payload_on["messages"],
                "max_tokens": 512,
                "temperature": 0.0,
                **_no_think_kwargs(),
            }
            tel_off = stream_chat_completion(host, port, payload_off, timeout_s=120.0)
            telems.append(tel_off)
            got_off = parse_integer_answer(tel_off.content) if tel_off.ok else None
            pass_off = tel_off.ok and got_off == probe["expected"]
            _log_probe("Cat8_OFF", probe["id"], pass_off, tel_off, f"got={got_off} exp={probe['expected']}")
            nothink_items.append(
                {
                    "id": probe["id"],
                    "expected": probe["expected"],
                    "got": got_off,
                    "passed": pass_off,
                    "completion_tokens": tel_off.completion_tokens,
                    "total_ms": tel_off.total_ms,
                }
            )

    on_pass = sum(1 for x in think_items if x["passed"])
    off_pass = sum(1 for x in nothink_items if x["passed"])
    return {
        "thinking_on_passed": on_pass,
        "thinking_on_total": len(think_items),
        "thinking_on_rate": round(on_pass / max(len(think_items), 1), 3),
        "thinking_off_passed": off_pass,
        "thinking_off_total": len(nothink_items),
        "thinking_off_rate": round(off_pass / max(len(nothink_items), 1), 3),
        "items_thinking_on": think_items,
        "items_thinking_off": nothink_items,
    }, telems

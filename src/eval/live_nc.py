"""Live nuclear_code (nc-core IPC daemon) Stage 3 benchmark runner (T1..T5)."""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import tomllib
from typing import Any
import urllib.error
import urllib.request

from distill.validator import (
    _RAW_JSON_LEAK_RE,
    has_repetition_loop,
    is_emoji_or_banned_symbol,
)

from .metrics import VramMonitor

DEFAULT_NC_REPO = Path(r"D:\Coding\nuclear_code")
DEFAULT_NC_CORE_EXE = DEFAULT_NC_REPO / "target" / "debug" / "nc-core.exe"


def _ipc_request(
    port: int,
    token: str,
    method: str,
    path: str,
    body_obj: dict[str, Any] | None = None,
    *,
    timeout_s: float = 300.0,
) -> tuple[int, Any]:
    raw_body = json.dumps(body_obj).encode("utf-8") if body_obj is not None else b""
    headers = {"Authorization": f"Bearer {token}"}
    if body_obj is not None:
        headers["Content-Type"] = "application/json"
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    try:
        conn.request(method, path, body=raw_body, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(data) if data else {}
        except json.JSONDecodeError:
            parsed = {"raw": data}
        return resp.status, parsed
    finally:
        conn.close()


def _create_temp_portable_root(
    root_dir: Path,
    *,
    nc_repo: Path,
    gguf_path: Path,
    ctx_profile: str,
) -> Path:
    if root_dir.exists():
        shutil.rmtree(root_dir, ignore_errors=True)
    brain_dir = root_dir / "brain"
    brain_dir.mkdir(parents=True, exist_ok=True)
    link_path = brain_dir / gguf_path.name
    try:
        os.link(gguf_path, link_path)
    except OSError:
        shutil.copy2(gguf_path, link_path)

    cfg = {
        "brain_dir": str(brain_dir),
        "runtime_dir": str(nc_repo / "runtime"),
        "themes_dir": str(nc_repo / "themes"),
        "prompts_dir": str(nc_repo / "prompts"),
        "active_brain": gguf_path.name,
        "backend": "auto",
        "ctx_profile": ctx_profile,
        "caveman_mode": True,
        "active_theme": "nuclear-dark",
        "permission_mode": "auto",
        "default_engine": "cli-p",
        "core_port": 0,
    }
    (root_dir / "nuclear.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return root_dir


def _reset_sandbox_workspace(ws_dir: Path) -> None:
    if ws_dir.exists():
        shutil.rmtree(ws_dir, ignore_errors=True)
    src_dir = ws_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (ws_dir / "Cargo.toml").write_text(
        '[package]\nname = "nc-eval-sandbox"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (src_dir / "lib.rs").write_text("pub mod math;\n", encoding="utf-8")
    (src_dir / "math.rs").write_text(
        "pub fn apply_tax(price: i32) -> i32 {\n"
        "    price + 1\n"
        "}\n\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    use super::apply_tax;\n"
        "    #[test]\n"
        "    fn test_apply_tax() {\n"
        "        assert_eq!(apply_tax(100), 110);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )


def _live_task_entry(
    task_id: str,
    passed: bool,
    latency_ms: float,
    resp: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    d = resp if isinstance(resp, dict) else {}
    receipts = d.get("receipts", []) if isinstance(d.get("receipts"), list) else []
    tools_used = [str(rc.get("tool", "")) for rc in receipts if isinstance(rc, dict)]
    usage = d.get("usage", {}) if isinstance(d.get("usage"), dict) else {}
    reqs = usage.get("requests", "-")
    ptok = usage.get("prompt_tokens", "-")
    ctok = usage.get("completion_tokens", "-")
    mode_used = d.get("mode_used")
    extra_bits = (
        " | " + " ".join(f"{k}={v}" for k, v in extra.items())
        if extra
        else ""
    )
    print(
        f"  [Cat10] {task_id:<26} -> {'PASS' if passed else 'FAIL'} | "
        f"{latency_ms:.0f}ms | mode={mode_used} | req={reqs} tok={ptok}/{ctok} | tools={tools_used}{extra_bits}",
        flush=True,
    )
    entry: dict[str, Any] = {
        "id": task_id,
        "passed": passed,
        "latency_ms": latency_ms,
        "mode_used": mode_used,
        "tools_used": tools_used,
        "receipts_count": len(receipts),
        "tokens_estimated": d.get("tokens_estimated", 0),
        "usage": usage,
        "answer_preview": str(d.get("final_answer", ""))[:220],
    }
    if extra:
        entry.update(extra)
    return entry


def _load_workspace_crate_names(nc_repo: Path) -> list[str]:
    cargo_data = tomllib.loads((nc_repo / "Cargo.toml").read_text(encoding="utf-8"))
    members = cargo_data.get("workspace", {}).get("members", [])
    return [Path(m).name for m in members]


def run_live_nc_suite(
    *,
    gguf_path: Path,
    work_dir: Path,
    ctx_profile: str = "8k",
    nc_repo: Path = DEFAULT_NC_REPO,
    nc_core_exe: Path = DEFAULT_NC_CORE_EXE,
) -> dict[str, Any]:
    """Run live T1..T5 evaluation against nc-core.exe without modifying nuclear_code repo."""
    if not nc_core_exe.is_file():
        return {"skipped": True, "reason": f"missing {nc_core_exe}"}

    work_dir = Path(work_dir).resolve()
    gguf_path = Path(gguf_path).resolve()
    nc_repo = Path(nc_repo).resolve()
    nc_core_exe = Path(nc_core_exe).resolve()
    portable_root = _create_temp_portable_root(
        work_dir / "portable",
        nc_repo=nc_repo,
        gguf_path=gguf_path,
        ctx_profile=ctx_profile,
    )
    sandbox_ws = work_dir / "sandbox_ws"
    _reset_sandbox_workspace(sandbox_ws)
    expected_crates = _load_workspace_crate_names(nc_repo)

    proc: subprocess.Popen[str] | None = None
    core_port = 0
    token = ""
    llama_port = 0

    try:
        with VramMonitor(interval_s=0.25) as vram_mon:
            proc = subprocess.Popen(
                [
                    str(nc_core_exe),
                    "--serve",
                    "--ephemeral",
                    "--print-token",
                    "--port",
                    "0",
                    "--workspace",
                    str(nc_repo),
                ],
                cwd=str(portable_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            deadline = time.monotonic() + 20.0
            assert proc.stdout is not None
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                m_port = re.search(r"listening on 127\.0\.0\.1:(\d+)", line)
                if m_port:
                    core_port = int(m_port.group(1))
                m_tok = re.search(r"bearer_token\s*:\s*([0-9a-fA-F]+)", line)
                if m_tok:
                    token = m_tok.group(1).strip()
                if core_port and token:
                    break

            if not (core_port and token):
                return {"passed": 0, "total": 5, "error": "nc-core failed to advertise port/token"}

            st_code, plan = _ipc_request(
                core_port,
                token,
                "POST",
                "/v1/brain/select",
                {"brain": gguf_path.name, "ctx_profile": ctx_profile, "start": True},
                timeout_s=30.0,
            )
            if st_code != 200 or not isinstance(plan, dict) or not plan.get("llama_port"):
                return {
                    "passed": 0,
                    "total": 5,
                    "error": f"/v1/brain/select failed ({st_code}): {plan}",
                }
            llama_port = int(plan["llama_port"])

            llama_ready = False
            health_deadline = time.monotonic() + 180.0
            while time.monotonic() < health_deadline:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{llama_port}/health", timeout=2.0
                    ) as hresp:
                        if hresp.status == 200:
                            llama_ready = True
                            break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                time.sleep(0.4)

            if not llama_ready:
                return {
                    "passed": 0,
                    "total": 5,
                    "error": f"nc-core llama-server on port {llama_port} failed health check",
                }

            tasks_out: list[dict[str, Any]] = []

            # T1: Multi-crate Cargo.toml workspace listing
            t0 = time.perf_counter()
            c1, r1 = _ipc_request(
                core_port,
                token,
                "POST",
                "/v1/agent/run",
                {"prompt": "List every workspace member crate listed in Cargo.toml verbatim.", "mode": "auto"},
                timeout_s=240.0,
            )
            ms1 = round((time.perf_counter() - t0) * 1000.0, 2)
            ans1 = r1.get("final_answer", "") if isinstance(r1, dict) else ""
            matched_crates = [c for c in expected_crates if c in ans1]
            t1_ok = c1 == 200 and len(matched_crates) == len(expected_crates)
            tasks_out.append(
                _live_task_entry(
                    "T1_cargo_crates",
                    t1_ok,
                    ms1,
                    r1,
                    {"crates_found": len(matched_crates), "crates_expected": len(expected_crates)},
                )
            )

            # T2: read_symbol TokenUsageStats::reset
            t0 = time.perf_counter()
            c2, r2 = _ipc_request(
                core_port,
                token,
                "POST",
                "/v1/agent/run",
                {
                    "prompt": "Inspect TokenUsageStats::reset using read_symbol and explain what it resets.",
                    "mode": "auto",
                },
                timeout_s=240.0,
            )
            ms2 = round((time.perf_counter() - t0) * 1000.0, 2)
            ans2 = r2.get("final_answer", "") if isinstance(r2, dict) else ""
            used_read_sym = any(
                rc.get("tool") == "read_symbol" for rc in (r2.get("receipts", []) if isinstance(r2, dict) else [])
            )
            t2_ok = c2 == 200 and bool(ans2.strip()) and (
                "reset" in ans2.lower() or "token" in ans2.lower() or "usage" in ans2.lower()
            )
            tasks_out.append(
                _live_task_entry(
                    "T2_read_symbol_reset",
                    t2_ok,
                    ms2,
                    r2,
                    {"used_read_symbol": used_read_sym},
                )
            )

            # T3: cross-file search_text final_only_gbnf
            t0 = time.perf_counter()
            c3, r3 = _ipc_request(
                core_port,
                token,
                "POST",
                "/v1/agent/run",
                {
                    "prompt": "Find which Rust files define or call `final_only_gbnf` using `search_text`.",
                    "mode": "auto",
                },
                timeout_s=240.0,
            )
            ms3 = round((time.perf_counter() - t0) * 1000.0, 2)
            ans3 = r3.get("final_answer", "") if isinstance(r3, dict) else ""
            t3_ok = (
                c3 == 200
                and "context.rs" in ans3
                and "orchestrator.rs" in ans3
                and "sandbox.rs" not in ans3
            )
            tasks_out.append(_live_task_entry("T3_search_final_only_gbnf", t3_ok, ms3, r3))

            # T4: Switch workspace -> edit src/math.rs on disk + rollback
            ws_code, _ = _ipc_request(
                core_port,
                token,
                "POST",
                "/v1/workspace/select",
                {"path": str(sandbox_ws)},
                timeout_s=30.0,
            )
            t0 = time.perf_counter()
            c4, r4 = _ipc_request(
                core_port,
                token,
                "POST",
                "/v1/agent/run",
                {
                    "prompt": (
                        "Edit `src/math.rs` so `apply_tax(price: i32) -> i32` computes `price * 110 / 100` "
                        "while preserving the exact `i32` types."
                    ),
                    "mode": "auto",
                },
                timeout_s=240.0,
            )
            ms4 = round((time.perf_counter() - t0) * 1000.0, 2)
            math_file = sandbox_ws / "src" / "math.rs"
            after_edit = math_file.read_text(encoding="utf-8") if math_file.is_file() else ""
            edited_ok = (
                ws_code == 200
                and c4 == 200
                and bool(re.search(r"\(?\s*price\s*\*\s*110\s*\)?\s*/\s*100", after_edit))
                and "-> i32" in after_edit
            )
            rollback_ok = False
            snap_id = r4.get("snapshot_id") or r4.get("turn_id") if isinstance(r4, dict) else None
            if edited_ok and snap_id:
                rc, _ = _ipc_request(
                    core_port,
                    token,
                    "POST",
                    "/v1/snapshots/rollback",
                    {"turn_id": snap_id},
                    timeout_s=30.0,
                )
                after_rb = math_file.read_text(encoding="utf-8") if math_file.is_file() else ""
                rollback_ok = rc == 200 and "price + 1" in after_rb

            ans4 = r4.get("final_answer", "") if isinstance(r4, dict) else ""
            tasks_out.append(
                _live_task_entry(
                    "T4_edit_and_rollback",
                    edited_ok and rollback_ok,
                    ms4,
                    r4,
                    {"edited_ok": edited_ok, "rollback_ok": rollback_ok},
                )
            )

            # T5: Anti-regression hygiene across T1..T4 answers
            all_answers = [ans1, ans2, ans3, ans4]
            t5_ok = all(
                not is_emoji_or_banned_symbol(a)
                and not _RAW_JSON_LEAK_RE.search(a)
                and not has_repetition_loop(a)
                for a in all_answers
            )
            print(
                f"  [Cat10] T5_hygiene -> {'PASS' if t5_ok else 'FAIL'}",
                flush=True,
            )
            tasks_out.append(
                {
                    "id": "T5_hygiene_zero_emoji_no_json_leak",
                    "passed": t5_ok,
                }
            )

            passed_cnt = sum(1 for t in tasks_out if t.get("passed"))
            return {
                "passed": passed_cnt,
                "total": len(tasks_out),
                "pass_rate": round(passed_cnt / max(len(tasks_out), 1), 3),
                "ctx_profile": ctx_profile,
                "nc_plan_ngl": plan.get("n_gpu_layers"),
                "peak_vram_mib": vram_mon.peak_mib,
                "tasks": tasks_out,
            }
    finally:
        if core_port and token:
            try:
                _ipc_request(core_port, token, "POST", "/v1/shutdown", {}, timeout_s=5.0)
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(portable_root, ignore_errors=True)

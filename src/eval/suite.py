"""Overnight benchmark orchestration, artifact downloader, and Markdown report builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from .live_nc import run_live_nc_suite
from .metrics import LlamaServerSession, aggregate_telemetry
from .probes import (
    ensure_multimodal_assets,
    run_adaptive_thinking_probes,
    run_coding_probes,
    run_math_probes,
    run_multimodal_probes,
    run_native_tool_probes,
    run_per_loop_probes,
    run_vocab_diagnostics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_DIR = Path(r"D:\Coding\nuclear_code\runtime\windows-x64\vulkan")
DEFAULT_SERVER_EXE = DEFAULT_RUNTIME_DIR / "llama-server.exe"
DEFAULT_MTMD_EXE = DEFAULT_RUNTIME_DIR / "llama-mtmd-cli.exe"
DEFAULT_TOKENIZE_EXE = DEFAULT_RUNTIME_DIR / "llama-tokenize.exe"
FAT32_MAX_BYTES = 4_294_967_295


@dataclass(frozen=True)
class ModelArtifactSpec:
    """File paths and remote Hugging Face / Modal sources for one model family."""

    key: str
    display_name: str
    kind: str  # "instruction" | "nuclear_agent"
    family: str  # "e2b" | "e4b"
    hf_repo: str
    hf_subfolder: str
    mmproj_repo: str
    mmproj_subfolder: str
    quants: dict[str, str]
    mmproj_filename: str
    local_dir: Path
    reuse_paths: dict[str, Path]
    modal_dir: str | None = None


def build_model_specs(brain_dir: Path | None = None) -> dict[str, ModelArtifactSpec]:
    """Return canonical benchmark registry covering all 4 Gemma 4 models."""
    bdir = Path(brain_dir) if brain_dir is not None else (REPO_ROOT / "brain")
    return {
        "e2b_inst": ModelArtifactSpec(
            key="e2b_inst",
            display_name="Gemma 4 E2B Instruction",
            kind="instruction",
            family="e2b",
            hf_repo="unsloth/gemma-4-E2B-it-GGUF",
            hf_subfolder="",
            mmproj_repo="ggml-org/gemma-4-E2B-it-GGUF",
            mmproj_subfolder="",
            quants={
                "IQ4_XS": "gemma-4-E2B-it-IQ4_XS.gguf",
                "Q4_K_M": "gemma-4-E2B-it-Q4_K_M.gguf",
            },
            mmproj_filename="mmproj-gemma-4-E2B-it-Q8_0.gguf",
            local_dir=bdir / "baselines" / "e2b",
            reuse_paths={"Q4_K_M": bdir / "gemma-4-E2B-it-Q4_K_M.gguf"},
        ),
        "e4b_inst": ModelArtifactSpec(
            key="e4b_inst",
            display_name="Gemma 4 E4B Instruction",
            kind="instruction",
            family="e4b",
            hf_repo="unsloth/gemma-4-E4B-it-GGUF",
            hf_subfolder="",
            mmproj_repo="ggml-org/gemma-4-E4B-it-GGUF",
            mmproj_subfolder="",
            quants={
                "IQ4_XS": "gemma-4-E4B-it-IQ4_XS.gguf",
                "Q4_K_M": "gemma-4-E4B-it-Q4_K_M.gguf",
            },
            mmproj_filename="mmproj-gemma-4-E4B-it-Q8_0.gguf",
            local_dir=bdir / "baselines" / "e4b",
            reuse_paths={},
        ),
        "e4b_nc": ModelArtifactSpec(
            key="e4b_nc",
            display_name="Gemma 4 E4B Nuclear Agent",
            kind="nuclear_agent",
            family="e4b",
            hf_repo="haku-nekooo/gemma4-e4b-nuclear-agent",
            hf_subfolder="gguf",
            mmproj_repo="haku-nekooo/gemma4-e4b-nuclear-agent",
            mmproj_subfolder="gguf",
            quants={
                "IQ4_XS": "gemma4-e4b-nc-IQ4_XS.gguf",
                "Q4_K_M": "gemma4-e4b-nc-Q4_K_M.gguf",
            },
            mmproj_filename="mmproj-gemma4-e4b-nc-q8_0.gguf",
            local_dir=bdir / "agents" / "e4b",
            reuse_paths={},
            modal_dir="/outputs/e4b/gguf",
        ),
        "e2b_nc": ModelArtifactSpec(
            key="e2b_nc",
            display_name="Gemma 4 E2B Nuclear Agent",
            kind="nuclear_agent",
            family="e2b",
            hf_repo="haku-nekooo/gemma4-e2b-nuclear-agent",
            hf_subfolder="gguf",
            mmproj_repo="haku-nekooo/gemma4-e2b-nuclear-agent",
            mmproj_subfolder="gguf",
            quants={
                "IQ4_XS": "gemma4-e2b-nc-IQ4_XS.gguf",
                "Q4_K_M": "gemma4-e2b-nc-Q4_K_M.gguf",
            },
            mmproj_filename="mmproj-gemma4-e2b-nc-q8_0.gguf",
            local_dir=bdir / "agents" / "e2b",
            reuse_paths={},
            modal_dir="/outputs/e2b/gguf",
        ),
        "e4b_nc_v2": ModelArtifactSpec(
            key="e4b_nc_v2",
            display_name="Gemma 4 E4B Nuclear Agent v2",
            kind="nuclear_agent",
            family="e4b",
            hf_repo="haku-nekooo/gemma4-e4b-nuclear-agent-v2",
            hf_subfolder="gguf",
            mmproj_repo="haku-nekooo/gemma4-e4b-nuclear-agent-v2",
            mmproj_subfolder="gguf",
            quants={
                "IQ4_XS": "gemma4-e4b-nc-IQ4_XS.gguf",
                "Q4_K_M": "gemma4-e4b-nc-Q4_K_M.gguf",
            },
            mmproj_filename="mmproj-gemma4-e4b-nc-q8_0.gguf",
            local_dir=bdir / "agents" / "e4b_v2",
            reuse_paths={},
            modal_dir="/outputs/e4b_v2/gguf",
        ),
        "e2b_nc_v2": ModelArtifactSpec(
            key="e2b_nc_v2",
            display_name="Gemma 4 E2B Nuclear Agent v2",
            kind="nuclear_agent",
            family="e2b",
            hf_repo="haku-nekooo/gemma4-e2b-nuclear-agent-v2",
            hf_subfolder="gguf",
            mmproj_repo="haku-nekooo/gemma4-e2b-nuclear-agent-v2",
            mmproj_subfolder="gguf",
            quants={
                "IQ4_XS": "gemma4-e2b-nc-IQ4_XS.gguf",
                "Q4_K_M": "gemma4-e2b-nc-Q4_K_M.gguf",
            },
            mmproj_filename="mmproj-gemma4-e2b-nc-q8_0.gguf",
            local_dir=bdir / "agents" / "e2b_v2",
            reuse_paths={},
            modal_dir="/outputs/e2b_v2/gguf",
        ),
        "e4b_nc_v2_1": ModelArtifactSpec(
            key="e4b_nc_v2_1",
            display_name="Gemma 4 E4B Nuclear Agent v2.1",
            kind="nuclear_agent",
            family="e4b",
            hf_repo="haku-nekooo/gemma4-e4b-nuclear-agent-v2.1",
            hf_subfolder="gguf",
            mmproj_repo="haku-nekooo/gemma4-e4b-nuclear-agent-v2.1",
            mmproj_subfolder="gguf",
            quants={
                "IQ4_XS": "gemma4-e4b-nc-IQ4_XS.gguf",
                "Q4_K_M": "gemma4-e4b-nc-Q4_K_M.gguf",
            },
            mmproj_filename="mmproj-gemma4-e4b-nc-q8_0.gguf",
            local_dir=bdir / "agents" / "e4b_v2_1",
            reuse_paths={},
            modal_dir="/outputs/e4b_v2_1/gguf",
        ),
        "e2b_nc_v2_1": ModelArtifactSpec(
            key="e2b_nc_v2_1",
            display_name="Gemma 4 E2B Nuclear Agent v2.1",
            kind="nuclear_agent",
            family="e2b",
            hf_repo="haku-nekooo/gemma4-e2b-nuclear-agent-v2.1",
            hf_subfolder="gguf",
            mmproj_repo="haku-nekooo/gemma4-e2b-nuclear-agent-v2.1",
            mmproj_subfolder="gguf",
            quants={
                "IQ4_XS": "gemma4-e2b-nc-IQ4_XS.gguf",
                "Q4_K_M": "gemma4-e2b-nc-Q4_K_M.gguf",
            },
            mmproj_filename="mmproj-gemma4-e2b-nc-q8_0.gguf",
            local_dir=bdir / "agents" / "e2b_v2_1",
            reuse_paths={},
            modal_dir="/outputs/e2b_v2_1/gguf",
        ),
    }


MODEL_SPECS = build_model_specs()


def resolve_local_gguf_path(spec: ModelArtifactSpec, quant: str) -> Path:
    """Return reused path if present on disk, else the target cache path under brain/."""
    reused = spec.reuse_paths.get(quant)
    if reused is not None and reused.is_file() and reused.stat().st_size > 1_000_000:
        return reused
    return spec.local_dir / spec.quants[quant]


def resolve_local_mmproj_path(spec: ModelArtifactSpec) -> Path:
    return spec.local_dir / spec.mmproj_filename


def _hf_download_clean(
    repo_id: str,
    filename: str,
    subfolder: str,
    dest_path: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file() and dest_path.stat().st_size > 1_000_000:
        return dest_path
    remote_path = f"{subfolder}/{filename}" if subfolder else filename
    cached = Path(hf_hub_download(repo_id=repo_id, filename=remote_path))
    shutil.copy2(cached, dest_path)
    return dest_path


def download_baseline_artifacts(
    *,
    brain_dir: Path | None = None,
    quants: tuple[str, ...] = ("IQ4_XS", "Q4_K_M"),
) -> dict[str, dict[str, str]]:
    """Download public Gemma 4 E2B/E4B Instruction GGUFs and Q8_0 mmproj files."""
    specs = build_model_specs(brain_dir)
    downloaded: dict[str, dict[str, str]] = {}
    for key in ("e2b_inst", "e4b_inst"):
        spec = specs[key]
        out: dict[str, str] = {}
        for q in quants:
            dest = resolve_local_gguf_path(spec, q)
            if not dest.is_file() or dest.stat().st_size <= 1_000_000:
                print(f"[download] {spec.hf_repo}/{spec.quants[q]} -> {dest}")
                _hf_download_clean(spec.hf_repo, spec.quants[q], spec.hf_subfolder, dest)
            else:
                print(f"[reuse]    {dest}")
            out[q] = str(dest)
        mm_dest = resolve_local_mmproj_path(spec)
        if not mm_dest.is_file() or mm_dest.stat().st_size <= 1_000_000:
            print(f"[download] {spec.mmproj_repo}/{spec.mmproj_filename} -> {mm_dest}")
            _hf_download_clean(spec.mmproj_repo, spec.mmproj_filename, spec.mmproj_subfolder, mm_dest)
        else:
            print(f"[reuse]    {mm_dest}")
        out["mmproj"] = str(mm_dest)
        downloaded[key] = out
    return downloaded


def _modal_volume_get(remote_path: str, dest_path: Path) -> None:
    modal_exe = Path(sys.executable).parent / ("modal.exe" if sys.platform == "win32" else "modal")
    cmd = [str(modal_exe) if modal_exe.is_file() else "modal", "volume", "get", "nuclear-distill-vol", remote_path, str(dest_path)]
    subprocess.run(cmd, stdin=subprocess.DEVNULL, check=True)


def download_agent_artifacts(
    variant: str,
    *,
    brain_dir: Path | None = None,
    quants: tuple[str, ...] = ("IQ4_XS", "Q4_K_M"),
) -> dict[str, str]:
    """Pull trained Nuclear Agent GGUFs + Q8_0 mmproj from HF (Modal Volume fallback)."""
    alias_map = {
        "e4b": "e4b_nc",
        "e2b": "e2b_nc",
        "e4b_v2": "e4b_nc_v2",
        "e2b_v2": "e2b_nc_v2",
        "e4b_v2_1": "e4b_nc_v2_1",
        "e2b_v2_1": "e2b_nc_v2_1",
    }
    norm = variant.strip().lower().replace(".", "_")
    key = alias_map.get(norm, norm if "_nc_" in norm else f"{norm}_nc")
    specs = build_model_specs(brain_dir)
    if key not in specs:
        raise ValueError(
            f"Unknown agent variant {variant!r}; expected e2b, e4b, e2b_v2, e4b_v2, e2b_v2_1, or e4b_v2_1"
        )
    spec = specs[key]
    out: dict[str, str] = {}
    targets = [(q, spec.quants[q], resolve_local_gguf_path(spec, q)) for q in quants]
    targets.append(("mmproj", spec.mmproj_filename, resolve_local_mmproj_path(spec)))

    for tag, fname, dest in targets:
        if dest.is_file() and dest.stat().st_size > 1_000_000:
            print(f"[reuse]    {dest}")
            out[tag] = str(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"[download] {spec.hf_repo}/{spec.hf_subfolder}/{fname} -> {dest}")
            _hf_download_clean(spec.hf_repo, fname, spec.hf_subfolder, dest)
        except Exception as hf_exc:  # noqa: BLE001
            if spec.modal_dir:
                remote = f"{spec.modal_dir}/{fname}"
                print(f"[fallback modal volume] {remote} -> {dest} (HF err: {hf_exc})")
                _modal_volume_get(remote, dest)
            else:
                raise
        out[tag] = str(dest)
    return out


# ---------------------------------------------------------------------------
# Per-(model, quant) end-to-end 10-category benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark_for_variant(
    *,
    model_key: str,
    quant: str,
    work_root: Path,
    brain_dir: Path | None = None,
    server_exe: Path = DEFAULT_SERVER_EXE,
    mtmd_exe: Path = DEFAULT_MTMD_EXE,
    tokenize_exe: Path = DEFAULT_TOKENIZE_EXE,
    run_live_nc: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Execute all 10 benchmark categories for a single (model_key, quant) pair."""
    specs = build_model_specs(Path(brain_dir).resolve() if brain_dir is not None else None)
    spec = specs[model_key]
    gguf_path = resolve_local_gguf_path(spec, quant).resolve()
    mmproj_path = resolve_local_mmproj_path(spec).resolve()
    work_root = Path(work_root).resolve()
    if not gguf_path.is_file():
        raise FileNotFoundError(f"Missing GGUF for {model_key}/{quant}: {gguf_path}")

    results_dir = Path(work_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{model_key}_{quant}.json"
    if out_json.is_file() and not force:
        print(f"[skip] already completed: {out_json}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    assets = ensure_multimodal_assets(Path(work_root) / "assets")
    logs_dir = Path(work_root) / "logs" / f"{model_key}_{quant}"
    size_bytes = gguf_path.stat().st_size

    # Category 2: Vocabulary Surgery & Tokenizer Diagnostics
    print(f"\n=== [{spec.display_name} | {quant}] Cat 2: Vocab Diagnostics ===", flush=True)
    vocab_res = run_vocab_diagnostics(tokenize_exe, gguf_path)

    # Categories 1, 3, 4, 5, 7, 8, 9 inside a single text-only llama-server session
    print(f"=== [{spec.display_name} | {quant}] Cat 1,3,4,5,7,8,9: Text Server Bench ===", flush=True)
    all_telems = []
    with LlamaServerSession(
        server_exe=server_exe,
        model_path=gguf_path,
        ctx_size=8192,
        ple_cpu_offload=True,
        log_dir=logs_dir,
    ) as srv:
        nt_res, nt_tels = run_native_tool_probes(srv.host, srv.port)
        per_res, per_tels = run_per_loop_probes(srv.host, srv.port)
        adapt_res, adapt_tels = run_adaptive_thinking_probes(srv.host, srv.port)
        code_res, code_tels = run_coding_probes(srv.host, srv.port)
        math_res, math_tels = run_math_probes(srv.host, srv.port)

        all_telems.extend([t.to_dict() for t in (*nt_tels, *per_tels, *adapt_tels, *code_tels, *math_tels)])
        text_vram = {
            "idle_vram_mib": srv.idle_vram_mib,
            "loaded_vram_mib": srv.loaded_vram_mib,
            "peak_vram_mib": srv.current_peak_vram_mib(),
            "delta_loaded_mib": max(0, srv.loaded_vram_mib - srv.idle_vram_mib),
            "ngl_used": srv.ngl_used,
            "boot_ms": srv.boot_ms,
        }

    # Category 6: Multimodal Eyes & Ears (mmproj-q8_0)
    if mmproj_path.is_file():
        print(f"=== [{spec.display_name} | {quant}] Cat 6: Multimodal (mmproj-q8_0) ===", flush=True)
        mm_res = run_multimodal_probes(
            server_exe=server_exe,
            mtmd_exe=mtmd_exe,
            model_path=gguf_path,
            mmproj_path=mmproj_path,
            assets=assets,
            log_dir=logs_dir,
        )
        all_telems.extend(mm_res.get("telemetries", []))
    else:
        mm_res = {"skipped": True, "reason": f"missing {mmproj_path}"}

    # Category 10: Live nuclear-code (nc-core.exe T1..T5 + /rollback)
    if run_live_nc:
        print(f"=== [{spec.display_name} | {quant}] Cat 10: Live nuclear-code T1..T5 ===", flush=True)
        nc_ctx = "16k" if spec.family == "e2b" else "8k"
        live_res = run_live_nc_suite(
            gguf_path=gguf_path,
            work_dir=Path(work_root) / "live_nc" / f"{model_key}_{quant}",
            ctx_profile=nc_ctx,
        )
    else:
        live_res = {"skipped": True, "reason": "skipped via --skip-live-nc"}

    telemetry_summary = aggregate_telemetry(all_telems)
    report_obj: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "display_name": spec.display_name,
        "kind": spec.kind,
        "family": spec.family,
        "quant": quant,
        "gguf_path": str(gguf_path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 1),
        "fits_fat32": size_bytes <= FAT32_MAX_BYTES,
        "cat1_vram_speed": text_vram,
        "cat2_vocab_surgery": vocab_res,
        "cat3_native_tools": nt_res,
        "cat4_per_loop": per_res,
        "cat5_adaptive_thinking": adapt_res,
        "cat6_multimodal": mm_res,
        "cat7_coding_algorithms": code_res,
        "cat8_math_computation": math_res,
        "cat9_verbose_telemetry": telemetry_summary,
        "cat10_live_nuclear_code": live_res,
    }
    out_json.write_text(json.dumps(report_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[saved] {out_json} | VRAM={text_vram['loaded_vram_mib']}/{text_vram['peak_vram_mib']}MiB "
        f"(ngl={text_vram['ngl_used']}) | avg_ttfb={telemetry_summary.get('avg_ttfb_ms')}ms "
        f"pp={telemetry_summary.get('avg_pp_tps')} tg={telemetry_summary.get('avg_tg_tps')} | "
        f"NT={nt_res['passed']}/{nt_res['total']} PER={per_res['passed']}/{per_res['total']} "
        f"Adapt={adapt_res['passed']}/{adapt_res['total']} MM={mm_res.get('passed', '-')}/{mm_res.get('total', '-')} "
        f"Code={code_res['passed']}/{code_res['total']} Math={math_res['thinking_on_passed']}/{math_res['thinking_on_total']} "
        f"NC={live_res.get('passed', '-')}/{live_res.get('total', '-')}",
        flush=True,
    )
    return report_obj


def refresh_live_nc_for_variant(
    *,
    model_key: str,
    quant: str,
    work_root: Path,
    brain_dir: Path | None = None,
    runner_fn: Any | None = None,
) -> dict[str, Any]:
    """Re-run only Cat 10 live nc-core checks and patch the saved variant JSON."""
    specs = build_model_specs(Path(brain_dir).resolve() if brain_dir is not None else None)
    spec = specs[model_key]
    gguf_path = resolve_local_gguf_path(spec, quant).resolve()
    work_root = Path(work_root).resolve()
    out_json = work_root / "results" / f"{model_key}_{quant}.json"
    if not out_json.is_file():
        raise FileNotFoundError(f"Cannot refresh live_nc before base result exists: {out_json}")
    if not gguf_path.is_file():
        raise FileNotFoundError(f"Missing GGUF for {model_key}/{quant}: {gguf_path}")

    print(f"=== [{spec.display_name} | {quant}] Refresh Cat 10: Live nuclear-code ===", flush=True)
    nc_ctx = "16k" if spec.family == "e2b" else "8k"
    fn = runner_fn or run_live_nc_suite
    live_res = fn(
        gguf_path=gguf_path,
        work_dir=work_root / "live_nc" / f"{model_key}_{quant}",
        ctx_profile=nc_ctx,
    )
    report_obj = json.loads(out_json.read_text(encoding="utf-8"))
    report_obj["cat10_live_nuclear_code"] = live_res
    out_json.write_text(json.dumps(report_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[refreshed] {out_json} | NC={live_res.get('passed', '-')}/{live_res.get('total', '-')}",
        flush=True,
    )
    return report_obj


# ---------------------------------------------------------------------------
# Aggregation & Markdown Report Generator
# ---------------------------------------------------------------------------


def _fmt_rate(section: dict[str, Any], pass_k: str = "passed", tot_k: str = "total") -> str:
    if section.get("skipped"):
        return "SKIP"
    p = section.get(pass_k)
    t = section.get(tot_k)
    if p is None or t is None:
        return "-"
    pct = round(100.0 * p / max(t, 1))
    return f"{p}/{t} ({pct}%)"


def _fmt_num(val: Any, suffix: str = "") -> str:
    if val is None:
        return "-"
    return f"{val}{suffix}"


def render_markdown_report(records: list[dict[str, Any]]) -> str:
    """Build a multi-table Markdown comparison report covering all 10 categories."""
    order = {
        "e2b_inst": 0,
        "e2b_nc": 1,
        "e2b_nc_v2": 2,
        "e2b_nc_v2_1": 3,
        "e4b_inst": 4,
        "e4b_nc": 5,
        "e4b_nc_v2": 6,
        "e4b_nc_v2_1": 7,
    }
    q_order = {"IQ4_XS": 0, "Q4_K_M": 1}
    rows = sorted(
        records,
        key=lambda r: (order.get(r.get("model_key", ""), 99), q_order.get(r.get("quant", ""), 99)),
    )

    lines: list[str] = [
        "# Overnight Gemma 4 Nuclear Agent vs Instruction Benchmark Report",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}` | Variants tested: **{len(rows)}** | Hardware: **GTX 1650 Ti (4,096 MiB VRAM)** (`-ot per_layer_token_embd\\.weight=CPU`, KV `q8_0`, `-fa on`).",
        "",
        "## 1. Executive Scorecard (Categories 2–10)",
        "",
        "| Model | Quant | Native Tools (Cat 3) | PER Loop (Cat 4) | Adaptive Think / d=1 (Cat 5) | Multimodal PNG+WAV (Cat 6) | Coding Exec (Cat 7) | Math On / Off (Cat 8) | Vocab `<unk>` (Cat 2) | Live `nc-core` T1–T5 (Cat 10) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        math_s = r.get("cat8_math_computation", {})
        math_on = f"{math_s.get('thinking_on_passed', '-')}/{math_s.get('thinking_on_total', '-')}"
        math_off = f"{math_s.get('thinking_off_passed', '-')}/{math_s.get('thinking_off_total', '-')}"
        vocab_s = r.get("cat2_vocab_surgery", {})
        unk_str = f"{vocab_s.get('total_unks', '-')} (`{vocab_s.get('avg_chars_per_token', '-')} ch/tok`)"
        lines.append(
            f"| {r['display_name']} | `{r['quant']}` | "
            f"{_fmt_rate(r.get('cat3_native_tools', {}))} | "
            f"{_fmt_rate(r.get('cat4_per_loop', {}))} | "
            f"{_fmt_rate(r.get('cat5_adaptive_thinking', {}))} | "
            f"{_fmt_rate(r.get('cat6_multimodal', {}))} | "
            f"{_fmt_rate(r.get('cat7_coding_algorithms', {}))} | "
            f"{math_on} / {math_off} | "
            f"{unk_str} | "
            f"{_fmt_rate(r.get('cat10_live_nuclear_code', {}))} |"
        )

    lines.extend(
        [
            "",
            "## 2. Category 1 & 9 — VRAM Budget, Speed & Full Verbose Telemetry",
            "",
            "| Model | Quant | GGUF MB | FAT32 | `-ngl` (Text / MM) | Boot ms | Text VRAM (Load / Peak MiB) | +mmproj VRAM (Load / Peak MiB) | Avg TTFB ms | p95 TTFB ms | pp tok/s | tg tok/s | Prompt / Comp / Reason / Cached Tokens |",
            "|---|---|---:|:---:|:---:|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for r in rows:
        v = r.get("cat1_vram_speed", {})
        mm = r.get("cat6_multimodal", {})
        t = r.get("cat9_verbose_telemetry", {})
        ngl_str = f"{_fmt_num(v.get('ngl_used'))} / {_fmt_num(mm.get('ngl_used'))}"
        txt_vram = f"{_fmt_num(v.get('loaded_vram_mib'))} / {_fmt_num(v.get('peak_vram_mib'))}"
        mm_vram = (
            "SKIP"
            if mm.get("skipped")
            else f"{_fmt_num(mm.get('loaded_vram_mib'))} / {_fmt_num(mm.get('peak_vram_mib'))}"
        )
        toks = (
            f"{t.get('prompt_tokens_total', 0):,} / "
            f"{t.get('completion_tokens_total', 0):,} / "
            f"{t.get('reasoning_tokens_total', 0):,} / "
            f"{t.get('cached_tokens_total', 0):,}"
        )
        lines.append(
            f"| {r['display_name']} | `{r['quant']}` | {r.get('size_mb', '-')} | "
            f"{'YES' if r.get('fits_fat32') else 'NO'} | {ngl_str} | {_fmt_num(v.get('boot_ms'))} | "
            f"{txt_vram} | {mm_vram} | {_fmt_num(t.get('avg_ttfb_ms'))} | {_fmt_num(t.get('p95_ttfb_ms'))} | "
            f"{_fmt_num(t.get('avg_pp_tps'))} | {_fmt_num(t.get('avg_tg_tps'))} | {toks} |"
        )

    lines.extend(["", "## 3. Per-Category Drilldown", ""])
    for r in rows:
        lines.append(f"### {r['display_name']} (`{r['quant']}`)")
        for label, sec_key, list_key in (
            ("Cat 3 Native Tools", "cat3_native_tools", "items"),
            ("Cat 4 PER Loop", "cat4_per_loop", "items"),
            ("Cat 5 Adaptive Thinking", "cat5_adaptive_thinking", "items"),
            ("Cat 6 Multimodal", "cat6_multimodal", "items"),
            ("Cat 7 Coding Exec", "cat7_coding_algorithms", "items"),
            ("Cat 10 Live `nc-core`", "cat10_live_nuclear_code", "tasks"),
        ):
            sec = r.get(sec_key, {})
            if sec.get("skipped"):
                lines.append(f"- **{label}**: SKIP ({sec.get('reason', '')})")
                continue
            item_list = sec.get(list_key, [])
            summary_bits = [
                f"`{it.get('id', '?')}`={'PASS' if it.get('passed') else 'FAIL'}"
                for it in item_list
            ]
            lines.append(f"- **{label}**: {', '.join(summary_bits) if summary_bits else '-'}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def aggregate_results(results_dir: Path, report_out: Path | None = None) -> dict[str, Any]:
    """Load all completed `<model>_<quant>.json` files, write summary JSON and Markdown report."""
    rdir = Path(results_dir)
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(rdir.glob("*.json"))
        if p.name != "overnight_summary.json"
    ]
    summary_path = rdir.parent / "overnight_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    md = render_markdown_report(records)
    if report_out is not None:
        out_p = Path(report_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md, encoding="utf-8")
    return {"count": len(records), "summary_json": str(summary_path), "report_md": str(report_out) if report_out else None}

"""Modular GGUF exporter (IQ4_XS / Q4_K_M with --imatrix + bf16/q8_0 --mmproj)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

FAT32_MAX_BYTES = 4_294_967_295
DEFAULT_QUANTS: tuple[str, ...] = ("IQ4_XS", "Q4_K_M")
DEFAULT_MMPROJ_OUTTYPES: tuple[str, ...] = ("bf16", "q8_0")
DEFAULT_PLE_CPU_OFFLOAD = r'per_layer_token_embd\.weight=CPU'


def _sha256_file(path: Path, chunk_bytes: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def write_imatrix_calib_corpus(
    sft_jsonl: str | Path,
    out_txt: str | Path,
    max_samples: int = 128,
) -> Path:
    """Sample evenly across compiled Gemma 4 SFT buckets to build an imatrix calibration file."""
    out = Path(out_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    for raw in Path(sft_jsonl).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        txt = row.get("text")
        if isinstance(txt, str) and txt.strip():
            texts.append(txt.strip())

    if len(texts) > max_samples > 0:
        step = len(texts) / max_samples
        texts = [texts[int(i * step)] for i in range(max_samples)]

    out.write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    return out


def _normalize_mmproj_types(
    mmproj_outtypes: tuple[str, ...] | list[str] | None,
    mmproj_outtype: str | None,
) -> tuple[str, ...]:
    if mmproj_outtype is not None:
        return (mmproj_outtype.strip().lower(),)
    if mmproj_outtypes is None:
        return DEFAULT_MMPROJ_OUTTYPES
    return tuple(str(t).strip().lower() for t in mmproj_outtypes if str(t).strip())


def build_export_commands(
    model_dir: str | Path,
    out_dir: str | Path,
    llama_cpp_dir: str | Path,
    *,
    model_prefix: str = "gemma4-e4b-nc",
    quants: tuple[str, ...] | list[str] = DEFAULT_QUANTS,
    mmproj_outtypes: tuple[str, ...] | list[str] | None = DEFAULT_MMPROJ_OUTTYPES,
    mmproj_outtype: str | None = None,
    mmproj_model_dir: str | Path | None = None,
    imatrix_calib_file: str | Path | None = None,
    imatrix_chunks: int = 128,
    imatrix_ngl: int = 0,
    python_exe: str | None = None,
) -> dict[str, list[str]]:
    """Build exact argv lists for convert_hf_to_gguf.py, llama-imatrix, and llama-quantize."""
    py = python_exe or sys.executable
    model = Path(model_dir)
    mmproj_src = Path(mmproj_model_dir) if mmproj_model_dir is not None else model
    out = Path(out_dir)
    llama = Path(llama_cpp_dir)

    convert_py = llama / "convert_hf_to_gguf.py"
    bin_dir = llama / "build" / "bin"
    quantize_bin = bin_dir / "llama-quantize"
    imatrix_bin = bin_dir / "llama-imatrix"

    bf16_gguf = out / f"{model_prefix}-bf16.gguf"
    imatrix_path = out / f"{model_prefix}.imatrix"

    cmds: dict[str, list[str]] = {
        "convert_text_bf16": [
            py,
            str(convert_py),
            str(model),
            "--outfile",
            str(bf16_gguf),
            "--outtype",
            "bf16",
        ],
    }

    for mm_t in _normalize_mmproj_types(mmproj_outtypes, mmproj_outtype):
        mmproj_gguf = out / f"mmproj-{model_prefix}-{mm_t}.gguf"
        cmds[f"convert_mmproj_{mm_t}"] = [
            py,
            str(convert_py),
            str(mmproj_src),
            "--mmproj",
            "--outfile",
            str(mmproj_gguf),
            "--outtype",
            mm_t,
        ]

    use_imatrix = imatrix_calib_file is not None
    if use_imatrix:
        imatrix_cmd = [
            str(imatrix_bin),
            "-m",
            str(bf16_gguf),
            "-f",
            str(imatrix_calib_file),
            "-o",
            str(imatrix_path),
            "--chunks",
            str(imatrix_chunks),
        ]
        if imatrix_ngl > 0:
            imatrix_cmd.extend(["-ngl", str(imatrix_ngl)])
        cmds["imatrix"] = imatrix_cmd

    for qtype in quants:
        q_upper = qtype.upper()
        q_out = out / f"{model_prefix}-{q_upper}.gguf"
        q_cmd = [str(quantize_bin)]
        if use_imatrix:
            q_cmd.extend(["--imatrix", str(imatrix_path)])
        q_cmd.extend([str(bf16_gguf), str(q_out), q_upper])
        cmds[f"quantize_{q_upper}"] = q_cmd

    return cmds


def export_modular_gguf(
    model_dir: str | Path,
    out_dir: str | Path,
    llama_cpp_dir: str | Path,
    *,
    model_prefix: str = "gemma4-e4b-nc",
    quants: tuple[str, ...] | list[str] = DEFAULT_QUANTS,
    mmproj_outtypes: tuple[str, ...] | list[str] | None = DEFAULT_MMPROJ_OUTTYPES,
    mmproj_outtype: str | None = None,
    mmproj_model_dir: str | Path | None = None,
    imatrix_calib_file: str | Path | None = None,
    imatrix_chunks: int = 128,
    imatrix_ngl: int = 0,
    keep_bf16: bool = False,
) -> dict[str, Any]:
    """Run BF16 GGUF conversion, bf16/q8_0 --mmproj export, optional imatrix, and quantization."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eff_mm_types = _normalize_mmproj_types(mmproj_outtypes, mmproj_outtype)
    cmds = build_export_commands(
        model_dir,
        out,
        llama_cpp_dir,
        model_prefix=model_prefix,
        quants=quants,
        mmproj_outtypes=eff_mm_types,
        mmproj_model_dir=mmproj_model_dir,
        imatrix_calib_file=imatrix_calib_file,
        imatrix_chunks=imatrix_chunks,
        imatrix_ngl=imatrix_ngl,
    )

    for step_name, cmd in cmds.items():
        print(f"[export_gguf] Running {step_name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    bf16_path = Path(cmds["convert_text_bf16"][cmds["convert_text_bf16"].index("--outfile") + 1])

    artifacts: dict[str, Any] = {}
    for qtype in quants:
        q_upper = qtype.upper()
        q_cmd = cmds[f"quantize_{q_upper}"]
        q_path = Path(q_cmd[-2])
        size_bytes = q_path.stat().st_size
        artifacts[q_upper] = {
            "path": str(q_path),
            "bytes": size_bytes,
            "gb": round(size_bytes / (1024**3), 3),
            "fat32_safe": size_bytes <= FAT32_MAX_BYTES,
            "sha256": _sha256_file(q_path),
        }

    for mm_t in eff_mm_types:
        mm_cmd = cmds[f"convert_mmproj_{mm_t}"]
        mmproj_path = Path(mm_cmd[mm_cmd.index("--outfile") + 1])
        mm_bytes = mmproj_path.stat().st_size
        artifacts[f"mmproj_{mm_t}"] = {
            "path": str(mmproj_path),
            "bytes": mm_bytes,
            "gb": round(mm_bytes / (1024**3), 3),
            "outtype": mm_t,
            "sha256": _sha256_file(mmproj_path),
        }

    if not keep_bf16 and bf16_path.exists():
        bf16_path.unlink()

    pref_q = "IQ4_XS" if "IQ4_XS" in artifacts else next(iter(artifacts))
    pref_mm = "mmproj_q8_0" if "mmproj_q8_0" in artifacts else f"mmproj_{eff_mm_types[0]}"
    main_name = Path(artifacts[pref_q]["path"]).name
    mm_name = Path(artifacts[pref_mm]["path"]).name
    recommended_cmd = (
        f'llama-server -m {main_name} --mmproj {mm_name} '
        f'-ot "{DEFAULT_PLE_CPU_OFFLOAD}" -ngl 99 --jinja -c 16384 -ctk q8_0 -ctv q8_0 -fa on'
    )

    manifest = {
        "model_dir": str(model_dir),
        "mmproj_model_dir": str(mmproj_model_dir or model_dir),
        "out_dir": str(out),
        "model_prefix": model_prefix,
        "imatrix_used": imatrix_calib_file is not None,
        "ple_cpu_offload_flag": f'-ot "{DEFAULT_PLE_CPU_OFFLOAD}"',
        "recommended_4gb_vram_cmd": recommended_cmd,
        "artifacts": artifacts,
    }
    (out / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export modular text GGUF + bf16/q8_0 --mmproj GGUF")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--mmproj-model-dir", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--llama-cpp-dir", required=True)
    p.add_argument("--model-prefix", default="gemma4-e4b-nc")
    p.add_argument("--quants", nargs="+", default=list(DEFAULT_QUANTS))
    p.add_argument("--mmproj-outtypes", nargs="+", default=list(DEFAULT_MMPROJ_OUTTYPES))
    p.add_argument("--mmproj-outtype", default=None)
    p.add_argument("--imatrix-calib-file", default=None)
    p.add_argument("--imatrix-chunks", type=int, default=128)
    p.add_argument("--imatrix-ngl", type=int, default=0)
    p.add_argument("--keep-bf16", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    manifest = export_modular_gguf(
        ns.model_dir,
        ns.out_dir,
        ns.llama_cpp_dir,
        model_prefix=ns.model_prefix,
        quants=tuple(ns.quants),
        mmproj_outtypes=tuple(ns.mmproj_outtypes),
        mmproj_outtype=ns.mmproj_outtype,
        mmproj_model_dir=ns.mmproj_model_dir,
        imatrix_calib_file=ns.imatrix_calib_file,
        imatrix_chunks=ns.imatrix_chunks,
        imatrix_ngl=ns.imatrix_ngl,
        keep_bf16=ns.keep_bf16,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

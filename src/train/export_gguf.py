"""Modular GGUF exporter (Q4_K_M / IQ4_XS with optional imatrix + separate --mmproj)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

FAT32_MAX_BYTES = 4_294_967_295


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
    max_samples: int = 64,
) -> Path:
    """Write calibration text from compiled Gemma 4 SFT samples for llama-imatrix."""
    out = Path(out_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for raw in Path(sft_jsonl).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        txt = row.get("text")
        if isinstance(txt, str) and txt.strip():
            chunks.append(txt.strip())
        if len(chunks) >= max_samples:
            break
    out.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    return out


def build_export_commands(
    model_dir: str | Path,
    out_dir: str | Path,
    llama_cpp_dir: str | Path,
    *,
    model_prefix: str = "gemma4-e4b-nc",
    quants: tuple[str, ...] | list[str] = ("Q4_K_M", "IQ4_XS"),
    mmproj_outtype: str = "bf16",
    imatrix_calib_file: str | Path | None = None,
    imatrix_chunks: int = 64,
    python_exe: str | None = None,
) -> dict[str, list[str]]:
    """Build exact argv lists for convert_hf_to_gguf.py, llama-imatrix, and llama-quantize."""
    py = python_exe or sys.executable
    model = Path(model_dir)
    out = Path(out_dir)
    llama = Path(llama_cpp_dir)

    convert_py = llama / "convert_hf_to_gguf.py"
    bin_dir = llama / "build" / "bin"
    quantize_bin = bin_dir / "llama-quantize"
    imatrix_bin = bin_dir / "llama-imatrix"

    bf16_gguf = out / f"{model_prefix}-bf16.gguf"
    mmproj_gguf = out / f"mmproj-{model_prefix}-{mmproj_outtype}.gguf"
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
        "convert_mmproj": [
            py,
            str(convert_py),
            str(model),
            "--mmproj",
            "--outfile",
            str(mmproj_gguf),
            "--outtype",
            mmproj_outtype,
        ],
    }

    use_imatrix = imatrix_calib_file is not None
    if use_imatrix:
        cmds["imatrix"] = [
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
    quants: tuple[str, ...] | list[str] = ("Q4_K_M", "IQ4_XS"),
    mmproj_outtype: str = "bf16",
    imatrix_calib_file: str | Path | None = None,
    imatrix_chunks: int = 64,
    keep_bf16: bool = False,
) -> dict[str, Any]:
    """Run BF16 GGUF conversion, --mmproj export, optional imatrix, and quantization."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmds = build_export_commands(
        model_dir,
        out,
        llama_cpp_dir,
        model_prefix=model_prefix,
        quants=quants,
        mmproj_outtype=mmproj_outtype,
        imatrix_calib_file=imatrix_calib_file,
        imatrix_chunks=imatrix_chunks,
    )

    for step_name, cmd in cmds.items():
        print(f"[export_gguf] Running {step_name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    bf16_path = Path(cmds["convert_text_bf16"][cmds["convert_text_bf16"].index("--outfile") + 1])
    mmproj_path = Path(cmds["convert_mmproj"][cmds["convert_mmproj"].index("--outfile") + 1])

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

    mm_bytes = mmproj_path.stat().st_size
    artifacts["mmproj"] = {
        "path": str(mmproj_path),
        "bytes": mm_bytes,
        "gb": round(mm_bytes / (1024**3), 3),
        "outtype": mmproj_outtype,
        "sha256": _sha256_file(mmproj_path),
    }

    if not keep_bf16 and bf16_path.exists():
        bf16_path.unlink()

    manifest = {
        "model_dir": str(model_dir),
        "out_dir": str(out),
        "model_prefix": model_prefix,
        "imatrix_used": imatrix_calib_file is not None,
        "artifacts": artifacts,
    }
    (out / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export modular text GGUF + --mmproj GGUF")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--llama-cpp-dir", required=True)
    p.add_argument("--model-prefix", default="gemma4-e4b-nc")
    p.add_argument("--quants", nargs="+", default=["Q4_K_M", "IQ4_XS"])
    p.add_argument("--mmproj-outtype", default="bf16")
    p.add_argument("--imatrix-calib-file", default=None)
    p.add_argument("--imatrix-chunks", type=int, default=64)
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
        mmproj_outtype=ns.mmproj_outtype,
        imatrix_calib_file=ns.imatrix_calib_file,
        imatrix_chunks=ns.imatrix_chunks,
        keep_bf16=ns.keep_bf16,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

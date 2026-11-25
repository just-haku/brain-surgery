"""Modal A100 end-to-end runner for Step 2: QLoRA/BF16 SFT -> ORPO -> GGUF + HF push.

Usage:
    modal run --detach scripts/run_modal_step2.py --variant e4b
    modal run --detach scripts/run_modal_step2.py --variant e2b
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

APP_NAME = "brain-surgery-step2"
VOLUME_NAME = "nuclear-distill-vol"
VOLUME_MOUNT = "/vol"
LLAMA_CPP_DIR = "/opt/llama.cpp"
HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "huggingface-secret")

VARIANT_PRESETS: dict[str, dict[str, Any]] = {
    "e4b": {
        "base_repo": "haku-nekooo/gemma4-e4b-nuclear-pruned",
        "target_repo": "haku-nekooo/gemma4-e4b-nuclear-agent-v2",
        "output_subdir": "e4b_v2",
        "learning_rate": 1.5e-4,
        "embedding_lr": 1.5e-5,
        "train_embeddings": True,
        "model_prefix": "gemma4-e4b-nc",
    },
    "e2b": {
        "base_repo": "haku-nekooo/gemma4-e2b-nuclear-pruned",
        "target_repo": "haku-nekooo/gemma4-e2b-nuclear-agent-v2",
        "output_subdir": "e2b_v2",
        "learning_rate": 2.0e-4,
        "embedding_lr": 2.0e-5,
        "train_embeddings": False,
        "model_prefix": "gemma4-e2b-nc",
    },
}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "cmake", "build-essential", "pkg-config")
    .pip_install(
        "torch>=2.5.0",
        "transformers>=5.5.0",
        "peft>=0.15.0",
        "trl>=0.16.0",
        "accelerate>=1.3.0",
        "bitsandbytes>=0.45.0",
        "datasets>=3.0.0",
        "huggingface_hub>=0.28.0",
        "hf_transfer>=0.1.8",
        "gguf",
        "sentencepiece",
        "protobuf",
        "safetensors>=0.4.5",
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR}",
        (
            "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && "
            "LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH "
            f"cmake -B {LLAMA_CPP_DIR}/build -S {LLAMA_CPP_DIR} "
            '-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80" -DLLAMA_CURL=OFF '
            '-DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" '
            '-DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" && '
            "LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH "
            f"cmake --build {LLAMA_CPP_DIR}/build --config Release "
            "--target llama-quantize llama-imatrix llama-cli -j8 && "
            "rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1"
        ),
    )
    .env(
        {
            "PYTHONPATH": "/app/src",
            "HF_HOME": f"{VOLUME_MOUNT}/cache/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(str(SRC_DIR), remote_path="/app/src", ignore=["__pycache__", "*.pyc"])
)


def resolve_variant_preset(variant: str) -> dict[str, Any]:
    """Validate and return training/export hyperparameters for `e4b` or `e2b`."""
    key = variant.strip().lower()
    if key not in VARIANT_PRESETS:
        raise ValueError(f"Unknown --variant {variant!r}; expected one of {sorted(VARIANT_PRESETS)}")
    return dict(VARIANT_PRESETS[key])


def _resolve_hf_token() -> str | None:
    for env_key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    return None


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=32768,
    timeout=14400,
    volumes={VOLUME_MOUNT: vol},
    secrets=[hf_secret],
)
def run_step2_pipeline(
    variant: str = "e4b",
    use_4bit: bool = False,
    run_sft: bool = True,
    run_orpo: bool = True,
    run_export: bool = True,
    push_to_hf: bool = True,
) -> dict[str, Any]:
    """Execute SFT -> ORPO -> GGUF (IQ4_XS/Q4_K_M + bf16/q8_0 mmproj) -> HF upload on A100."""
    from huggingface_hub import HfApi, login, snapshot_download
    from transformers import AutoTokenizer

    from train.check_token_lengths import measure_dataset_token_lengths
    from train.export_gguf import (
        DEFAULT_PLE_CPU_OFFLOAD,
        export_modular_gguf,
        write_imatrix_calib_corpus,
    )
    from train.train_orpo import ORPOConfig, run_orpo_training
    from train.train_sft import SFTConfig, run_sft_training

    preset = resolve_variant_preset(variant)
    vol.reload()

    hf_token = _resolve_hf_token()
    if push_to_hf and not hf_token:
        raise RuntimeError(
            f"Missing HF_TOKEN inside Modal Secret '{HF_SECRET_NAME}'."
        )
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    sft_path = Path(VOLUME_MOUNT) / "distill" / "sft_gemma4.jsonl"
    orpo_path = Path(VOLUME_MOUNT) / "distill" / "orpo_gemma4.jsonl"
    if not sft_path.exists() or not orpo_path.exists():
        raise FileNotFoundError(
            f"Dataset missing in Modal Volume '{VOLUME_NAME}': {sft_path} / {orpo_path}"
        )

    out_root = Path(VOLUME_MOUNT) / "outputs" / preset.get("output_subdir", variant.lower())
    base_dir = Path(VOLUME_MOUNT) / "cache" / variant.lower() / "base_ckpt"
    out_root.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=preset["base_repo"],
        local_dir=str(base_dir),
        token=hf_token,
    )
    vol.commit()

    tok = AutoTokenizer.from_pretrained(str(base_dir), use_fast=True)
    tok_report = measure_dataset_token_lengths(
        sft_path,
        orpo_path,
        tokenizer=tok,
        sft_limit=8192,
        orpo_limit=4096,
        orpo_prompt_limit=2560,
    )
    print("Token length check:", json.dumps(tok_report, indent=2))
    assert tok_report["unk_count"] == 0, f"Found {tok_report['unk_count']} <unk> tokens"
    assert tok_report["fits_100_pct"], "Dataset exceeded sequence limits"

    sft_lora_dir = out_root / "stage2a_sft_lora"
    sft_merged_dir = out_root / "stage2a_sft_merged"
    orpo_lora_dir = out_root / "stage2b_orpo_lora"
    orpo_merged_dir = out_root / "stage2b_orpo_merged"
    gguf_dir = out_root / "gguf"

    summary: dict[str, Any] = {
        "variant": variant.lower(),
        "preset": preset,
        "token_lengths": tok_report,
    }

    if run_sft:
        sft_cfg = SFTConfig(
            base_model_id=str(base_dir),
            train_jsonl=str(sft_path),
            output_dir=str(sft_lora_dir),
            merged_dir=str(sft_merged_dir),
            max_seq_length=8192,
            use_4bit=use_4bit,
            learning_rate=preset["learning_rate"],
            embedding_lr=preset["embedding_lr"],
            train_embeddings=preset["train_embeddings"],
        )
        summary["sft"] = run_sft_training(sft_cfg)
        vol.commit()
    elif not sft_merged_dir.exists():
        raise FileNotFoundError(f"skip_sft requested but {sft_merged_dir} does not exist")

    final_merged_dir = sft_merged_dir
    if run_orpo:
        orpo_cfg = ORPOConfig(
            sft_merged_dir=str(sft_merged_dir),
            orpo_jsonl=str(orpo_path),
            output_dir=str(orpo_lora_dir),
            merged_dir=str(orpo_merged_dir),
            max_length=4096,
            max_prompt_length=2560,
            use_4bit=use_4bit,
            beta=0.1,
            learning_rate=8e-6,
        )
        summary["orpo"] = run_orpo_training(orpo_cfg)
        final_merged_dir = orpo_merged_dir
        vol.commit()
    elif (orpo_merged_dir / "config.json").exists():
        final_merged_dir = orpo_merged_dir

    print(f"[run_modal_step2] Final merged checkpoint: {final_merged_dir}", flush=True)

    if run_export:
        calib_txt = write_imatrix_calib_corpus(
            sft_path, out_root / "imatrix_calib.txt", max_samples=128
        )
        export_manifest = export_modular_gguf(
            model_dir=final_merged_dir,
            out_dir=gguf_dir,
            llama_cpp_dir=LLAMA_CPP_DIR,
            model_prefix=preset["model_prefix"],
            quants=("IQ4_XS", "Q4_K_M"),
            mmproj_outtypes=("bf16", "q8_0"),
            mmproj_model_dir=base_dir,
            imatrix_calib_file=calib_txt,
            imatrix_chunks=128,
            imatrix_ngl=99,
        )

        cli_bin = f"{LLAMA_CPP_DIR}/build/bin/llama-cli"
        cli_help = subprocess.run([cli_bin, "--help"], capture_output=True, text=True).stdout
        cli_st = (
            ["--single-turn"]
            if "--single-turn" in cli_help
            else (["-no-cnv"] if "-no-cnv" in cli_help else [])
        )
        iq4_path = export_manifest["artifacts"]["IQ4_XS"]["path"]
        smoke = subprocess.run(
            [
                cli_bin,
                "-m",
                iq4_path,
                "-ot",
                DEFAULT_PLE_CPU_OFFLOAD,
                "--jinja",
                "-p",
                "Viết 1 hàm Rust tính tổng i32:",
                "-n",
                "96",
                "--no-display-prompt",
                *cli_st,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
        )
        export_manifest["smoke_preview"] = smoke.stdout[:500]
        summary["export"] = export_manifest
        vol.commit()

    if push_to_hf:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=preset["target_repo"], private=True, exist_ok=True)
        api.upload_folder(
            repo_id=preset["target_repo"],
            folder_path=str(final_merged_dir),
            commit_message=f"Modal A100 Step 2 ({variant}): merged BF16 checkpoint",
        )
        if run_export and gguf_dir.exists():
            api.upload_folder(
                repo_id=preset["target_repo"],
                folder_path=str(gguf_dir),
                path_in_repo="gguf",
                commit_message=f"Modal A100 Step 2 ({variant}): IQ4_XS/Q4_K_M + bf16/q8_0 mmproj GGUFs",
            )
        summary["hf_repo"] = f"https://huggingface.co/{preset['target_repo']}"

    (out_root / "step2_modal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    vol.commit()
    return summary


@app.local_entrypoint()
def main(
    variant: str = "e4b",
    use_4bit: bool = False,
    skip_sft: bool = False,
    skip_orpo: bool = False,
    skip_export: bool = False,
    no_push: bool = False,
) -> None:
    """Launch Step 2 training + GGUF export on Modal A100."""
    preset = resolve_variant_preset(variant)
    print(
        f"Launching Modal Step 2 ({variant}) on A100: "
        f"{preset['base_repo']} -> {preset['target_repo']} "
        f"(lr={preset['learning_rate']}, train_embeddings={preset['train_embeddings']}, use_4bit={use_4bit})"
    )
    result = run_step2_pipeline.remote(
        variant=variant,
        use_4bit=use_4bit,
        run_sft=not skip_sft,
        run_orpo=not skip_orpo,
        run_export=not skip_export,
        push_to_hf=not no_push,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

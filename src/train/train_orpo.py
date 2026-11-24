"""Stage 2B short ORPO pass over SFT-merged Gemma 4 E4B (embeddings/PLE + Vision/Audio frozen)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

from .dataset_loader import load_orpo_jsonl
from .mm_integrity import (
    freeze_multimodal_parameters,
    hash_multimodal_safetensors,
    resolve_model_dir,
)
from .train_sft import (
    SFTConfig,
    build_warmup_kwargs,
    load_gemma4_for_training,
    merge_lora_to_bf16,
    resolve_target_and_embed_modules,
)


@dataclass
class ORPOConfig:
    """Hyperparameters and paths for Stage 2B ORPO alignment pass."""

    sft_merged_dir: str = "outputs/stage2a_sft_merged"
    orpo_jsonl: str = "data/distill/orpo_gemma4.jsonl"
    output_dir: str = "outputs/stage2b_orpo_lora"
    merged_dir: str = "outputs/stage2b_orpo_merged"
    max_length: int = 4096
    max_prompt_length: int = 2560
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    beta: float = 0.1
    learning_rate: float = 8e-6
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    logging_steps: int = 5
    seed: int = 42


def freeze_embeddings_and_ple(
    model: Any,
    embed_keywords: tuple[str, ...] = ("embed_tokens", "embed_tokens_per_layer", "lm_head"),
) -> int:
    """Freeze text embeddings, PLE tables, and lm_head during ORPO to preserve alignment."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(k in name for k in embed_keywords):
            param.requires_grad = False
            frozen += 1
    return frozen


def build_orpo_config_kwargs(
    config_cls: Any,
    cfg: ORPOConfig,
    *,
    out_dir: str | Path,
    use_8bit_adam: bool = False,
) -> dict[str, Any]:
    """Build TRL ORPOConfig kwargs filtered cleanly across TRL v4/v5+ versions."""
    params = inspect.signature(config_cls.__init__).parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "beta": cfg.beta,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "logging_steps": cfg.logging_steps,
        "save_strategy": "epoch",
        "report_to": "none",
        "seed": cfg.seed,
        **build_warmup_kwargs(config_cls, cfg.warmup_ratio),
    }
    optional_candidates: dict[str, Any] = {
        "max_length": cfg.max_length,
        "max_prompt_length": cfg.max_prompt_length,
        "dataset_num_proc": 1,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_bnb_8bit" if use_8bit_adam else "adamw_torch",
    }
    for key, val in optional_candidates.items():
        if key in params or accepts_var_kw:
            kwargs[key] = val
    return kwargs


def run_orpo_training(cfg: ORPOConfig) -> dict[str, Any]:
    """Execute 1-epoch ORPO alignment with Vision/Audio and embeddings/PLE frozen."""
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    import torch

    try:
        from trl import ORPOConfig as TRLORPOConfig, ORPOTrainer
    except ImportError:
        from trl.experimental.orpo import ORPOConfig as TRLORPOConfig, ORPOTrainer

    if cfg.beta > 0.1:
        raise ValueError(f"ORPO beta={cfg.beta} exceeds safe ceiling 0.1 for vocab-pruned Gemma 4")

    base_dir = resolve_model_dir(cfg.sft_merged_dir)
    before_hashes = hash_multimodal_safetensors(base_dir)

    rows = load_orpo_jsonl(cfg.orpo_jsonl)
    ds = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )

    tok, model = load_gemma4_for_training(base_dir, use_4bit=cfg.use_4bit)
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    probe_cfg = SFTConfig(train_embeddings=False)
    resolved = resolve_target_and_embed_modules([n for n, _ in model.named_modules()], probe_cfg)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=resolved["target_modules"],
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_cfg)
    freeze_stats = freeze_multimodal_parameters(model)
    frozen_embed_tensors = freeze_embeddings_and_ple(model)

    out_dir = Path(cfg.output_dir)
    merged_dir = Path(cfg.merged_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_bnb = importlib.util.find_spec("bitsandbytes") is not None
    trl_kwargs = build_orpo_config_kwargs(
        TRLORPOConfig,
        cfg,
        out_dir=out_dir,
        use_8bit_adam=(not cfg.use_4bit) and has_bnb,
    )
    trl_args = TRLORPOConfig(**trl_kwargs)

    trainer_tok_kw = (
        {"processing_class": tok}
        if "processing_class" in inspect.signature(ORPOTrainer.__init__).parameters
        else {"tokenizer": tok}
    )
    trainer = ORPOTrainer(
        model=model,
        args=trl_args,
        train_dataset=ds,
        **trainer_tok_kw,
    )
    train_result = trainer.train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mm_check = merge_lora_to_bf16(base_dir, out_dir, merged_dir, before_hashes)
    summary = {
        "stage": "stage2b_orpo",
        "config": asdict(cfg),
        "orpo_pairs": len(rows),
        "resolved_target_count": len(resolved["target_modules"]),
        "frozen_embed_tensors": frozen_embed_tensors,
        "freeze_stats": freeze_stats,
        "train_loss": float(getattr(train_result, "training_loss", 0.0)),
        "mm_integrity": mm_check,
        "merged_dir": str(merged_dir),
    }
    (merged_dir / "stage2b_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2B ORPO alignment trainer for Gemma 4 E4B")
    p.add_argument("--sft-merged-dir", default=ORPOConfig.sft_merged_dir)
    p.add_argument("--orpo-jsonl", default=ORPOConfig.orpo_jsonl)
    p.add_argument("--output-dir", default=ORPOConfig.output_dir)
    p.add_argument("--merged-dir", default=ORPOConfig.merged_dir)
    p.add_argument("--max-length", type=int, default=ORPOConfig.max_length)
    p.add_argument("--max-prompt-length", type=int, default=ORPOConfig.max_prompt_length)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--beta", type=float, default=ORPOConfig.beta)
    p.add_argument("--lr", type=float, default=ORPOConfig.learning_rate)
    p.add_argument("--epochs", type=float, default=ORPOConfig.num_train_epochs)
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    cfg = ORPOConfig(
        sft_merged_dir=ns.sft_merged_dir,
        orpo_jsonl=ns.orpo_jsonl,
        output_dir=ns.output_dir,
        merged_dir=ns.merged_dir,
        max_length=ns.max_length,
        max_prompt_length=ns.max_prompt_length,
        use_4bit=not ns.no_4bit,
        beta=ns.beta,
        learning_rate=ns.lr,
        num_train_epochs=ns.epochs,
    )
    summary = run_orpo_training(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

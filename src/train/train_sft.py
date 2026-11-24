"""Stage 2A QLoRA/BF16 SFT trainer for pruned Gemma 4 E4B (Vision/Audio frozen)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
from typing import Any

from .dataset_loader import (
    load_sft_jsonl,
    make_sft_collator,
    split_train_eval,
    tokenize_sft_records,
)
from .mm_integrity import (
    copy_non_weight_assets,
    freeze_multimodal_parameters,
    hash_multimodal_safetensors,
    is_multimodal_tensor_name,
    resolve_model_dir,
    verify_multimodal_integrity,
)


@dataclass
class SFTConfig:
    """Hyperparameters and paths for Stage 2A SFT on Colab Pro (L4 24GB / A100 40GB)."""

    base_model_id: str = "haku-nekooo/gemma4-e4b-nuclear-pruned"
    train_jsonl: str = "data/distill/sft_gemma4.jsonl"
    output_dir: str = "outputs/stage2a_sft_lora"
    merged_dir: str = "outputs/stage2a_sft_merged"
    max_seq_length: int = 8192
    use_4bit: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    learning_rate: float = 1.5e-4
    embedding_lr: float = 1.5e-5
    weight_decay: float = 0.01
    train_embeddings: bool = True
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    embedding_module_names: tuple[str, ...] = (
        "embed_tokens",
        "embed_tokens_per_layer",
        "lm_head",
    )
    num_train_epochs: float = 2.5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.05
    eval_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 50
    seed: int = 42


def build_warmup_kwargs(cls: Any, warmup_ratio: float) -> dict[str, float]:
    """Return warmup_ratio on Transformers 4.x or warmup_steps fraction on Transformers 5.x."""
    params = inspect.signature(cls.__init__).parameters
    if "warmup_ratio" in params:
        return {"warmup_ratio": warmup_ratio}
    return {"warmup_steps": warmup_ratio}


def resolve_target_and_embed_modules(
    module_names: list[str] | tuple[str, ...],
    cfg: SFTConfig,
) -> dict[str, list[str]]:
    """Select exact non-multimodal language-model LoRA targets and embed/PLE modules."""
    target_set = set(cfg.target_modules)
    embed_set = set(cfg.embedding_module_names)
    target_paths: list[str] = []
    embed_paths: list[str] = []

    for full_name in module_names:
        if not full_name or is_multimodal_tensor_name(full_name):
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf in target_set:
            target_paths.append(full_name)
        elif cfg.train_embeddings and leaf in embed_set:
            embed_paths.append(full_name)

    if not target_paths:
        raise ValueError("No non-multimodal LoRA target modules matched model architecture")

    return {
        "target_modules": sorted(set(target_paths)),
        "modules_to_save": sorted(set(embed_paths)),
    }


def build_param_groups(
    named_params: Any,
    cfg: SFTConfig,
) -> list[dict[str, Any]]:
    """Separate LoRA weights (learning_rate) from embed/PLE/lm_head (low embedding_lr)."""
    embed_keys = set(cfg.embedding_module_names)
    lora_decay: list[Any] = []
    lora_nodecay: list[Any] = []
    embed_params: list[Any] = []

    for name, param in named_params:
        if not getattr(param, "requires_grad", False):
            continue
        is_embed = any(k in name for k in embed_keys)
        if is_embed:
            embed_params.append(param)
        elif name.endswith(".bias") or "norm" in name.lower():
            lora_nodecay.append(param)
        else:
            lora_decay.append(param)

    groups: list[dict[str, Any]] = []
    if lora_decay:
        groups.append({"params": lora_decay, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
    if lora_nodecay:
        groups.append({"params": lora_nodecay, "lr": cfg.learning_rate, "weight_decay": 0.0})
    if embed_params:
        groups.append({"params": embed_params, "lr": cfg.embedding_lr, "weight_decay": 0.0})
    return groups


def load_gemma4_for_training(model_dir: Path, *, use_4bit: bool = True) -> tuple[Any, Any]:
    """Load pruned Gemma 4 tokenizer + BF16/4-bit NF4 base model with gradient checkpointing."""
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    try:
        from transformers import Gemma4ForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelCls

    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0} if torch.cuda.is_available() else "cpu",
    }
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = ModelCls.from_pretrained(str(model_dir), **load_kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if not use_4bit and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return tok, model


def merge_lora_to_bf16(
    base_dir: Path,
    adapter_dir: Path,
    merged_dir: Path,
    before_hashes: dict[str, str],
) -> dict[str, Any]:
    """Merge trained LoRA adapter cleanly onto BF16 base weights and verify Vision/Audio SHA-256."""
    import gc
    import torch
    from peft import PeftModel

    try:
        from transformers import Gemma4ForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelCls

    base_model = ModelCls.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    # Keep config tie_word_embeddings consistent if embed_tokens & lm_head diverged.
    cfg = getattr(merged, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    lm_head = getattr(merged, "lm_head", None)
    get_emb = getattr(merged, "get_input_embeddings", None)
    emb = get_emb() if callable(get_emb) else None
    if (
        lm_head is not None
        and emb is not None
        and getattr(text_cfg, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", False))
        and not torch.equal(lm_head.weight, emb.weight)
    ):
        text_cfg.tie_word_embeddings = False
        if cfg is not None and hasattr(cfg, "tie_word_embeddings"):
            cfg.tie_word_embeddings = False

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    copy_non_weight_assets(base_dir, merged_dir)

    del merged, peft_model, base_model
    gc.collect()

    after_hashes = hash_multimodal_safetensors(merged_dir)
    return verify_multimodal_integrity(before_hashes, after_hashes)


def run_sft_training(cfg: SFTConfig) -> dict[str, Any]:
    """Execute Stage 2A SFT (assistant-only loss, low-LR embeddings/PLE) and export merged BF16."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    import torch
    from transformers import Trainer, TrainingArguments

    base_dir = resolve_model_dir(cfg.base_model_id)
    before_hashes = hash_multimodal_safetensors(base_dir)

    records = load_sft_jsonl(cfg.train_jsonl)
    train_raw, eval_raw = split_train_eval(records, eval_ratio=cfg.eval_ratio, seed=cfg.seed)

    tok, model = load_gemma4_for_training(base_dir, use_4bit=cfg.use_4bit)
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    module_names = [name for name, _ in model.named_modules()]
    resolved = resolve_target_and_embed_modules(module_names, cfg)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=resolved["target_modules"],
        modules_to_save=resolved["modules_to_save"] or None,
    )
    model = get_peft_model(model, lora_cfg)
    freeze_stats = freeze_multimodal_parameters(model)

    train_ds = tokenize_sft_records(train_raw, tok, max_seq_length=cfg.max_seq_length)
    eval_ds = tokenize_sft_records(eval_raw, tok, max_seq_length=cfg.max_seq_length) if eval_raw else None

    out_dir = Path(cfg.output_dir)
    merged_dir = Path(cfg.merged_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.save_steps if eval_ds else None,
        save_total_limit=2,
        load_best_model_at_end=bool(eval_ds),
        report_to="none",
        remove_unused_columns=False,
        seed=cfg.seed,
        **build_warmup_kwargs(TrainingArguments, cfg.warmup_ratio),
    )

    param_groups = build_param_groups(model.named_parameters(), cfg)
    try:
        from bitsandbytes.optim import AdamW8bit

        optimizer = AdamW8bit(param_groups, betas=(0.9, 0.999), eps=1e-8)
    except ImportError:
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_sft_collator(tok.pad_token_id),
        optimizers=(optimizer, None),
    )
    train_result = trainer.train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    del trainer, optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mm_check = merge_lora_to_bf16(base_dir, out_dir, merged_dir, before_hashes)
    summary = {
        "stage": "stage2a_sft",
        "config": asdict(cfg),
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds) if eval_ds else 0,
        "resolved_target_count": len(resolved["target_modules"]),
        "resolved_embed_modules": resolved["modules_to_save"],
        "freeze_stats": freeze_stats,
        "train_loss": float(getattr(train_result, "training_loss", 0.0)),
        "mm_integrity": mm_check,
        "merged_dir": str(merged_dir),
    }
    (merged_dir / "stage2a_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2A QLoRA/BF16 SFT trainer for Gemma 4 E4B")
    p.add_argument("--base-model", default=SFTConfig.base_model_id)
    p.add_argument("--train-jsonl", default=SFTConfig.train_jsonl)
    p.add_argument("--output-dir", default=SFTConfig.output_dir)
    p.add_argument("--merged-dir", default=SFTConfig.merged_dir)
    p.add_argument("--max-seq-length", type=int, default=SFTConfig.max_seq_length)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--no-train-embeddings", action="store_true")
    p.add_argument("--epochs", type=float, default=SFTConfig.num_train_epochs)
    p.add_argument("--lr", type=float, default=SFTConfig.learning_rate)
    p.add_argument("--embed-lr", type=float, default=SFTConfig.embedding_lr)
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    cfg = SFTConfig(
        base_model_id=ns.base_model,
        train_jsonl=ns.train_jsonl,
        output_dir=ns.output_dir,
        merged_dir=ns.merged_dir,
        max_seq_length=ns.max_seq_length,
        use_4bit=not ns.no_4bit,
        train_embeddings=not ns.no_train_embeddings,
        num_train_epochs=ns.epochs,
        learning_rate=ns.lr,
        embedding_lr=ns.embed_lr,
    )
    summary = run_sft_training(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

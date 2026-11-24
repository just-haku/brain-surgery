"""Step 2 training, Vision/Audio SHA-256 integrity, and modular GGUF export for Gemma 4 E4B."""

from .check_token_lengths import (
    Gemma4BpeEncoder,
    measure_dataset_token_lengths,
    summarize_lengths,
)
from .dataset_loader import (
    build_assistant_loss_labels,
    load_jsonl,
    load_orpo_jsonl,
    load_sft_jsonl,
    split_train_eval,
)
from .export_gguf import (
    DEFAULT_MMPROJ_OUTTYPES,
    DEFAULT_PLE_CPU_OFFLOAD,
    DEFAULT_QUANTS,
    FAT32_MAX_BYTES,
    build_export_commands,
    export_modular_gguf,
    write_imatrix_calib_corpus,
)
from .mm_integrity import (
    MULTIMODAL_KEYWORDS,
    freeze_multimodal_parameters,
    hash_multimodal_safetensors,
    is_multimodal_tensor_name,
    verify_multimodal_integrity,
)
from .train_orpo import ORPOConfig, freeze_embeddings_and_ple, run_orpo_training
from .train_sft import (
    SFTConfig,
    build_param_groups,
    resolve_target_and_embed_modules,
    run_sft_training,
)

__all__ = [
    "DEFAULT_MMPROJ_OUTTYPES",
    "DEFAULT_PLE_CPU_OFFLOAD",
    "DEFAULT_QUANTS",
    "FAT32_MAX_BYTES",
    "Gemma4BpeEncoder",
    "MULTIMODAL_KEYWORDS",
    "ORPOConfig",
    "SFTConfig",
    "build_assistant_loss_labels",
    "build_export_commands",
    "build_param_groups",
    "export_modular_gguf",
    "freeze_embeddings_and_ple",
    "freeze_multimodal_parameters",
    "hash_multimodal_safetensors",
    "is_multimodal_tensor_name",
    "load_jsonl",
    "load_orpo_jsonl",
    "load_sft_jsonl",
    "measure_dataset_token_lengths",
    "resolve_target_and_embed_modules",
    "run_orpo_training",
    "run_sft_training",
    "split_train_eval",
    "summarize_lengths",
    "verify_multimodal_integrity",
    "write_imatrix_calib_corpus",
]

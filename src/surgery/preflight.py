"""Preflight inspection of Gemma 4 checkpoint metadata before downloading weights."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .bpe_closure import parse_merges
from .vocab_filter import is_unused_token

REQUIRED_PROCESSOR_FILES: tuple[str, ...] = (
    "processor_config.json",
)
OPTIONAL_PROCESSOR_FILES: tuple[str, ...] = (
    "preprocessor_config.json",
    "image_processor_config.json",
    "feature_extractor_config.json",
    "audio_processor_config.json",
)
DEFAULT_PROCESSOR_FALLBACK_REPOS: tuple[str, ...] = (
    "google/gemma-4-E2B-it",
    "Austriani/Gemma-4-E2B-it-ARA-Abliterated",
    "unsloth/gemma-4-E2B-it",
)

CRITICAL_SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<mask>",
    "[multimodal]",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    "<|think|>",
    "<|channel>",
    "<channel|>",
    "<|turn>",
    "<turn|>",
    "<|image>",
    "<|audio>",
    "<|image|>",
    "<|audio|>",
    "<image|>",
    "<audio|>",
    "<|video|>",
)

_TOKEN_KEY_RE = re.compile(r"(?:^vocab_size(?:_per_layer_input)?$|_token_ids?$|_token_index$)")


def ensure_processor_configs(
    local_dir: str | Path,
    fallback_repos: tuple[str, ...] = DEFAULT_PROCESSOR_FALLBACK_REPOS,
    token: str | None = None,
    download_fn: Callable[[str, str, Path, str | None], Any] | None = None,
) -> dict[str, Any]:
    """Backfill missing multimodal processor configs from official/mirror Gemma 4 repos."""
    out = Path(local_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing_req = [f for f in REQUIRED_PROCESSOR_FILES if not (out / f).exists()]
    if not missing_req:
        return {"backfilled": [], "sources": {}, "missing_required": []}

    if download_fn is None:
        from huggingface_hub import hf_hub_download

        def _hf_dl(repo_id: str, filename: str, target_dir: Path, tok: str | None) -> Any:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                token=tok,
            )

        download_fn = _hf_dl

    candidates = (*REQUIRED_PROCESSOR_FILES, *OPTIONAL_PROCESSOR_FILES)
    backfilled: list[str] = []
    sources: dict[str, str] = {}
    errors: dict[str, str] = {}

    for repo_id in fallback_repos:
        still_missing = [f for f in candidates if not (out / f).exists()]
        if not still_missing or all((out / f).exists() for f in REQUIRED_PROCESSOR_FILES):
            break
        for fname in still_missing:
            try:
                download_fn(repo_id, fname, out, token)
                if (out / fname).exists():
                    backfilled.append(fname)
                    sources[fname] = repo_id
            except Exception as exc:  # noqa: BLE001 - gated repo (401/403) or missing optional file (404)
                errors[f"{repo_id}/{fname}"] = str(exc)
        if all((out / f).exists() for f in REQUIRED_PROCESSOR_FILES):
            break

    final_missing = [f for f in REQUIRED_PROCESSOR_FILES if not (out / f).exists()]
    if backfilled:
        print(f"Backfilled missing processor config(s): {sources}")
    if final_missing:
        print(
            f"WARNING: Required processor config(s) still missing after fallbacks: {final_missing}. "
            f"Errors sample: {list(errors.items())[:3]}"
        )
    return {
        "backfilled": backfilled,
        "sources": sources,
        "missing_required": final_missing,
    }


def download_metadata_only(
    repo_id: str,
    local_dir: str | Path,
    token: str | None = None,
    fallback_repos: tuple[str, ...] = DEFAULT_PROCESSOR_FALLBACK_REPOS,
) -> Path:
    """Download config, tokenizer, and index metadata (~30 MB), backfilling processor configs."""
    from huggingface_hub import snapshot_download

    out = Path(local_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out),
        token=token,
        allow_patterns=["*.json", "*.jinja", "tokenizer.model", "*.txt"],
    )
    ensure_processor_configs(out, fallback_repos=fallback_repos, token=token)
    return out


def find_token_id_fields(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively collect vocab_size and *_token_id(s)/_token_index fields."""
    found: dict[str, Any] = {}
    if not isinstance(obj, dict):
        return found
    for key, val in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            found.update(find_token_id_fields(val, path))
        elif _TOKEN_KEY_RE.search(key):
            found[path] = val
    return found


def _load_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_vocab_lookup(tokenizer_json: dict[str, Any]) -> dict[str, int]:
    model = tokenizer_json.get("model") or {}
    vocab: dict[str, int] = dict(model.get("vocab") or {})
    for item in tokenizer_json.get("added_tokens") or []:
        tok = item.get("content")
        tid = item.get("id")
        if isinstance(tok, str) and isinstance(tid, int):
            vocab.setdefault(tok, tid)
    return vocab


def inspect_checkpoint_metadata(
    model_dir: str | Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Inspect config.json, tokenizer files, special tokens, and shard layout."""
    root = Path(model_dir)
    config = _load_json_optional(root / "config.json")
    if config is None:
        raise FileNotFoundError(f"Missing config.json in {root}")

    tok_path = root / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {root}")
    tokenizer_json = json.loads(tok_path.read_text(encoding="utf-8"))

    model_block = tokenizer_json.get("model") or {}
    tok_type = model_block.get("type")
    if tok_type != "BPE":
        raise ValueError(f"Expected BPE tokenizer.json, found model.type={tok_type!r}")

    vocab_map = _build_vocab_lookup(tokenizer_json)
    raw_merges = model_block.get("merges") or []
    merge_pairs, merges_is_list = parse_merges(raw_merges)

    tok_cfg = _load_json_optional(root / "tokenizer_config.json")
    gen_cfg = _load_json_optional(root / "generation_config.json")
    index_json = _load_json_optional(root / "model.safetensors.index.json")

    tok_model_path = root / "tokenizer.model"
    tok_model_bytes = tok_model_path.stat().st_size if tok_model_path.exists() else None

    config_fields = find_token_id_fields(config)
    gen_fields = find_token_id_fields(gen_cfg) if gen_cfg else {}

    special_ids: dict[str, int] = {}
    missing_critical: list[str] = []
    for tok in CRITICAL_SPECIAL_TOKENS:
        if tok in vocab_map:
            special_ids[tok] = vocab_map[tok]
        else:
            missing_critical.append(tok)

    # Check both core thinking/multimodal tokens rigorously; fail fast if absent.
    must_have = {"<pad>", "<eos>", "<bos>", "<unk>", "[multimodal]", "<|think|>", "<|image|>", "<|audio|>"}
    missing_must = [t for t in missing_critical if t in must_have]
    if missing_must:
        raise ValueError(f"Critical Gemma 4 tokens missing from tokenizer.json: {missing_must}")

    byte_count = sum(1 for b in range(256) if f"<0x{b:02X}>" in vocab_map)
    unused_count = sum(1 for tok in vocab_map if is_unused_token(tok))

    weight_map = (index_json or {}).get("weight_map") or {}
    vocab_shards: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        if "embed_tokens" in tensor_name or "lm_head" in tensor_name:
            vocab_shards.setdefault(shard_name, []).append(tensor_name)

    processor_files_present = [
        f for f in (*REQUIRED_PROCESSOR_FILES, *OPTIONAL_PROCESSOR_FILES) if (root / f).exists()
    ]
    missing_processor_files = [f for f in REQUIRED_PROCESSOR_FILES if not (root / f).exists()]

    summary = {
        "model_dir": str(root),
        "config_fields": config_fields,
        "generation_fields": gen_fields,
        "tie_word_embeddings": config.get(
            "tie_word_embeddings",
            (config.get("text_config") or {}).get("tie_word_embeddings"),
        ),
        "has_vision_config": "vision_config" in config,
        "has_audio_config": "audio_config" in config,
        "processor_files_present": processor_files_present,
        "missing_processor_files": missing_processor_files,
        "tokenizer_model_present": tok_model_bytes is not None,
        "tokenizer_model_bytes": tok_model_bytes,
        "tokenizer_type": tok_type,
        "vocab_count": len(vocab_map),
        "merge_count": len(merge_pairs),
        "merges_is_list_format": merges_is_list,
        "byte_fallback_count": byte_count,
        "unused_count": unused_count,
        "special_ids": special_ids,
        "missing_optional_specials": [t for t in missing_critical if t not in must_have],
        "has_tokenizer_config": tok_cfg is not None,
        "vocab_shards": vocab_shards,
        "total_shards": len(set(weight_map.values())) if weight_map else (1 if (root / "model.safetensors").exists() else 0),
    }

    if verbose:
        print("=== PREFLIGHT CHECKPOINT INSPECTION ===")
        print(f"Dir: {summary['model_dir']}")
        print(f"tie_word_embeddings={summary['tie_word_embeddings']} | vision={summary['has_vision_config']} | audio={summary['has_audio_config']}")
        print(
            f"Processor files present: {processor_files_present}"
            + (f" | MISSING REQUIRED: {missing_processor_files}" if missing_processor_files else "")
        )
        print("Config vocab/token_id fields (root + nested):")
        for k, v in sorted(config_fields.items()):
            print(f"  {k}: {v}")
        if gen_fields:
            print("Generation config token_id fields:")
            for k, v in sorted(gen_fields.items()):
                print(f"  {k}: {v}")
        print(
            f"tokenizer.model present: {summary['tokenizer_model_present']}"
            + (f" ({tok_model_bytes} bytes — will remove in pruned output)" if tok_model_bytes else "")
        )
        print(
            f"tokenizer.json: type={tok_type}, vocab={summary['vocab_count']}, "
            f"merges={summary['merge_count']} (list_format={merges_is_list}), "
            f"bytes={byte_count}/256, <unused>={unused_count}"
        )
        print("Critical special token IDs:")
        for tok, tid in special_ids.items():
            print(f"  {tid:>6d}: {tok!r}")
        if summary["missing_optional_specials"]:
            print(f"Note — optional special tokens not in vocab: {summary['missing_optional_specials']}")
        print(f"Total weight shards: {summary['total_shards']}")
        for shard_name, tnames in sorted(vocab_shards.items()):
            print(f"  {shard_name}: {tnames}")

    return summary

"""Vision/Audio parameter freeze, SHA-256 safetensors verification, and asset helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
from typing import Any

from surgery.config_remap import AUX_METADATA_FILES

MULTIMODAL_KEYWORDS: tuple[str, ...] = (
    "vision",
    "audio",
    "multi_modal_projector",
    "embed_vision",
    "embed_audio",
)


def is_multimodal_tensor_name(name: str) -> bool:
    """Return True when a module/tensor name belongs to Vision/Audio encoders or projectors."""
    lowered = name.lower()
    return any(kw in lowered for kw in MULTIMODAL_KEYWORDS)


def freeze_multimodal_parameters(model: Any) -> dict[str, int]:
    """Set requires_grad=False on all Vision/Audio/projector params and return counts."""
    frozen_tensors = 0
    frozen_params = 0
    trainable_tensors = 0
    for name, param in model.named_parameters():
        if is_multimodal_tensor_name(name):
            param.requires_grad = False
            frozen_tensors += 1
            frozen_params += int(param.numel())
        elif getattr(param, "requires_grad", False):
            trainable_tensors += 1
    return {
        "frozen_tensors": frozen_tensors,
        "frozen_params": frozen_params,
        "trainable_tensors": trainable_tensors,
    }


def _iter_safetensors_shards(root: Path) -> list[Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index_data["weight_map"].values()))
        return [root / s for s in shard_names]
    single = root / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(root.glob("*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"No .safetensors weights found in {root}")


def hash_multimodal_safetensors(model_dir: str | Path, chunk_bytes: int = 1 << 20) -> dict[str, str]:
    """Stream raw bytes of every Vision/Audio tensor from safetensors into SHA-256 digests."""
    root = Path(model_dir)
    digests: dict[str, str] = {}
    for shard in _iter_safetensors_shards(root):
        with shard.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
            data_base = 8 + header_len
            for name, meta in sorted(header.items()):
                if name == "__metadata__" or not is_multimodal_tensor_name(name):
                    continue
                start, end = meta["data_offsets"]
                f.seek(data_base + start)
                remaining = end - start
                h = hashlib.sha256()
                h.update(f"{meta.get('dtype')}:{meta.get('shape')}:".encode("utf-8"))
                while remaining > 0:
                    buf = f.read(min(chunk_bytes, remaining))
                    if not buf:
                        raise EOFError(f"Unexpected EOF hashing {name} in {shard}")
                    h.update(buf)
                    remaining -= len(buf)
                digests[name] = h.hexdigest()
    if not digests:
        raise ValueError(f"No multimodal tensors found to hash in {root}")
    return digests


def verify_multimodal_integrity(
    before_hashes: dict[str, str],
    after_hashes: dict[str, str],
) -> dict[str, Any]:
    """Assert 100% key and SHA-256 match for Vision/Audio tensors before vs after training."""
    if not before_hashes:
        raise AssertionError("before_hashes is empty; multimodal baseline was not captured")
    missing = sorted(set(before_hashes) - set(after_hashes))
    extra = sorted(set(after_hashes) - set(before_hashes))
    if missing or extra:
        raise AssertionError(
            f"Multimodal tensor set drifted: missing={missing[:5]}, extra={extra[:5]}"
        )
    mismatched = sorted(k for k in before_hashes if before_hashes[k] != after_hashes[k])
    if mismatched:
        raise AssertionError(
            f"Multimodal SHA-256 mismatch on {len(mismatched)} tensors (e.g. {mismatched[:5]})"
        )
    return {
        "verified": True,
        "tensor_count": len(before_hashes),
        "sample_key": next(iter(sorted(before_hashes))),
    }


def resolve_model_dir(model_id_or_dir: str | Path) -> Path:
    """Return local checkpoint Path, downloading via huggingface_hub if a repo ID is given."""
    candidate = Path(model_id_or_dir)
    if candidate.exists():
        return candidate
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(model_id_or_dir)))


def copy_non_weight_assets(src_dir: str | Path, dst_dir: str | Path) -> list[str]:
    """Copy missing processor/chat_template/tokenizer metadata files into merged output dir."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    candidates = tuple(AUX_METADATA_FILES) + ("tokenizer.json", "tokenizer_config.json", "generation_config.json")
    for fname in candidates:
        s_file = src / fname
        d_file = dst / fname
        if s_file.exists() and not d_file.exists():
            shutil.copy2(s_file, d_file)
            copied.append(fname)
    return copied


def assert_tied_embeddings_intact(model: Any) -> None:
    """Re-tie word embeddings before save when config enables tie_word_embeddings."""
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    if getattr(text_cfg, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", False)):
        if hasattr(model, "tie_weights"):
            model.tie_weights()

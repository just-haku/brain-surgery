"""Low-RAM shard-by-shard safetensors slicing (< 4-5 GB peak CPU RAM)."""

from __future__ import annotations

from bisect import bisect_left
import gc
import json
from pathlib import Path
import shutil
import struct
from typing import Any


def get_rss_gb() -> float:
    """Return current process resident set size in GiB (Linux/Colab), else 0.0."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return 0.0
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return round(kb / (1024 * 1024), 3)
    except OSError:
        return 0.0
    return 0.0


def _read_safetensors_tensor_bytes(shard_path: Path) -> int:
    """Read only the JSON header of a .safetensors file to sum tensor byte sizes."""
    with shard_path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    total = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        total += end - start
    return total


def _slice_tensor_chunked(
    safe_slice: Any,
    kept_old_ids: list[int],
    orig_vocab_size: int,
    torch_mod: Any,
    chunk_rows: int = 32768,
) -> Any:
    """Slice dim-0 of a large vocab tensor in chunks so full matrix is never materialized."""
    shape = list(safe_slice.get_shape())
    first_probe = safe_slice[0:1]
    dtype = first_probe.dtype
    del first_probe

    out_shape = [len(kept_old_ids), *shape[1:]]
    out = torch_mod.empty(out_shape, dtype=dtype, device="cpu")

    for start in range(0, orig_vocab_size, chunk_rows):
        end = min(start + chunk_rows, orig_vocab_size)
        lo = bisect_left(kept_old_ids, start)
        hi = bisect_left(kept_old_ids, end)
        if lo == hi:
            continue
        chunk = safe_slice[start:end]
        local_idx = torch_mod.tensor(
            [oid - start for oid in kept_old_ids[lo:hi]],
            dtype=torch_mod.long,
        )
        out[lo:hi] = chunk.index_select(0, local_idx)
        del chunk, local_idx
    return out


def slice_checkpoint_shards(
    src_dir: str | Path,
    dst_dir: str | Path,
    kept_old_ids: list[int],
    orig_vocab_size: int = 262144,
    chunk_rows: int = 32768,
) -> dict[str, Any]:
    """Slice vocab-indexed tensors shard-by-shard using safe_open + save_file."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index_data["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    elif (src / "model.safetensors").exists():
        index_data = None
        shard_names = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No model.safetensors.index.json or model.safetensors in {src}")

    sliced_tensors: list[dict[str, Any]] = []
    copied_shards: list[str] = []
    peak_rss_gb = get_rss_gb()
    total_bytes_before = 0
    total_bytes_after = 0

    for shard_idx, shard_name in enumerate(shard_names, start=1):
        src_shard = src / shard_name
        dst_shard = dst / shard_name
        total_bytes_before += _read_safetensors_tensor_bytes(src_shard)

        with safe_open(str(src_shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            metadata = f.metadata()
            vocab_keys: list[str] = []
            for key in keys:
                shape = f.get_slice(key).get_shape()
                if orig_vocab_size in shape:
                    if len(shape) == 0 or shape[0] != orig_vocab_size:
                        raise ValueError(
                            f"Tensor {key} has vocab dim {orig_vocab_size} on non-zero axis: shape={shape}"
                        )
                    vocab_keys.append(key)

            if not vocab_keys:
                shutil.copy2(src_shard, dst_shard)
                copied_shards.append(shard_name)
                total_bytes_after += _read_safetensors_tensor_bytes(dst_shard)
                rss = get_rss_gb()
                peak_rss_gb = max(peak_rss_gb, rss)
                print(
                    f"[{shard_idx}/{len(shard_names)}] {shard_name}: zero vocab tensors -> copied directly (RSS={rss:.2f} GB)"
                )
                continue

            out_tensors: dict[str, Any] = {}
            vocab_key_set = set(vocab_keys)
            for key in keys:
                if key in vocab_key_set:
                    s = f.get_slice(key)
                    old_shape = list(s.get_shape())
                    sliced = _slice_tensor_chunked(
                        s, kept_old_ids, orig_vocab_size, torch, chunk_rows=chunk_rows
                    )
                    out_tensors[key] = sliced.contiguous()
                    sliced_tensors.append(
                        {
                            "name": key,
                            "shard": shard_name,
                            "old_shape": old_shape,
                            "new_shape": list(out_tensors[key].shape),
                            "dtype": str(out_tensors[key].dtype),
                        }
                    )
                else:
                    out_tensors[key] = f.get_tensor(key)
                rss = get_rss_gb()
                peak_rss_gb = max(peak_rss_gb, rss)

        save_file(out_tensors, str(dst_shard), metadata=metadata)
        del out_tensors
        gc.collect()

        total_bytes_after += _read_safetensors_tensor_bytes(dst_shard)
        rss = get_rss_gb()
        peak_rss_gb = max(peak_rss_gb, rss)
        print(
            f"[{shard_idx}/{len(shard_names)}] {shard_name}: sliced {len(vocab_keys)} tensor(s) -> saved (RSS={rss:.2f} GB)"
        )

    if not sliced_tensors:
        raise RuntimeError(
            f"No tensor with axis-0 size {orig_vocab_size} was found across {shard_names}"
        )

    if index_data is not None:
        new_index = dict(index_data)
        meta = dict(new_index.get("metadata") or {})
        meta["total_size"] = total_bytes_after
        new_index["metadata"] = meta
        (dst / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=2) + "\n", encoding="utf-8"
        )

    has_ple = any("per_layer" in item["name"] for item in sliced_tensors)
    if not has_ple:
        print("WARNING: No 'per_layer' PLE tensor was found in sliced_tensors; verify model architecture!")

    return {
        "shard_count": len(shard_names),
        "copied_untouched_shards": copied_shards,
        "sliced_tensors": sliced_tensors,
        "has_ple_sliced": has_ple,
        "bytes_before": total_bytes_before,
        "bytes_after": total_bytes_after,
        "saved_gb": round((total_bytes_before - total_bytes_after) / (1024**3), 3),
        "peak_rss_gb": peak_rss_gb,
    }

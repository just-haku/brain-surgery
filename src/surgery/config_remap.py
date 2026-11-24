"""Synchronous config & tokenizer remapping plus decisive tokenizer.model removal."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from .vocab_filter import CRITICAL_TOKEN_STRINGS

AUX_METADATA_FILES: tuple[str, ...] = (
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "image_processor_config.json",
    "feature_extractor_config.json",
    "audio_processor_config.json",
    "chat_template.jinja",
)


def build_id_map(kept_old_ids: list[int]) -> dict[int, int]:
    """Map kept original token IDs to contiguous [0 .. new_vocab_size - 1]."""
    return {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}


def remap_config_dict(
    cfg: dict[str, Any],
    id_map: dict[int, int],
    new_vocab_size: int,
    _path: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """Remap vocab_size and *_token_id(s)/_token_index across root and nested configs."""
    out = copy.deepcopy(cfg) if _path == "" else cfg
    changes: list[str] = []

    for key, val in list(out.items()):
        cur_path = f"{_path}.{key}" if _path else key
        if isinstance(val, dict):
            sub_out, sub_changes = remap_config_dict(val, id_map, new_vocab_size, cur_path)
            out[key] = sub_out
            changes.extend(sub_changes)
        elif key in ("vocab_size", "vocab_size_per_layer_input"):
            out[key] = new_vocab_size
            changes.append(f"{cur_path}: {val} -> {new_vocab_size}")
        elif key.endswith(("_token_id", "_token_index", "_token_ids")):
            if val is None:
                continue
            if isinstance(val, int):
                if val not in id_map:
                    raise KeyError(f"Config token ID {cur_path}={val} was pruned and missing from id_map")
                out[key] = id_map[val]
                changes.append(f"{cur_path}: {val} -> {id_map[val]}")
            elif isinstance(val, list):
                new_list = []
                for idx, item in enumerate(val):
                    if isinstance(item, int):
                        if item not in id_map:
                            raise KeyError(
                                f"Config token ID {cur_path}[{idx}]={item} was pruned and missing from id_map"
                            )
                        new_list.append(id_map[item])
                    else:
                        new_list.append(item)
                out[key] = new_list
                changes.append(f"{cur_path}: {val} -> {new_list}")

    return out, changes


def assert_no_oob_token_ids(cfg: dict[str, Any], new_vocab_size: int, _path: str = "") -> None:
    """Assert zero out-of-bounds token IDs remain anywhere in a config dictionary."""
    for key, val in cfg.items():
        cur_path = f"{_path}.{key}" if _path else key
        if isinstance(val, dict):
            assert_no_oob_token_ids(val, new_vocab_size, cur_path)
        elif key in ("vocab_size", "vocab_size_per_layer_input"):
            if val != new_vocab_size:
                raise AssertionError(f"{cur_path}={val} != new_vocab_size={new_vocab_size}")
        elif key.endswith(("_token_id", "_token_index", "_token_ids")):
            if isinstance(val, int) and not (0 <= val < new_vocab_size):
                raise AssertionError(f"OOB {cur_path}={val} for vocab_size={new_vocab_size}")
            if isinstance(val, list):
                for idx, item in enumerate(val):
                    if isinstance(item, int) and not (0 <= item < new_vocab_size):
                        raise AssertionError(
                            f"OOB {cur_path}[{idx}]={item} for vocab_size={new_vocab_size}"
                        )


def _rewrite_tokenizer_json(
    src_tok: dict[str, Any],
    old_vocab_by_id: dict[int, str],
    id_map: dict[int, int],
    pruned_merges_raw: list,
    mandatory_added_tokens: tuple[str, ...] = CRITICAL_TOKEN_STRINGS,
) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(src_tok)
    model = out["model"]
    orig_model_ids = set((model.get("vocab") or {}).values())
    # Insert in new_id order so serialized JSON stays cleanly ordered 0..new_vocab_size-1.
    model["vocab"] = {
        old_vocab_by_id[oid]: nid
        for oid, nid in sorted(id_map.items(), key=lambda kv: kv[1])
        if oid in orig_model_ids
    }
    model["merges"] = pruned_merges_raw

    new_added: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_contents: set[str] = set()
    for entry in out.get("added_tokens") or []:
        oid = entry.get("id")
        if isinstance(oid, int) and oid in id_map:
            updated = dict(entry)
            new_id = id_map[oid]
            updated["id"] = new_id
            new_added.append(updated)
            seen_ids.add(new_id)
            content = updated.get("content")
            if isinstance(content, str):
                seen_contents.add(content)

    # Guarantee mandatory special/multimodal tokens (esp. ID 5 [multimodal] which has token_type=1)
    # are registered in added_tokens so BPE never splits them into sub-tokens.
    new_id_by_tok = {old_vocab_by_id[oid]: nid for oid, nid in id_map.items()}
    injected_tokens: list[str] = []
    for tok_str in mandatory_added_tokens:
        new_id = new_id_by_tok.get(tok_str)
        if new_id is None or new_id in seen_ids or tok_str in seen_contents:
            continue
        new_added.append(
            {
                "id": new_id,
                "content": tok_str,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": tok_str not in ("<bbox>", "</bbox>"),
            }
        )
        seen_ids.add(new_id)
        seen_contents.add(tok_str)
        injected_tokens.append(tok_str)

    new_added.sort(key=lambda item: item["id"])
    out["added_tokens"] = new_added
    return out, injected_tokens


def _rewrite_tokenizer_config(
    src_tok_cfg: dict[str, Any],
    id_map: dict[int, int],
    added_tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    out = copy.deepcopy(src_tok_cfg)
    decoder = out.get("added_tokens_decoder")
    new_decoder: dict[str, Any] = {}
    if isinstance(decoder, dict):
        for old_key, val in decoder.items():
            try:
                oid = int(old_key)
            except ValueError:
                continue
            if oid in id_map:
                new_decoder[str(id_map[oid])] = val

    for entry in added_tokens:
        key = str(entry["id"])
        if key not in new_decoder:
            new_decoder[key] = {
                "content": entry["content"],
                "lstrip": bool(entry.get("lstrip", False)),
                "normalized": bool(entry.get("normalized", False)),
                "rstrip": bool(entry.get("rstrip", False)),
                "single_word": bool(entry.get("single_word", False)),
                "special": bool(entry.get("special", True)),
            }

    out["added_tokens_decoder"] = dict(
        sorted(new_decoder.items(), key=lambda kv: int(kv[0]))
    )
    return out


def remap_and_save_metadata(
    src_dir: str | Path,
    dst_dir: str | Path,
    kept_old_ids: list[int],
    pruned_merges_raw: list,
    id_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Remap config.json (root + text_config), tokenizer files, and remove tokenizer.model."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    if id_map is None:
        id_map = build_id_map(kept_old_ids)
    new_vocab_size = len(kept_old_ids)

    tok_json = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    old_vocab_by_id: dict[int, str] = {
        int(tid): tok for tok, tid in (tok_json.get("model") or {}).get("vocab", {}).items()
    }
    for entry in tok_json.get("added_tokens") or []:
        tid = entry.get("id")
        content = entry.get("content")
        if isinstance(tid, int) and isinstance(content, str):
            old_vocab_by_id.setdefault(tid, content)

    # 1. Remap config.json (root + nested text_config, etc.)
    src_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    new_cfg, cfg_changes = remap_config_dict(src_cfg, id_map, new_vocab_size)
    assert_no_oob_token_ids(new_cfg, new_vocab_size)
    (dst / "config.json").write_text(
        json.dumps(new_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 2. Remap generation_config.json if present.
    gen_changes: list[str] = []
    gen_path = src / "generation_config.json"
    if gen_path.exists():
        src_gen = json.loads(gen_path.read_text(encoding="utf-8"))
        new_gen, gen_changes = remap_config_dict(src_gen, id_map, new_vocab_size)
        assert_no_oob_token_ids(new_gen, new_vocab_size)
        (dst / "generation_config.json").write_text(
            json.dumps(new_gen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # 3. Rewrite tokenizer.json and tokenizer_config.json.
    new_tok_json, injected_tokens = _rewrite_tokenizer_json(
        tok_json, old_vocab_by_id, id_map, pruned_merges_raw
    )
    (dst / "tokenizer.json").write_text(
        json.dumps(new_tok_json, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    tok_cfg_path = src / "tokenizer_config.json"
    if tok_cfg_path.exists() or new_tok_json.get("added_tokens"):
        src_tok_cfg = (
            json.loads(tok_cfg_path.read_text(encoding="utf-8"))
            if tok_cfg_path.exists()
            else {}
        )
        new_tok_cfg = _rewrite_tokenizer_config(
            src_tok_cfg, id_map, new_tok_json.get("added_tokens") or []
        )
        (dst / "tokenizer_config.json").write_text(
            json.dumps(new_tok_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # 4. Copy auxiliary metadata / processor / jinja files (never tokenizer.model).
    copied_aux: list[str] = []
    for fname in AUX_METADATA_FILES:
        fpath = src / fname
        if fpath.exists():
            shutil.copy2(fpath, dst / fname)
            copied_aux.append(fname)
    for jinja_file in src.glob("*.jinja"):
        if jinja_file.name not in copied_aux:
            shutil.copy2(jinja_file, dst / jinja_file.name)
            copied_aux.append(jinja_file.name)

    # 5. Decisively exclude / unlink tokenizer.model so tools read strictly from tokenizer.json.
    src_has_spm = (src / "tokenizer.model").exists()
    dst_spm = dst / "tokenizer.model"
    if dst_spm.exists():
        dst_spm.unlink()

    missing_processor_files = [
        fname
        for fname in ("processor_config.json", "chat_template.jinja")
        if not (dst / fname).exists()
    ]

    return {
        "new_vocab_size": new_vocab_size,
        "config_changes": cfg_changes,
        "generation_changes": gen_changes,
        "injected_added_tokens": injected_tokens,
        "copied_auxiliary_files": copied_aux,
        "missing_processor_files": missing_processor_files,
        "tokenizer_model_present_in_src": src_has_spm,
        "tokenizer_model_action": "removed_use_tokenizer_json_only",
    }

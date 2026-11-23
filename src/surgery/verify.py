"""Verification Gates 1-3 for the pruned Gemma 4 E2B checkpoint."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

GATE1_TEXT_SAMPLES: tuple[str, ...] = (
    "Explain how Python asyncio event loops prevent deadlocks when using asyncio.Lock.",
    "Hãy giải thích cơ chế quản lý bộ nhớ và con trỏ thông minh trong C++ bằng tiếng Việt dễ hiểu.",
    "def quicksort(xs: list[int]) -> list[int]:\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs) // 2]\n    return quicksort([x for x in xs if x < pivot]) + [x for x in xs if x == pivot] + quicksort([x for x in xs if x > pivot])",
    "SELECT id, name FROM projects WHERE status = 'active' AND score >= 95 ORDER BY id DESC;",
    "<div class=\"panel\">\n  <span>OK :) :))) ._. .-. =))</span>\n</div>",
)

GATE1_SPECIAL_STRINGS: tuple[str, ...] = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "[multimodal]",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|think|>",
    "<|channel>",
    "<channel|>",
    "<|turn>",
    "<turn|>",
    "<|image>",
    "<|audio>",
    "<|image|>",
    "<|audio|>",
)

GATE2_DEFAULT_PROBES: tuple[str, ...] = (
    "In Python, why should mutable default arguments be avoided in function definitions?",
    "Giải thích ngắn gọn sự khác biệt giữa stack và heap trong C++:",
    "def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n    out = []\n",
    "SELECT user_id, COUNT(*) FROM events WHERE created_at >= '2026-01-01' GROUP BY",
)


def _load_model_and_processor(model_dir: Path, torch_mod: Any):
    from transformers import AutoProcessor

    try:
        from transformers import Gemma4ForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelCls

    processor = AutoProcessor.from_pretrained(str(model_dir))
    device_map = {"": 0} if torch_mod.cuda.is_available() else "cpu"
    model = ModelCls.from_pretrained(
        str(model_dir),
        torch_dtype=torch_mod.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def run_gate1_tokenizer(
    pruned_dir: str | Path,
    id_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Gate 1: lossless EN/VI/Code round-trip, special token IDs, and emoji ban policy."""
    from transformers import AutoTokenizer

    root = Path(pruned_dir)
    if (root / "tokenizer.model").exists():
        raise AssertionError("Gate 1 failed: stale tokenizer.model still exists in pruned_dir")

    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(root), use_fast=True)

    # 1. Lossless round-trip on EN, VI, Code, indentation, and ASCII emoticons.
    for idx, sample in enumerate(GATE1_TEXT_SAMPLES):
        ids = tok.encode(sample, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=False)
        if decoded != sample:
            raise AssertionError(
                f"Gate 1 round-trip mismatch on sample #{idx}:\nexpected={sample!r}\ngot={decoded!r}"
            )
        if tok.unk_token_id is not None and tok.unk_token_id in ids:
            raise AssertionError(f"Gate 1 failed: <unk> emitted for sample #{idx}")

    # 2. Verify critical special tokens encode to a single ID matching config.json (root & text_config).
    vocab = tok.get_vocab()
    special_check: dict[str, int] = {}
    for sp in GATE1_SPECIAL_STRINGS:
        if sp not in vocab:
            raise AssertionError(f"Gate 1 failed: critical token {sp!r} missing from vocab")
        ids = tok.encode(sp, add_special_tokens=False)
        if len(ids) != 1 or ids[0] != vocab[sp]:
            raise AssertionError(
                f"Gate 1 failed: special token {sp!r} encoded to {ids} instead of [{vocab[sp]}]"
            )
        special_check[sp] = ids[0]

    for key, tok_attr in (("pad_token_id", "pad_token_id"), ("bos_token_id", "bos_token_id"), ("eos_token_id", "eos_token_id")):
        tok_val = getattr(tok, tok_attr)
        for scope_name, scope_cfg in (("root", cfg), ("text_config", cfg.get("text_config") or {})):
            if key in scope_cfg and scope_cfg[key] is not None:
                cfg_val = scope_cfg[key]
                match = tok_val in cfg_val if isinstance(cfg_val, list) else tok_val == cfg_val
                if not match:
                    raise AssertionError(
                        f"Gate 1 mismatch: {scope_name}.{key}={cfg_val} vs tokenizer.{tok_attr}={tok_val}"
                    )

    if id_map is not None:
        if 5 in id_map and special_check["[multimodal]"] != id_map[5]:
            raise AssertionError("Gate 1 failed: [multimodal] ID does not match id_map[5]")

    # 3. Verify Unicode emoji are NOT in vocab and encode safely via byte-fallback without <unk>.
    emoji_samples = ("🔥", "😂", "🚀", "👨‍💻")
    emoji_behavior: dict[str, list[int]] = {}
    for em in emoji_samples:
        if em in vocab:
            raise AssertionError(f"Gate 1 failed: banned Unicode emoji {em!r} found in pruned vocab")
        ids = tok.encode(em, add_special_tokens=False)
        if not ids:
            raise AssertionError(f"Gate 1 failed: empty encoding for emoji {em!r}")
        if tok.unk_token_id is not None and tok.unk_token_id in ids:
            raise AssertionError(f"Gate 1 failed: emoji {em!r} hit <unk> instead of byte-fallback")
        decoded = tok.decode(ids, skip_special_tokens=False)
        if decoded != em:
            raise AssertionError(f"Gate 1 failed: byte-fallback decode mismatch for {em!r}: {decoded!r}")
        emoji_behavior[em] = ids

    print("Gate 1 PASSED: EN/VI/Code/ASCII-emoticon round-trip, special IDs, and emoji byte-fallback OK.")
    return {
        "gate1_passed": True,
        "vocab_size": len(vocab),
        "special_ids": special_check,
        "emoji_byte_fallback_lengths": {em: len(ids) for em, ids in emoji_behavior.items()},
    }


def capture_orig_logits(
    orig_dir: str | Path,
    kept_old_ids: list[int],
    save_path: str | Path,
    probes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Phase 2A: capture original-model next-token logits on kept IDs, then free RAM/VRAM."""
    import torch
    from transformers import AutoTokenizer

    orig_path = Path(orig_dir)
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(orig_path), use_fast=True)
    kept_set = set(kept_old_ids)
    kept_idx_cpu = torch.tensor(kept_old_ids, dtype=torch.long)

    selected_probes: list[str] = []
    orig_input_ids: list[list[int]] = []
    for text in (probes or GATE2_DEFAULT_PROBES):
        ids = tok.encode(text, add_special_tokens=True)
        if all(tid in kept_set for tid in ids):
            selected_probes.append(text)
            orig_input_ids.append(ids)
        else:
            print(f"Skipping probe (contains pruned token): {text[:50]!r}")

    if not selected_probes:
        raise RuntimeError("No Gate 2 probes survived tokenization with kept_old_ids")

    _, model = _load_model_and_processor(orig_path, torch)
    kept_idx_dev = kept_idx_cpu.to(model.device)
    logits_list: list[Any] = []

    with torch.no_grad():
        for ids in orig_input_ids:
            inp = torch.tensor([ids], dtype=torch.long, device=model.device)
            out = model(input_ids=inp)
            last_kept = out.logits[0, -1, :].index_select(0, kept_idx_dev).float().cpu()
            logits_list.append(last_kept)
            del inp, out

    torch.save(
        {
            "probes": selected_probes,
            "orig_input_ids": orig_input_ids,
            "orig_logits_kept": logits_list,
        },
        str(out_path),
    )

    del model, kept_idx_dev
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Phase 2A complete: captured {len(selected_probes)} original logit vectors -> {out_path}")
    return {"captured_probes": len(selected_probes), "save_path": str(out_path)}


def run_gate2_and_gate3(
    pruned_dir: str | Path,
    id_map: dict[int, int],
    orig_logits_path: str | Path,
    max_abs_diff_tol: float = 1e-2,
) -> dict[str, Any]:
    """Phase 2B: Gate 2 logit equivalence vs unpruned + Gate 3 E2E generate & tower check."""
    import torch

    root = Path(pruned_dir)
    payload = torch.load(str(orig_logits_path), map_location="cpu", weights_only=True)
    probes: list[str] = payload["probes"]
    orig_input_ids: list[list[int]] = payload["orig_input_ids"]
    orig_logits_kept: list[Any] = payload["orig_logits_kept"]

    processor, model = _load_model_and_processor(root, torch)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Gate 2: compare pruned IDs + pruned logits against orig_logits_kept.
    max_diff_overall = 0.0
    argmax_matches = 0
    with torch.no_grad():
        for text, orig_ids, orig_vec in zip(probes, orig_input_ids, orig_logits_kept):
            pruned_ids = tok.encode(text, add_special_tokens=True)
            expected_ids = [id_map[oid] for oid in orig_ids]
            if pruned_ids != expected_ids:
                raise AssertionError(
                    f"Gate 2 ID remap mismatch on probe {text[:40]!r}:\nexpected={expected_ids}\ngot={pruned_ids}"
                )
            inp = torch.tensor([pruned_ids], dtype=torch.long, device=model.device)
            out = model(input_ids=inp)
            pruned_vec = out.logits[0, -1, :].float().cpu()
            diff = float((pruned_vec - orig_vec).abs().max().item())
            max_diff_overall = max(max_diff_overall, diff)
            if int(pruned_vec.argmax().item()) == int(orig_vec.argmax().item()):
                argmax_matches += 1
            del inp, out

    if max_diff_overall > max_abs_diff_tol:
        raise AssertionError(
            f"Gate 2 failed: max_abs_diff={max_diff_overall:.6f} > tol={max_abs_diff_tol}"
        )
    if argmax_matches != len(probes):
        raise AssertionError(
            f"Gate 2 failed: argmax matched {argmax_matches}/{len(probes)} probes"
        )
    print(f"Gate 2 PASSED: max_abs_diff={max_diff_overall:.6f}, argmax={argmax_matches}/{len(probes)}")

    # Gate 3: E2E bilingual Code generation with native thinking + frozen multimodal tower check.
    messages = [
        {
            "role": "user",
            "content": "Viết một hàm Python tính Fibonacci dùng dynamic programming và giải thích ngắn gọn bằng tiếng Việt.",
        }
    ]
    if hasattr(tok, "apply_chat_template"):
        try:
            prompt_text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            prompt_text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt_text = messages[0]["content"]

    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
        )
    new_tokens = gen_ids[0][inputs["input_ids"].shape[-1] :]
    gen_text = tok.decode(new_tokens, skip_special_tokens=False)
    if not gen_text.strip():
        raise AssertionError("Gate 3 failed: empty generation output")
    if tok.unk_token and tok.unk_token in gen_text:
        raise AssertionError(f"Gate 3 failed: <unk> generated in output: {gen_text!r}")

    param_names = [name for name, _ in model.named_parameters()]
    has_vision_params = any("vision" in name for name in param_names)
    has_audio_params = any("audio" in name for name in param_names)
    if not (has_vision_params and has_audio_params):
        raise AssertionError(
            f"Gate 3 failed: missing multimodal towers (vision={has_vision_params}, audio={has_audio_params})"
        )

    del model, inputs, gen_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"Gate 3 PASSED: generated {len(new_tokens)} tokens, vision={has_vision_params}, audio={has_audio_params}"
    )
    return {
        "gate2_passed": True,
        "max_abs_diff": round(max_diff_overall, 6),
        "argmax_matches": f"{argmax_matches}/{len(probes)}",
        "gate3_passed": True,
        "generated_preview": gen_text[:300],
        "has_vision_params": has_vision_params,
        "has_audio_params": has_audio_params,
    }

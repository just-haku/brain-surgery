"""Unit tests, notebook validation, and real Gemma 4 E2B vocab smoke test."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surgery.bpe_closure import (  # noqa: E402
    close_and_align_vocab,
    format_merges,
    parse_merges,
    verify_merge_closure,
)
from surgery.config_remap import (  # noqa: E402
    assert_no_oob_token_ids,
    build_id_map,
    remap_and_save_metadata,
    remap_config_dict,
)
from surgery.preflight import ensure_processor_configs, inspect_checkpoint_metadata  # noqa: E402
from surgery.vocab_filter import (  # noqa: E402
    collect_mandatory_whitelist,
    is_ascii_vi_code_candidate,
    is_emoji_or_banned_symbol,
    is_unused_token,
    select_hybrid_vocab,
)

GGUF_BASELINE = ROOT / "brain" / "Gemma-4-E2B-IT-AB.gguf"
NOTEBOOK_PATH = ROOT / "notebooks" / "step0_vocab_pruning.ipynb"


def _load_vocab_and_merges_from_gguf(gguf_path: Path) -> tuple[dict[int, str], list[str]]:
    with gguf_path.open("rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF"
        _ver, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

        def read_str() -> str:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", errors="replace")

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        tokens = None
        merges = None
        for _ in range(n_kv):
            key = read_str()
            val_type = struct.unpack("<I", f.read(4))[0]
            if key == "tokenizer.ggml.tokens":
                _et, ln = struct.unpack("<IQ", f.read(12))
                tokens = [read_str() for _ in range(ln)]
            elif key == "tokenizer.ggml.merges":
                _et, ln = struct.unpack("<IQ", f.read(12))
                merges = [read_str() for _ in range(ln)]
            else:
                if val_type == 8:
                    n = struct.unpack("<Q", f.read(8))[0]
                    f.seek(n, 1)
                elif val_type == 9:
                    et, ln = struct.unpack("<IQ", f.read(12))
                    if et == 8:
                        for _ in range(ln):
                            n = struct.unpack("<Q", f.read(8))[0]
                            f.seek(n, 1)
                    else:
                        f.seek(sizes[et] * ln, 1)
                else:
                    f.seek(sizes[val_type], 1)
            if tokens is not None and merges is not None:
                break
    assert tokens is not None and merges is not None
    return {idx: tok for idx, tok in enumerate(tokens)}, merges


class TestVocabFilter(unittest.TestCase):
    def test_emoji_and_unused_blacklists(self) -> None:
        self.assertTrue(is_unused_token("<unused0>"))
        self.assertTrue(is_unused_token("<unused6226>"))
        self.assertFalse(is_unused_token("<|think|>"))

        for bad in ("🔥", "😂", "🚀", "☀", "✈", "\u200d", "\ufe0f", "a🔥b"):
            self.assertTrue(is_emoji_or_banned_symbol(bad), bad)
            self.assertFalse(is_ascii_vi_code_candidate(bad), bad)

        for good in ("▁tiếng", "Việt", "đường", "▁def", ":)))", "._.", "\n\n", "\t\t"):
            self.assertFalse(is_emoji_or_banned_symbol(good), good)
            self.assertTrue(is_ascii_vi_code_candidate(good), good)

        # Reject non-Vietnamese European diacritics and non-Latin scripts.
        for foreign in ("▁straße", "▁привет", "▁日本語", "▁español", "▁français"):
            self.assertFalse(is_ascii_vi_code_candidate(foreign), foreign)


class TestBpeClosure(unittest.TestCase):
    def test_both_merge_formats_and_ancestor_closure(self) -> None:
        pairs_str, is_list_str = parse_merges(["f u", "fu n", "c a", "a b"])
        self.assertFalse(is_list_str)
        self.assertEqual(format_merges(pairs_str, is_list_str), ["f u", "fu n", "c a", "a b"])

        pairs_list, is_list_arr = parse_merges([["f", "u"], ["fu", "n"], ["▁", "fun"]])
        self.assertTrue(is_list_arr)
        self.assertEqual(format_merges(pairs_list, is_list_arr), [["f", "u"], ["fu", "n"], ["▁", "fun"]])

        vocab_set = {"▁", "f", "u", "n", "fu", "fun", "▁fun", "x", "y", "xy"}
        # Pad synthetic vocab up to 128 single chars so target_size=128 is testable.
        extra = [chr(0x21 + i) for i in range(128)]
        for ch in extra:
            vocab_set.add(ch)

        kept, kept_merges = close_and_align_vocab(
            whitelist_tokens={"▁", "f", "u", "n"},
            primary_tokens=["▁fun"],
            fallback_ranked_tokens=extra,
            merge_pairs=[("f", "u"), ("fu", "n"), ("▁", "fun"), ("x", "y")],
            vocab_set=vocab_set,
            target_size=128,
        )
        self.assertEqual(len(kept), 128)
        self.assertIn("fu", kept)
        self.assertIn("fun", kept)
        self.assertIn("▁fun", kept)
        # Unused merge ('x', 'y') should be dropped while ancestor order is preserved.
        self.assertEqual(kept_merges, [("f", "u"), ("fu", "n"), ("▁", "fun")])
        self.assertEqual(verify_merge_closure(kept, kept_merges, {"fu", "fun", "▁fun", "xy"}), [])


class TestConfigRemapAndPreflight(unittest.TestCase):
    def test_remap_root_and_text_config_and_remove_tokenizer_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            src.mkdir()

            cfg = {
                "vocab_size": 262144,
                "pad_token_id": 0,
                "eos_token_id": [1, 106],
                "bos_token_id": 2,
                "image_token_id": 258880,
                "audio_token_id": 258881,
                "text_config": {
                    "vocab_size": 262144,
                    "vocab_size_per_layer_input": 262144,
                    "pad_token_id": 0,
                    "bos_token_id": 2,
                    "eos_token_id": 1,
                    "boi_token_id": 255999,
                    "eoi_token_id": 258882,
                },
                "vision_config": {"hidden_size": 768},
                "audio_config": {"hidden_size": 512},
            }
            gen_cfg = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0}
            tok_json = {
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "<pad>": 0,
                        "<eos>": 1,
                        "<bos>": 2,
                        "<unk>": 3,
                        "<mask>": 4,
                        "[multimodal]": 5,
                        "<|think|>": 98,
                        "<turn|>": 106,
                        "<unused0>": 200,
                        "<|image>": 255999,
                        "<|image|>": 258880,
                        "<|audio|>": 258881,
                        "<image|>": 258882,
                    },
                    "merges": [],
                },
                "added_tokens": [
                    {"id": 258880, "content": "<|image|>", "special": True},
                ],
            }
            tok_cfg = {
                "added_tokens_decoder": {
                    "0": {"content": "<pad>"},
                    "200": {"content": "<unused0>"},
                    "258880": {"content": "<|image|>"},
                }
            }

            (src / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            (src / "generation_config.json").write_text(json.dumps(gen_cfg), encoding="utf-8")
            (src / "tokenizer.json").write_text(json.dumps(tok_json), encoding="utf-8")
            (src / "tokenizer_config.json").write_text(json.dumps(tok_cfg), encoding="utf-8")
            (src / "processor_config.json").write_text("{\"processor_class\": \"Gemma4Processor\"}", encoding="utf-8")
            (src / "tokenizer.model").write_bytes(b"fake-spm-protobuf")

            summary = inspect_checkpoint_metadata(src, verbose=False)
            self.assertTrue(summary["tokenizer_model_present"])
            self.assertEqual(summary["special_ids"]["[multimodal]"], 5)
            self.assertEqual(summary["missing_processor_files"], [])

            kept_old_ids = [0, 1, 2, 3, 4, 5, 98, 106, 255999, 258880, 258881, 258882]
            id_map = build_id_map(kept_old_ids)
            remap_report = remap_and_save_metadata(src, dst, kept_old_ids, [], id_map)

            self.assertFalse((dst / "tokenizer.model").exists())
            self.assertEqual(remap_report["tokenizer_model_action"], "removed_use_tokenizer_json_only")
            self.assertEqual(remap_report["missing_processor_files"], [])
            self.assertIn("processor_config.json", remap_report["copied_auxiliary_files"])
            self.assertTrue((dst / "processor_config.json").exists())
            self.assertIn("[multimodal]", remap_report["injected_added_tokens"])

            new_tok_json = json.loads((dst / "tokenizer.json").read_text(encoding="utf-8"))
            added_by_content = {e["content"]: e for e in new_tok_json["added_tokens"]}
            self.assertIn("[multimodal]", added_by_content)
            self.assertEqual(added_by_content["[multimodal]"]["id"], id_map[5])
            self.assertFalse(added_by_content["[multimodal]"]["normalized"])
            self.assertTrue(added_by_content["[multimodal]"]["special"])
            self.assertEqual(added_by_content["<|think|>"]["id"], id_map[98])
            self.assertEqual(added_by_content["<|audio|>"]["id"], id_map[258881])

            new_cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
            assert_no_oob_token_ids(new_cfg, len(kept_old_ids))
            self.assertEqual(new_cfg["vocab_size"], 12)
            self.assertEqual(new_cfg["image_token_id"], id_map[258880])
            self.assertEqual(new_cfg["eos_token_id"], [id_map[1], id_map[106]])
            self.assertEqual(new_cfg["text_config"]["vocab_size"], 12)
            self.assertEqual(new_cfg["text_config"]["vocab_size_per_layer_input"], 12)
            self.assertEqual(new_cfg["text_config"]["boi_token_id"], id_map[255999])
            self.assertEqual(new_cfg["text_config"]["eoi_token_id"], id_map[258882])

            new_tok_cfg = json.loads((dst / "tokenizer_config.json").read_text(encoding="utf-8"))
            self.assertNotIn("200", new_tok_cfg["added_tokens_decoder"])
            self.assertIn(str(id_map[258880]), new_tok_cfg["added_tokens_decoder"])
            self.assertIn(str(id_map[5]), new_tok_cfg["added_tokens_decoder"])
            self.assertEqual(new_tok_cfg["added_tokens_decoder"][str(id_map[5])]["content"], "[multimodal]")
            self.assertFalse(new_tok_cfg["added_tokens_decoder"][str(id_map[5])]["normalized"])

            # Verify remap_config_dict fails loudly if a required token ID was pruned.
            bad_map = dict(id_map)
            del bad_map[258881]
            with self.assertRaises(KeyError):
                remap_config_dict(cfg, bad_map, len(bad_map))

    def test_ensure_processor_configs_fallback_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "ckpt"
            target.mkdir()

            def fake_download(repo_id: str, filename: str, out_dir: Path, token: str | None) -> None:
                if repo_id == "google/gemma-4-E2B-it":
                    raise PermissionError("401 gated repo")
                if repo_id == "Austriani/Gemma-4-E2B-it-ARA-Abliterated" and filename == "processor_config.json":
                    (out_dir / filename).write_text("{}", encoding="utf-8")

            res = ensure_processor_configs(
                target,
                fallback_repos=("google/gemma-4-E2B-it", "Austriani/Gemma-4-E2B-it-ARA-Abliterated"),
                download_fn=fake_download,
            )
            self.assertEqual(res["missing_required"], [])
            self.assertEqual(res["sources"]["processor_config.json"], "Austriani/Gemma-4-E2B-it-ARA-Abliterated")


@unittest.skipUnless(GGUF_BASELINE.exists(), "Baseline GGUF not present in brain/")
class TestRealGemma4VocabSmoke(unittest.TestCase):
    def test_real_262k_vocab_pruning_to_49152(self) -> None:
        vocab_by_id, raw_merges = _load_vocab_and_merges_from_gguf(GGUF_BASELINE)
        self.assertEqual(len(vocab_by_id), 262144)

        whitelist_ids = collect_mandatory_whitelist(vocab_by_id=vocab_by_id)
        self.assertIn(5, whitelist_ids)
        self.assertIn(98, whitelist_ids)
        self.assertIn(255999, whitelist_ids)
        self.assertIn(258880, whitelist_ids)
        self.assertIn(258881, whitelist_ids)

        kept_old_ids, pruned_merges_raw, report = select_hybrid_vocab(
            vocab_by_id=vocab_by_id,
            raw_merges=raw_merges,
            whitelist_ids=whitelist_ids,
            target_size=49152,
        )
        self.assertEqual(len(kept_old_ids), 49152)
        self.assertEqual(len(kept_old_ids) % 128, 0)
        self.assertEqual(report["unreachable_merge_tokens"], 0)
        self.assertGreater(len(pruned_merges_raw), 10000)
        self.assertGreater(report["kept_vi_diacritic_count"], 1000)

        # Ensure IDs 0..5 stay in their exact slots after mapping.
        id_map = build_id_map(kept_old_ids)
        for tid in range(6):
            self.assertEqual(id_map[tid], tid)

        # Ensure zero <unused...> or Unicode emoji remain.
        for tid in kept_old_ids:
            tok = vocab_by_id[tid]
            self.assertFalse(is_unused_token(tok))
            self.assertFalse(is_emoji_or_banned_symbol(tok))


class TestNotebookStructure(unittest.TestCase):
    def test_colab_notebook_valid_json_order_and_python_syntax(self) -> None:
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)

        code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        full_text = "\n".join(code_cells)

        # Preflight must precede shard slicing, and all 4 gates must be present.
        idx_preflight = full_text.index("inspect_checkpoint_metadata")
        idx_slice = full_text.index("slice_checkpoint_shards")
        idx_g1 = full_text.index("run_gate1_tokenizer")
        idx_g23 = full_text.index("run_gate2_and_gate3")
        idx_g4 = full_text.index("convert_hf_to_gguf.py")
        self.assertLess(idx_preflight, idx_slice)
        self.assertLess(idx_slice, idx_g1)
        self.assertLess(idx_g1, idx_g23)
        self.assertLess(idx_g23, idx_g4)
        self.assertNotIn("from_pretrained(\n    ORIG_DIR", full_text)

        # Parse every Python cell ignoring IPython shell/magic lines ('!' / '%').
        for i, src in enumerate(code_cells):
            cleaned = "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
            )
            ast.parse(cleaned, filename=f"notebook_cell_{i}")


if __name__ == "__main__":
    unittest.main()

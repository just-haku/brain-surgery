"""Unit tests for Step 2 training scaffold (stdlib-only mocks, no torch required)."""

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

from distill.formatter import render_gemma4_training_sample  # noqa: E402
from distill.nc_schema import get_nc_openai_tools  # noqa: E402
from train.check_token_lengths import (  # noqa: E402
    Gemma4BpeEncoder,
    measure_dataset_token_lengths,
    summarize_lengths,
)
from train.dataset_loader import build_assistant_loss_labels, split_train_eval  # noqa: E402
from train.export_gguf import DEFAULT_PLE_CPU_OFFLOAD, build_export_commands  # noqa: E402
from train.mm_integrity import (  # noqa: E402
    freeze_multimodal_parameters,
    hash_multimodal_safetensors,
    is_multimodal_tensor_name,
    verify_multimodal_integrity,
)
from train.train_orpo import ORPOConfig, build_orpo_config_kwargs, freeze_embeddings_and_ple  # noqa: E402
from train.train_sft import (  # noqa: E402
    SFTConfig,
    build_param_groups,
    build_warmup_kwargs,
    resolve_target_and_embed_modules,
)

STEP2_NOTEBOOK = ROOT / "notebooks" / "step2_train_qlora_orpo.ipynb"
MODAL_SCRIPT = ROOT / "scripts" / "run_modal_step2.py"


class _FakeParam:
    def __init__(self, numel: int = 16, requires_grad: bool = True) -> None:
        self._numel = numel
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._numel


class _FakeModel:
    def __init__(self, names: list[str]) -> None:
        self._params = [(n, _FakeParam()) for n in names]

    def named_parameters(self):
        return self._params


class _MockOffsetTokenizer:
    """Character-level offset tokenizer simulating HF fast tokenizer return_offsets_mapping=True."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = True,
        truncation: bool = True,
        max_length: int = 8192,
    ) -> dict[str, list]:
        del add_special_tokens, return_offsets_mapping
        sliced = text[:max_length] if truncation else text
        ids = [ord(ch) for ch in sliced]
        offsets = [(i, i + 1) for i in range(len(sliced))]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


def _write_fake_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    header: dict[str, object] = {}
    offset = 0
    chunks: list[bytes] = []
    for name, raw in tensors.items():
        header[name] = {
            "dtype": "BF16",
            "shape": [max(1, len(raw) // 2)],
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        chunks.append(raw)
    hdr_bytes = json.dumps(header).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_padded = hdr_bytes + b" " * pad
    path.write_bytes(struct.pack("<Q", len(hdr_padded)) + hdr_padded + b"".join(chunks))


class TestMultimodalIntegrity(unittest.TestCase):
    def test_freeze_and_safetensors_sha256_invariant(self) -> None:
        self.assertTrue(is_multimodal_tensor_name("model.vision_tower.blocks.0.attn.q_proj.weight"))
        self.assertTrue(is_multimodal_tensor_name("model.audio_tower.layers.2.fc1.weight"))
        self.assertTrue(is_multimodal_tensor_name("model.multi_modal_projector.linear.weight"))
        self.assertFalse(is_multimodal_tensor_name("model.language_model.layers.0.self_attn.q_proj.weight"))
        self.assertFalse(is_multimodal_tensor_name("model.language_model.embed_tokens_per_layer.weight"))

        fake = _FakeModel([
            "model.vision_tower.patch.weight",
            "model.audio_tower.conv.weight",
            "model.multi_modal_projector.proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
        ])
        stats = freeze_multimodal_parameters(fake)
        self.assertEqual(stats["frozen_tensors"], 3)
        self.assertEqual(stats["trainable_tensors"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            d1 = Path(tmp) / "before"
            d2 = Path(tmp) / "after_ok"
            d3 = Path(tmp) / "after_drifted"
            d1.mkdir()
            d2.mkdir()
            d3.mkdir()

            vis_bytes = b"\x01\x02\x03\x04" * 8
            aud_bytes = b"\x10\x20\x30\x40" * 8
            _write_fake_safetensors(
                d1 / "model.safetensors",
                {
                    "model.vision_tower.patch.weight": vis_bytes,
                    "model.audio_tower.conv.weight": aud_bytes,
                    "model.language_model.embed_tokens.weight": b"\xaa\xbb" * 8,
                },
            )
            # Language embed changed, vision/audio untouched -> must PASS!
            _write_fake_safetensors(
                d2 / "model.safetensors",
                {
                    "model.vision_tower.patch.weight": vis_bytes,
                    "model.audio_tower.conv.weight": aud_bytes,
                    "model.language_model.embed_tokens.weight": b"\xcc\xdd" * 8,
                },
            )
            # 1 byte altered in vision_tower -> must FAIL!
            _write_fake_safetensors(
                d3 / "model.safetensors",
                {
                    "model.vision_tower.patch.weight": b"\xff" + vis_bytes[1:],
                    "model.audio_tower.conv.weight": aud_bytes,
                    "model.language_model.embed_tokens.weight": b"\xcc\xdd" * 8,
                },
            )

            h_before = hash_multimodal_safetensors(d1)
            h_ok = hash_multimodal_safetensors(d2)
            h_bad = hash_multimodal_safetensors(d3)

            res = verify_multimodal_integrity(h_before, h_ok)
            self.assertTrue(res["verified"])
            self.assertEqual(res["tensor_count"], 2)
            with self.assertRaises(AssertionError):
                verify_multimodal_integrity(h_before, h_bad)


class TestDatasetMaskingAndTrainingConfig(unittest.TestCase):
    def test_assistant_only_loss_masking_excludes_prompt_and_tool_responses(self) -> None:
        tools = get_nc_openai_tools(["read_symbol"])
        msgs = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Inspect f"},
            {
                "role": "assistant",
                "reasoning_content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read_symbol", "arguments": '{"path":"a.rs","symbol":"f"}'}}
                ],
                "content": None,
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read_symbol", "content": "fn f() -> i32 { 1 }"},
            {"role": "assistant", "reasoning_content": None, "content": "f returns 1."},
        ]
        rendered = render_gemma4_training_sample(msgs, tools=tools, enable_thinking=False)
        text = rendered["text"]
        out = build_assistant_loss_labels(
            text, rendered["assistant_spans"], _MockOffsetTokenizer(), max_seq_length=8192
        )

        labels = out["labels"]
        self.assertEqual(len(labels), len(text))

        tr_start = text.index("<|tool_response>")
        tr_end = text.index("<tool_response|>") + len("<tool_response|>")
        self.assertTrue(all(lb == -100 for lb in labels[: rendered["assistant_spans"][0][0]]))
        self.assertTrue(all(lb == -100 for lb in labels[tr_start:tr_end]))

        unmasked_text = "".join(
            chr(lb) for lb in labels if lb != -100
        )
        self.assertIn("<|tool_call>call:read_symbol{", unmasked_text)
        self.assertIn("f returns 1.<turn|>", unmasked_text)
        self.assertNotIn("<|tool_response>", unmasked_text)

        tr, ev = split_train_eval([{"id": i} for i in range(20)], eval_ratio=0.1, seed=42)
        self.assertEqual((len(tr), len(ev)), (18, 2))

    def test_target_resolution_param_groups_and_orpo_freeze(self) -> None:
        all_modules = [
            "model.vision_tower.blocks.0.attn.q_proj",
            "model.vision_tower.embed_vision",
            "model.audio_tower.layers.0.self_attn.k_proj",
            "model.language_model.embed_tokens",
            "model.language_model.embed_tokens_per_layer",
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "lm_head",
        ]
        cfg_sft = SFTConfig(train_embeddings=True)
        resolved = resolve_target_and_embed_modules(all_modules, cfg_sft)
        self.assertEqual(
            resolved["target_modules"],
            [
                "model.language_model.layers.0.mlp.down_proj",
                "model.language_model.layers.0.self_attn.q_proj",
            ],
        )
        self.assertEqual(
            resolved["modules_to_save"],
            [
                "lm_head",
                "model.language_model.embed_tokens",
                "model.language_model.embed_tokens_per_layer",
            ],
        )

        cfg_no_emb = SFTConfig(train_embeddings=False)
        self.assertEqual(resolve_target_and_embed_modules(all_modules, cfg_no_emb)["modules_to_save"], [])

        fake_params = [
            ("base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight", _FakeParam()),
            ("base_model.model.language_model.embed_tokens.modules_to_save.weight", _FakeParam()),
            ("base_model.model.language_model.embed_tokens_per_layer.modules_to_save.weight", _FakeParam()),
            ("base_model.lm_head.modules_to_save.weight", _FakeParam()),
        ]
        groups = build_param_groups(fake_params, cfg_sft)
        lrs = {round(g["lr"], 8): len(g["params"]) for g in groups}
        self.assertEqual(lrs[round(1.5e-4, 8)], 1)
        self.assertEqual(lrs[round(1.5e-5, 8)], 3)

        orpo_model = _FakeModel([p[0] for p in fake_params])
        frozen_cnt = freeze_embeddings_and_ple(orpo_model)
        self.assertEqual(frozen_cnt, 3)


class TestExportAndStep2Notebook(unittest.TestCase):
    def test_token_length_audit_bpe_and_summary(self) -> None:
        self.assertTrue(summarize_lengths([10, 20, 30, 40], 40)["fits_100_pct"])
        bad_sum = summarize_lengths([10, 50], 40)
        self.assertEqual(bad_sum["exceeds_limit"], 1)
        self.assertFalse(bad_sum["fits_100_pct"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tok_json = {
                "model": {
                    "vocab": {
                        "<pad>": 0,
                        "<eos>": 1,
                        "<bos>": 2,
                        "<unk>": 3,
                        "<|turn>": 4,
                        "<turn|>": 5,
                        "▁": 6,
                        "h": 7,
                        "i": 8,
                        "hi": 9,
                        "<0x0A>": 10,
                    },
                    "merges": [["h", "i"]],
                },
                "added_tokens": [
                    {"id": 4, "content": "<|turn>", "special": True},
                    {"id": 5, "content": "<turn|>", "special": True},
                ],
            }
            tj_path = tmp_dir / "tokenizer.json"
            tj_path.write_text(json.dumps(tok_json), encoding="utf-8")
            enc = Gemma4BpeEncoder.from_tokenizer_json(tj_path)
            ids = enc.encode("<|turn>hi<turn|>")
            self.assertEqual(ids, [4, 9, 5])

            sft_p = tmp_dir / "sft.jsonl"
            orpo_p = tmp_dir / "orpo.jsonl"
            sft_p.write_text(json.dumps({"text": "<|turn>hi<turn|>"}) + "\n", encoding="utf-8")
            orpo_p.write_text(
                json.dumps({"prompt": "<|turn>", "chosen": "hi", "rejected": "hi hi"}) + "\n",
                encoding="utf-8",
            )
            rep = measure_dataset_token_lengths(sft_p, orpo_p, tokenizer_json=tj_path)
            self.assertEqual(rep["unk_count"], 0)
            self.assertEqual(rep["sft_text"]["max"], 3)
            self.assertEqual(rep["orpo_rejected_total"]["max"], 4)

    def test_build_export_commands_and_notebook_ast(self) -> None:
        cmds = build_export_commands(
            model_dir="outputs/final_bf16",
            out_dir="outputs/gguf",
            llama_cpp_dir="/content/llama.cpp",
            quants=("IQ4_XS", "Q4_K_M"),
            mmproj_outtypes=("bf16", "q8_0"),
            mmproj_model_dir="cache/base_ckpt",
            imatrix_calib_file="work/calib.txt",
            imatrix_ngl=99,
            python_exe="python",
        )
        self.assertIn("--mmproj", cmds["convert_mmproj_bf16"])
        self.assertIn("bf16", cmds["convert_mmproj_bf16"])
        self.assertIn("--mmproj", cmds["convert_mmproj_q8_0"])
        self.assertIn("q8_0", cmds["convert_mmproj_q8_0"])
        self.assertEqual(cmds["convert_text_bf16"][2], str(Path("outputs/final_bf16")))
        self.assertEqual(cmds["convert_mmproj_bf16"][2], str(Path("cache/base_ckpt")))
        self.assertEqual(cmds["convert_mmproj_q8_0"][2], str(Path("cache/base_ckpt")))
        self.assertIn("imatrix", cmds)
        self.assertEqual(cmds["imatrix"][-2:], ["-ngl", "99"])
        self.assertIn("--imatrix", cmds["quantize_IQ4_XS"])
        self.assertIn("--imatrix", cmds["quantize_Q4_K_M"])
        self.assertEqual(cmds["quantize_IQ4_XS"][-1], "IQ4_XS")

        nb = json.loads(STEP2_NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        full_src = "\n".join(code_cells)
        self.assertIn("haku-nekooo/gemma4-e4b-nuclear-pruned", full_src)
        self.assertIn("haku-nekooo/gemma4-e4b-nuclear-agent", full_src)
        self.assertIn("DATA_SOURCE_MODE", full_src)
        self.assertIn("IQ4_XS", full_src)
        self.assertIn("q8_0", full_src)
        self.assertIn("per_layer_token_embd", DEFAULT_PLE_CPU_OFFLOAD)
        self.assertIn("DEFAULT_PLE_CPU_OFFLOAD", full_src)
        self.assertIn("measure_dataset_token_lengths", full_src)
        self.assertIn("run_sft_training", full_src)
        self.assertIn("run_orpo_training", full_src)
        self.assertIn("export_modular_gguf", full_src)
        for idx, src in enumerate(code_cells):
            ast.parse(src, filename=f"step2_cell_{idx}")

    def test_warmup_kwargs_and_modal_step2_script(self) -> None:
        class _V4Args:
            def __init__(self, warmup_ratio: float = 0.0) -> None:
                del warmup_ratio

        class _V5Args:
            def __init__(self, warmup_steps: float = 0.0) -> None:
                del warmup_steps

        self.assertEqual(build_warmup_kwargs(_V4Args, 0.05), {"warmup_ratio": 0.05})
        self.assertEqual(build_warmup_kwargs(_V5Args, 0.05), {"warmup_steps": 0.05})

        class _OldTRLORPOConfig:
            def __init__(
                self,
                output_dir: str,
                beta: float = 0.1,
                max_length: int = 1024,
                max_prompt_length: int = 512,
                dataset_num_proc: int = 1,
                warmup_ratio: float = 0.0,
                **_: object,
            ) -> None:
                pass

        class _NewTRLORPOConfig:
            def __init__(
                self,
                output_dir: str,
                beta: float = 0.1,
                max_length: int = 1024,
                warmup_steps: float = 0.0,
                optim: str = "adamw_torch",
            ) -> None:
                pass

        orpo_cfg = ORPOConfig(use_4bit=False)
        old_kw = build_orpo_config_kwargs(_OldTRLORPOConfig, orpo_cfg, out_dir="outputs/orpo", use_8bit_adam=True)
        self.assertIn("max_prompt_length", old_kw)
        self.assertIn("dataset_num_proc", old_kw)

        new_kw = build_orpo_config_kwargs(_NewTRLORPOConfig, orpo_cfg, out_dir="outputs/orpo", use_8bit_adam=True)
        self.assertNotIn("max_prompt_length", new_kw)
        self.assertNotIn("dataset_num_proc", new_kw)
        self.assertEqual(new_kw["beta"], 0.1)
        self.assertEqual(new_kw["max_length"], 4096)
        self.assertEqual(new_kw["warmup_steps"], 0.05)
        self.assertEqual(new_kw["optim"], "adamw_bnb_8bit")

        modal_src = MODAL_SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(modal_src, filename=str(MODAL_SCRIPT))
        for tok in (
            "nuclear-distill-vol",
            "huggingface-secret",
            'gpu="A100"',
            "timeout=14400",
            "vol.commit()",
            "haku-nekooo/gemma4-e4b-nuclear-pruned",
            "haku-nekooo/gemma4-e4b-nuclear-agent",
            "haku-nekooo/gemma4-e2b-nuclear-pruned",
            "haku-nekooo/gemma4-e2b-nuclear-agent",
            "IQ4_XS",
            "q8_0",
        ):
            self.assertIn(tok, modal_src)

        presets_val = next(
            stmt.value
            for stmt in tree.body
            if isinstance(stmt, ast.AnnAssign)
            and getattr(stmt.target, "id", None) == "VARIANT_PRESETS"
        )
        presets = ast.literal_eval(presets_val)
        self.assertTrue(presets["e4b"]["train_embeddings"])
        self.assertFalse(presets["e2b"]["train_embeddings"])
        self.assertEqual(presets["e4b"]["learning_rate"], 1.5e-4)
        self.assertEqual(presets["e2b"]["learning_rate"], 2.0e-4)


if __name__ == "__main__":
    unittest.main()

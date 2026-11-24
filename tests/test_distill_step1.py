"""Unit tests for Step 1 Native Gemma 4 Tool Calling, formatter, validator, and Colab notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from distill.formatter import (  # noqa: E402
    render_gemma4_orpo_pair,
    render_gemma4_training_sample,
    validate_gemma4_wire_format,
)
from distill.gemini_batch import (  # noqa: E402
    parse_and_validate_results,
    write_batch_jsonl_by_model,
)
from distill.nc_schema import (  # noqa: E402
    build_nc_system_prompt,
    build_nc_user_prompt,
    get_nc_openai_tools,
    validate_tool_call,
)
from distill.prompts import MODEL_FLASH, MODEL_PRO, generate_seed_tasks  # noqa: E402
from distill.validator import (  # noqa: E402
    contains_ascii_emoticon,
    has_repetition_loop,
    validate_orpo_pair,
    validate_sft_sample,
)

STEP1_NOTEBOOK = ROOT / "notebooks" / "step1_distill.ipynb"


class TestNcSchema(unittest.TestCase):
    def test_openai_tool_specs_and_validation(self) -> None:
        tools = get_nc_openai_tools()
        self.assertEqual(len(tools), 8)
        by_name = {t["function"]["name"]: t["function"] for t in tools}
        self.assertEqual(by_name["read_symbol"]["parameters"]["required"], ["path", "symbol"])
        self.assertEqual(by_name["search_text"]["parameters"]["required"], ["query"])
        self.assertEqual(
            by_name["edit_file"]["parameters"]["required"],
            ["path", "target", "replacement"],
        )

        self.assertEqual(validate_tool_call("read_symbol", {"path": "src/lib.rs", "symbol": "parse"}), [])
        self.assertEqual(validate_tool_call("search_text", {"query": "edit_file", "ext": "rs", "max_hits": 10}), [])
        self.assertEqual(
            validate_tool_call(
                "edit_file",
                {"path": "src/math.rs", "target": "let x: i32 = 0;", "replacement": "let x: i32 = 1;"},
            ),
            [],
        )

        bad_calls = [
            ("edit_file", {"path": "a.rs", "start_line": 1, "content": "x"}),
            ("edit_file", {"path": "a.rs", "target": "", "replacement": "x"}),
            ("search_text", {"query": "foo", "max_hits": True}),
            ("search_text", {"query": "foo", "max_hits": 0}),
            ("unknown_tool", {"path": "a.rs"}),
        ]
        for tname, targs in bad_calls:
            self.assertTrue(validate_tool_call(tname, targs), f"expected errors for {tname} {targs}")

    def test_system_and_user_prompts(self) -> None:
        sys_p = build_nc_system_prompt("coder")
        self.assertNotIn("JSON ONLY", sys_p)
        self.assertIn("Always read before edit_file", sys_p)

        prompt = build_nc_user_prompt(
            repo_map="src/lib.rs: parse, run",
            ledger="goal: fix parser",
            task="Fix parse()",
            scratchpad_entries=[
                {"step": 1, "tool": "read_outline", "evicted": True, "receipt": "outline src/lib.rs"},
                {"step": 2, "tool": "read_symbol", "evicted": False, "raw_output": "fn parse() -> i32 { 0 }"},
            ],
        )
        for header in ("## REPO_MAP", "## LEDGER", "## TASK", "## SCRATCHPAD", "## NEXT"):
            self.assertIn(header, prompt)
        self.assertIn("[step 1 read_outline -> outline src/lib.rs]", prompt)
        self.assertIn("[step 2 read_symbol]\nfn parse() -> i32 { 0 }", prompt)


class TestGemma4Formatter(unittest.TestCase):
    def test_native_multi_turn_tool_format_and_assistant_spans(self) -> None:
        tools = get_nc_openai_tools(["read_symbol"])
        msgs = [
            {"role": "system", "content": "Sys prompt"},
            {"role": "user", "content": "Check add()"},
            {
                "role": "assistant",
                "reasoning_content": "Read symbol first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_symbol", "arguments": {"symbol": "add", "path": "src/lib.rs"}},
                    }
                ],
                "content": "",
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_symbol",
                "content": "fn add(a: i32) -> i32 { a }",
            },
            {
                "role": "assistant",
                "reasoning_content": "Answer clearly.",
                "content": "Done.",
            },
        ]
        out = render_gemma4_training_sample(msgs, tools=tools, enable_thinking=True)
        text = out["text"]
        self.assertTrue(text.startswith("<bos><|turn>system\n<|think|>\nSys prompt<|tool>declaration:read_symbol{"))
        self.assertIn(
            "<|turn>model\n<|channel>thought\nRead symbol first.\n<channel|>"
            '<|tool_call>call:read_symbol{path:<|"|>src/lib.rs<|"|>,symbol:<|"|>add<|"|>}<tool_call|>'
            '<|tool_response>response:read_symbol{value:<|"|>fn add(a: i32) -> i32 { a }<|"|>}<tool_response|>'
            "<|channel>thought\nAnswer clearly.\n<channel|>Done.<turn|>\n",
            text,
        )
        self.assertEqual(validate_gemma4_wire_format(text), [])

        spans = out["assistant_spans"]
        self.assertEqual(len(spans), 2)
        span1 = text[spans[0][0] : spans[0][1]]
        span2 = text[spans[1][0] : spans[1][1]]
        self.assertTrue(span1.startswith("<|channel>thought\nRead symbol first."))
        self.assertTrue(span1.endswith("<tool_call|>"))
        self.assertNotIn("<|tool_response>", span1)
        self.assertEqual(span2, "<|channel>thought\nAnswer clearly.\n<channel|>Done.<turn|>")

    def test_orpo_post_user_and_post_tool_plus_xml_guard(self) -> None:
        pair = render_gemma4_orpo_pair(
            prompt_messages=[{"role": "user", "content": "Explain Rust i32"}],
            chosen={"reasoning_content": "Ngắn gọn.", "content": "Giữ nguyên `i32`."},
            rejected={"reasoning_content": "Lặp.", "content": "Đổi sang `f64`."},
            enable_thinking=True,
        )
        self.assertTrue(pair["prompt"].endswith("<|turn>model\n"))
        self.assertEqual(
            pair["chosen"],
            "<|channel>thought\nNgắn gọn.\n<channel|>Giữ nguyên `i32`.<turn|>\n",
        )

        post_tool_pair = render_gemma4_orpo_pair(
            prompt_messages=[
                {"role": "user", "content": "Check add"},
                {
                    "role": "assistant",
                    "reasoning_content": "Read add.",
                    "tool_calls": [{"id": "c1", "name": "read_symbol", "arguments": {"path": "a.rs", "symbol": "add"}}],
                    "content": "",
                },
                {"role": "tool", "tool_call_id": "c1", "name": "read_symbol", "content": "fn add() {}"},
            ],
            chosen={"reasoning_content": "Done.", "content": "OK."},
            rejected={"reasoning_content": "Loop.", "content": "Bad."},
            tools=get_nc_openai_tools(["read_symbol"]),
            enable_thinking=True,
        )
        self.assertTrue(post_tool_pair["prompt"].endswith("<tool_response|>"))
        self.assertNotIn("<tool_response|><turn|>", post_tool_pair["prompt"])

        self.assertTrue(validate_gemma4_wire_format("<thinking>bad</thinking>"))


class TestValidatorPolicy(unittest.TestCase):
    def test_ascii_emoticon_no_code_false_positives(self) -> None:
        for text in ("Xong rồi bro :)", "Chạy mượt :)))!", "Toang rồi ._.", "Chán ghê .-.", "Cười =))", "Ổn ^_^"):
            self.assertTrue(contains_ascii_emoticon(text), text)

        for code in (
            "fn main() { let x: i32 = 1; }",
            "def f(x: int) -> None:\n    return (x + 1)",
            "https://example.com:8080/api/v1",
            "vals = arr[0:1]; d = {'a': (1, 2)}",
            "std::vector<int> v{1, 2};",
        ):
            self.assertFalse(contains_ascii_emoticon(code), code)

    def test_sft_policy_and_loop_detection(self) -> None:
        good_chat = {
            "id": "c1",
            "bucket": "E_dev_chat",
            "mode": "chat",
            "role": "chat",
            "enable_thinking": True,
            "messages": [
                {"role": "system", "content": "Helper"},
                {"role": "user", "content": "Build chạy xanh chưa?"},
                {
                    "role": "assistant",
                    "reasoning_content": "Xác nhận trạng thái build ngắn gọn.",
                    "content": "Chạy xanh lè rồi nhé bro :)))```python\nassert 1 == 1\n```",
                },
            ],
        }
        self.assertEqual(validate_sft_sample(good_chat), [])

        bad_emoji = {
            **good_chat,
            "messages": [
                good_chat["messages"][0],
                good_chat["messages"][1],
                {"role": "assistant", "reasoning_content": "Ok 🔥", "content": "Xong rồi!"},
            ],
        }
        self.assertTrue(any("Unicode emoji" in e for e in validate_sft_sample(bad_emoji)))

        bad_code_emoticon = {
            **good_chat,
            "messages": [
                good_chat["messages"][0],
                good_chat["messages"][1],
                {"role": "assistant", "reasoning_content": "Ok", "content": "Xong:\n```python\n# :)\npass\n```"},
            ],
        }
        self.assertTrue(any("code block" in e for e in validate_sft_sample(bad_code_emoticon)))

        bad_thought_emoticon = {
            **good_chat,
            "messages": [
                good_chat["messages"][0],
                good_chat["messages"][1],
                {"role": "assistant", "reasoning_content": "Vui quá :)", "content": "Xong."},
            ],
        }
        self.assertTrue(any("thinking block" in e for e in validate_sft_sample(bad_thought_emoticon)))

        bad_json_leak = {
            **good_chat,
            "messages": [
                good_chat["messages"][0],
                good_chat["messages"][1],
                {
                    "role": "assistant",
                    "reasoning_content": "Ok",
                    "content": '{"thought":"leak","action":{"tool":"read_symbol"}}',
                },
            ],
        }
        self.assertTrue(any("raw nc-agent JSON" in e for e in validate_sft_sample(bad_json_leak)))

        loop_txt = ("kiểm tra biến này trong hàm xử lý lỗi " * 4).strip()
        self.assertTrue(has_repetition_loop(loop_txt))


class TestPromptsBatchAndNotebook(unittest.TestCase):
    def test_routing_and_native_batch_parse_end_to_end(self) -> None:
        tasks = generate_seed_tasks(total_sft=40, total_orpo=8, seed=7)
        self.assertEqual(len(tasks), 48)
        for t in tasks:
            if t["bucket"] in ("A_nc_agent", "C_direct_code", "ORPO_pairs"):
                self.assertEqual(t["model"], MODEL_PRO)
            else:
                self.assertEqual(t["model"], MODEL_FLASH)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            batch_files = write_batch_jsonl_by_model(tasks, tmp_dir / "batch_in")
            self.assertIn(MODEL_PRO, batch_files)
            self.assertIn(MODEL_FLASH, batch_files)

            valid_nc = {
                "id": "sft_1",
                "bucket": "A_nc_agent",
                "mode": "nc_agent",
                "role": "coder",
                "enable_thinking": True,
                "messages": [
                    {"role": "system", "content": build_nc_system_prompt("coder")},
                    {"role": "user", "content": build_nc_user_prompt(task="Fix calc")},
                    {
                        "role": "assistant",
                        "reasoning_content": "Read calc symbol first.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_symbol", "arguments": {"path": "src/lib.rs", "symbol": "calc"}},
                            }
                        ],
                        "content": "",
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "read_symbol", "content": "fn calc(x: i32) -> i32 { x }"},
                    {
                        "role": "assistant",
                        "reasoning_content": "Return plain explanation.",
                        "content": "Symbol `calc` returns `x` unchanged as `i32`.",
                    },
                ],
            }
            valid_chat = {
                "id": "sft_2",
                "bucket": "E_dev_chat",
                "mode": "chat",
                "role": "chat",
                "enable_thinking": False,
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "reasoning_content": "", "content": "Chào bro :)"},
                ],
            }
            bad_edit = {
                "id": "sft_bad_edit",
                "bucket": "A_nc_agent",
                "mode": "nc_agent",
                "role": "coder",
                "enable_thinking": True,
                "messages": [
                    {"role": "user", "content": "Edit"},
                    {
                        "role": "assistant",
                        "reasoning_content": "Edit line 1.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "edit_file", "arguments": {"path": "a.rs", "start_line": 1, "content": "x"}},
                            }
                        ],
                        "content": "",
                    },
                ],
            }
            bad_emoji = {
                "id": "sft_bad_emoji",
                "bucket": "B_bilingual_debug",
                "mode": "chat",
                "role": "chat",
                "enable_thinking": True,
                "messages": [
                    {"role": "user", "content": "Explain"},
                    {"role": "assistant", "reasoning_content": "Suy luận", "content": "Ổn rồi 🚀"},
                ],
            }
            valid_orpo = {
                "id": "orpo_1",
                "bucket": "ORPO_pairs",
                "failure_type": "type_mutation",
                "mode": "nc_agent",
                "role": "coder",
                "enable_thinking": True,
                "prompt_messages": [
                    {"role": "system", "content": build_nc_system_prompt("coder")},
                    {"role": "user", "content": build_nc_user_prompt(task="Fix add keeping i32")},
                ],
                "chosen": {
                    "reasoning_content": "Keep i32 signature intact.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": {
                                    "path": "src/lib.rs",
                                    "target": "fn add(a: i32) -> i32 { a }",
                                    "replacement": "fn add(a: i32) -> i32 { a + 1 }",
                                },
                            },
                        }
                    ],
                    "content": "",
                },
                "rejected": {
                    "reasoning_content": "Change to f64.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "edit_file",
                                "arguments": {
                                    "path": "src/lib.rs",
                                    "target": "fn add(a: i32) -> i32 { a }",
                                    "replacement": "fn add(a: f64) -> f64 { a + 1.0 }",
                                },
                            },
                        }
                    ],
                    "content": "",
                },
            }

            raw_lines = [
                {"key": "k1", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(valid_nc)}]}}]}},
                {"key": "k1_dup", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(valid_nc)}]}}]}},
                {"key": "k2", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(valid_chat)}]}}]}},
                {"key": "k3", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(bad_edit)}]}}]}},
                {"key": "k4", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(bad_emoji)}]}}]}},
                {"key": "orpo__type_mutation__1", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(valid_orpo)}]}}]}},
                {"key": "k5", "error": "rate limit"},
            ]
            raw_path = tmp_dir / "raw_out.jsonl"
            raw_path.write_text("\n".join(json.dumps(r) for r in raw_lines) + "\n", encoding="utf-8")

            report = parse_and_validate_results(
                raw_jsonl_paths=[raw_path],
                out_sft_path=tmp_dir / "sft.jsonl",
                out_orpo_path=tmp_dir / "orpo.jsonl",
                out_rejected_path=tmp_dir / "rejected.jsonl",
            )
            self.assertEqual(report["valid_sft_before_dedup"], 3)
            self.assertEqual(report["kept_sft"], 2)
            self.assertEqual(report["kept_orpo"], 1)
            self.assertEqual(report["rejected_count"], 3)

            sft_rows = [json.loads(ln) for ln in (tmp_dir / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(sft_rows), 2)
            self.assertIn("<|tool_call>call:read_symbol{", sft_rows[0]["text"])
            self.assertIn("<|tool>declaration:edit_file{", sft_rows[0]["text"])

            orpo_rows = [json.loads(ln) for ln in (tmp_dir / "orpo.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(orpo_rows), 1)
            self.assertIn("<|tool_call>call:edit_file{", orpo_rows[0]["chosen"])

            bad_orpo = {**valid_orpo, "chosen": valid_orpo["rejected"]}
            self.assertTrue(validate_orpo_pair(bad_orpo))

    def test_step1_notebook_valid_json_and_python_ast(self) -> None:
        nb = json.loads(STEP1_NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)

        code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        full_text = "\n".join(code_cells)
        self.assertLess(full_text.index("apply_chat_template"), full_text.index("run_sync_smoke"))
        self.assertIn("RUN_FULL_BATCH = False", full_text)

        for i, src in enumerate(code_cells):
            cleaned = "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
            )
            ast.parse(cleaned, filename=f"step1_cell_{i}")


if __name__ == "__main__":
    unittest.main()

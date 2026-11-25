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
    FINAL_STEP_PROMPT,
    WORKER_CODER_TOOLS,
    build_nc_system_prompt,
    build_nc_user_prompt,
    format_repo_map_dir,
    format_repo_map_file,
    get_nc_openai_tools,
    validate_tool_call,
)
from distill.prompts import MODEL_FLASH, MODEL_PRO, generate_seed_tasks  # noqa: E402
from distill.validator import (  # noqa: E402
    contains_ascii_emoticon,
    convert_and_validate_openai_jsonl,
    has_repetition_loop,
    validate_orpo_pair,
    validate_sft_sample,
)

STEP1_NOTEBOOK = ROOT / "notebooks" / "step1_distill.ipynb"


class TestNcSchema(unittest.TestCase):
    def test_openai_tool_specs_and_validation(self) -> None:
        tools = get_nc_openai_tools()
        self.assertEqual(len(tools), 9)
        worker_tools = get_nc_openai_tools(depth=1)
        self.assertEqual(len(worker_tools), len(WORKER_CODER_TOOLS))
        self.assertNotIn("spawn_subagent", [t["function"]["name"] for t in worker_tools])

        by_name = {t["function"]["name"]: t["function"] for t in tools}
        self.assertEqual(by_name["read_symbol"]["parameters"]["required"], ["path", "symbol"])
        self.assertEqual(by_name["search_text"]["parameters"]["required"], ["query"])
        self.assertEqual(
            by_name["read_slice"]["parameters"]["required"],
            ["path", "start_line", "end_line"],
        )
        self.assertEqual(
            by_name["spawn_subagent"]["parameters"]["required"],
            ["role", "task", "files"],
        )
        self.assertEqual(
            by_name["edit_file"]["parameters"]["required"],
            ["path", "target", "replacement"],
        )

        self.assertEqual(validate_tool_call("read_symbol", {"path": "src/lib.rs", "symbol": "parse"}), [])
        self.assertEqual(
            validate_tool_call("read_slice", {"path": "src/lib.rs", "start_line": 1, "end_line": 20}),
            [],
        )
        self.assertEqual(validate_tool_call("search_text", {"query": "edit_file", "ext": "rs", "max_hits": 10}), [])
        self.assertEqual(
            validate_tool_call(
                "spawn_subagent",
                {"role": "explorer", "task": "Inspect types", "files": ["src/lib.rs"]},
            ),
            [],
        )
        self.assertEqual(
            validate_tool_call(
                "spawn_subagent",
                {"role": "explorer", "task": "Scan workspace", "files": []},
            ),
            [],
        )
        self.assertEqual(
            validate_tool_call(
                "edit_file",
                {"path": "src/math.rs", "target": "let x: i32 = 0;", "replacement": "let x: i32 = 1;"},
            ),
            [],
        )

        bad_calls = [
            ("edit_file", {"path": "a.rs", "start_line": 1, "content": "x"}),
            ("edit_file", {"path": "a.rs", "target": "old", "replacement": "new", "content": "whole"}),
            ("edit_file", {"path": "a.rs", "target": "same", "replacement": "same"}),
            ("edit_file", {"path": "a.rs", "target": "", "replacement": "x"}),
            ("read_slice", {"path": "a.rs"}),
            ("read_slice", {"path": "a.rs", "start_line": 20, "end_line": 10}),
            ("spawn_subagent", {"role": "explorer", "task": "Missing files"}),
            ("spawn_subagent", {"role": "admin", "task": "Nope", "files": []}),
            ("spawn_subagent", {"role": "coder", "task": "Ok", "files": "a.rs"}),
            ("search_text", {"query": "foo", "max_hits": True}),
            ("search_text", {"query": "foo", "max_hits": 0}),
            ("unknown_tool", {"path": "a.rs"}),
        ]
        for tname, targs in bad_calls:
            self.assertTrue(validate_tool_call(tname, targs), f"expected errors for {tname} {targs}")

    def test_system_and_user_prompts(self) -> None:
        sys_lead = build_nc_system_prompt("coder", depth=0)
        self.assertNotIn("JSON ONLY", sys_lead)
        self.assertIn("Always read exact text before edit_file", sys_lead)
        self.assertIn("spawn_subagent", sys_lead)

        sys_worker = build_nc_system_prompt("coder", depth=1)
        self.assertNotIn("spawn_subagent", sys_worker)

        self.assertEqual(
            format_repo_map_file("src/lib.rs", 45, [("parse", 10), ("run", 28)]),
            "src/lib.rs: parse@10, run@28 [45L]",
        )
        self.assertEqual(format_repo_map_file("README.md", 12), "README.md: [12L]")
        self.assertEqual(
            format_repo_map_dir("crates/auth/", 18, 9400, 42),
            "crates/auth/ [18f, 9400L, 42 syms]",
        )
        self.assertEqual(
            FINAL_STEP_PROMPT,
            "[FINAL STEP] Answer TASK concisely in plain text now; do not call tools.",
        )

        native_prompt = build_nc_user_prompt(
            code_graph="src/lib.rs:L10 fn parse | called_by: run@src/main.rs",
            repo_map=format_repo_map_file("src/lib.rs", 45, [("parse", 10)]),
            ledger="goal: fix parser",
            task="Fix parse()",
        )
        for header in ("## REPO_MAP", "[CODE_GRAPH 1-HOP]", "[FILES]", "## LEDGER", "## TASK"):
            self.assertIn(header, native_prompt)
        self.assertNotIn("## SCRATCHPAD", native_prompt)
        self.assertNotIn("## NEXT", native_prompt)

        legacy_prompt = build_nc_user_prompt(
            repo_map="src/lib.rs: [45L]",
            task="Fix parse()",
            scratchpad_entries=[
                {"step": 1, "tool": "read_outline", "evicted": True, "receipt": "outline src/lib.rs"},
                {"step": 2, "tool": "read_symbol", "evicted": False, "raw_output": "fn parse() -> i32 { 0 }"},
            ],
            legacy_gbnf=True,
        )
        for header in ("## REPO_MAP", "## TASK", "## SCRATCHPAD", "## NEXT"):
            self.assertIn(header, legacy_prompt)
        self.assertIn("[step 1 read_outline -> outline src/lib.rs]", legacy_prompt)
        self.assertIn("[step 2 read_symbol]\nfn parse() -> i32 { 0 }", legacy_prompt)


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
            '<|tool_call>call:read_symbol{path:<|"|>src/lib.rs<|"|>,symbol:<|"|>add<|"|>}<tool_call|><turn|>\n'
            '<|turn>model\n<|tool_response>response:read_symbol{value:<|"|>fn add(a: i32) -> i32 { a }<|"|>}<tool_response|>'
            "<|channel>thought\nAnswer clearly.\n<channel|>Done.<turn|>\n",
            text,
        )
        self.assertEqual(validate_gemma4_wire_format(text), [])

        spans = out["assistant_spans"]
        self.assertEqual(len(spans), 2)
        span1 = text[spans[0][0] : spans[0][1]]
        span2 = text[spans[1][0] : spans[1][1]]
        self.assertTrue(span1.startswith("<|channel>thought\nRead symbol first."))
        self.assertTrue(span1.endswith("<tool_call|><turn|>"))
        self.assertNotIn("<|tool_response>", span1)
        self.assertEqual(span2, "<|channel>thought\nAnswer clearly.\n<channel|>Done.<turn|>")

    def test_parallel_tool_calls_share_single_turn_end_in_span_and_orpo(self) -> None:
        tools = get_nc_openai_tools(["read_symbol", "read_outline"])
        msgs = [
            {"role": "system", "content": "Sys prompt"},
            {"role": "user", "content": "Inspect both files"},
            {
                "role": "assistant",
                "reasoning_content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read_outline", "arguments": {"path": "src/a.rs"}}},
                    {"id": "c2", "type": "function", "function": {"name": "read_symbol", "arguments": {"path": "src/b.rs", "symbol": "run"}}},
                ],
                "content": "",
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read_outline", "content": "fn helper()"},
            {"role": "tool", "tool_call_id": "c2", "name": "read_symbol", "content": "fn run() {}"},
            {"role": "assistant", "reasoning_content": None, "content": "Checked both."},
        ]
        out = render_gemma4_training_sample(msgs, tools=tools, enable_thinking=False)
        text = out["text"]
        self.assertEqual(validate_gemma4_wire_format(text), [])
        span1 = text[out["assistant_spans"][0][0] : out["assistant_spans"][0][1]]
        self.assertIn("call:read_outline", span1)
        self.assertIn("call:read_symbol", span1)
        self.assertEqual(span1.count("<turn|>"), 1)
        self.assertTrue(span1.endswith("<tool_call|><turn|>"))

        pair = render_gemma4_orpo_pair(
            prompt_messages=[{"role": "user", "content": "Check a.rs"}],
            chosen={
                "role": "assistant",
                "content": None,
                "reasoning_content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_outline", "arguments": {"path": "src/a.rs"}}}],
            },
            rejected={
                "role": "assistant",
                "content": None,
                "reasoning_content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read_outline", "arguments": {"path": "src/a.rs"}}},
                    {"id": "c2", "type": "function", "function": {"name": "read_outline", "arguments": {"path": "src/a.rs"}}},
                ],
            },
            tools=tools,
            enable_thinking=False,
        )
        self.assertTrue(pair["chosen"].endswith("<tool_call|><turn|>\n"))
        self.assertEqual(validate_gemma4_wire_format(pair["prompt"] + pair["chosen"]), [])

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

    def test_adaptive_thinking_caps_duplicate_reads_and_openai_ingest(self) -> None:
        too_long_reason = "x" * 241
        over_cap = {
            "id": "over_cap",
            "thinking": True,
            "messages": [
                {"role": "system", "content": build_nc_system_prompt("coder")},
                {"role": "user", "content": build_nc_user_prompt(task="Explain add()")},
                {"role": "assistant", "reasoning_content": too_long_reason, "content": "Done."},
            ],
        }
        self.assertTrue(any("240" in e for e in validate_sft_sample(over_cap)))

        leaked_thought_when_off = {
            "id": "leak_off",
            "thinking": False,
            "messages": [
                {"role": "system", "content": build_nc_system_prompt("coder")},
                {"role": "user", "content": build_nc_user_prompt(task="Hi")},
                {"role": "assistant", "reasoning_content": "Should be null", "content": "Hi."},
            ],
        }
        self.assertTrue(any("thinking=False" in e for e in validate_sft_sample(leaked_thought_when_off)))

        dup_read = {
            "id": "dup_read",
            "thinking": False,
            "messages": [
                {"role": "system", "content": build_nc_system_prompt("coder")},
                {"role": "user", "content": build_nc_user_prompt(task="Check add()")},
                {
                    "role": "assistant",
                    "reasoning_content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "read_symbol", "arguments": '{"path":"src/lib.rs","symbol":"add"}'}}
                    ],
                    "content": None,
                },
                {"role": "tool", "tool_call_id": "c1", "name": "read_symbol", "content": "fn add() {}"},
                {
                    "role": "assistant",
                    "reasoning_content": None,
                    "tool_calls": [
                        {"id": "c2", "type": "function", "function": {"name": "read_symbol", "arguments": '{"path":"src/lib.rs","symbol":"add"}'}}
                    ],
                    "content": None,
                },
                {"role": "tool", "tool_call_id": "c2", "name": "read_symbol", "content": "fn add() {}"},
                {"role": "assistant", "reasoning_content": None, "content": "Done."},
            ],
        }
        self.assertTrue(any("duplicate consecutive read call" in e for e in validate_sft_sample(dup_read)))

        bad_scratchpad_native = {
            "id": "bad_scratchpad",
            "thinking": False,
            "messages": [
                {"role": "system", "content": build_nc_system_prompt("coder")},
                {"role": "user", "content": "## REPO_MAP\nsrc/lib.rs: [10L]\n\n## TASK\nFix\n\n## SCRATCHPAD\nok"},
                {"role": "assistant", "reasoning_content": None, "content": "Done."},
            ],
        }
        self.assertTrue(
            any("SCRATCHPAD" in e for e in validate_sft_sample(bad_scratchpad_native))
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_in = tmp_dir / "agy_in.jsonl"
            sft_fast = {
                "id": "agy_sft_1",
                "bucket": "A_nc_agent",
                "thinking": False,
                "ctx_profile": "8k",
                "messages": [
                    {"role": "system", "content": build_nc_system_prompt("coder", depth=0)},
                    {"role": "user", "content": build_nc_user_prompt(code_graph="src/lib.rs:L1 fn add", task="Fix add")},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": None,
                        "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "search_text", "arguments": '{"query":"add_old"}'}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "name": "search_text", "content": ""},
                    {"role": "user", "content": FINAL_STEP_PROMPT},
                    {"role": "assistant", "content": "`add` increments `x: i32` by `1`.", "reasoning_content": None},
                ],
            }
            sft_planner = {
                "id": "agy_planner_1",
                "bucket": "planner_dag",
                "grammar": "planner",
                "thinking": False,
                "ctx_profile": "8k",
                "messages": [
                    {"role": "system", "content": build_nc_system_prompt("planner")},
                    {"role": "user", "content": build_nc_user_prompt(task="Split work")},
                    {
                        "role": "assistant",
                        "reasoning_content": None,
                        "content": json.dumps(
                            {
                                "goal": "Split work",
                                "decisions": ["Keep i32"],
                                "tasks": [{"id": "T1", "title": "Update add", "target_files": ["src/lib.rs"], "verify_cmd": "cargo test"}],
                            }
                        ),
                    },
                ],
            }
            orpo_runaway = {
                "id": "agy_orpo_1",
                "bucket": "no_thinking_tool_vs_runaway_cot",
                "prompt": [
                    {"role": "system", "content": build_nc_system_prompt("coder", depth=0)},
                    {"role": "user", "content": build_nc_user_prompt(task="Inspect add")},
                ],
                "chosen": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "read_symbol", "arguments": '{"path":"src/lib.rs","symbol":"add"}'}}
                    ],
                },
                "rejected": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Let me think endlessly about many hypothetical edge cases before doing anything.",
                },
            }
            raw_in.write_text(
                "\n".join([json.dumps(sft_fast), json.dumps(sft_planner), json.dumps(orpo_runaway)]) + "\n",
                encoding="utf-8",
            )
            rep = convert_and_validate_openai_jsonl(
                [raw_in],
                tmp_dir / "sft.jsonl",
                tmp_dir / "orpo.jsonl",
                tmp_dir / "rej.jsonl",
            )
            self.assertEqual(rep["kept_sft"], 2)
            self.assertEqual(rep["kept_orpo"], 1)
            self.assertEqual(rep["rejected_count"], 0)
            self.assertEqual(rep["thinking_true_ratio"], 0.0)
            sft_out_rows = [
                json.loads(line)
                for line in (tmp_dir / "sft.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            fast_row = sft_out_rows[0]
            self.assertNotIn("<turn|>", [fast_row["text"][s:e] for s, e in fast_row["assistant_spans"]])
            self.assertIn(FINAL_STEP_PROMPT, fast_row["text"])
            self.assertNotIn("<|tool>", sft_out_rows[1]["text"])
            orpo_row = json.loads((tmp_dir / "orpo.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("<|channel>thought", orpo_row["chosen"])
            self.assertIn("<|channel>thought", orpo_row["rejected"])

    def test_wave4_per_loop_rules_and_search_text_path(self) -> None:
        search_decl = next(t for t in get_nc_openai_tools() if t["function"]["name"] == "search_text")
        self.assertNotIn("path", search_decl["function"]["parameters"]["properties"])
        self.assertEqual(
            validate_tool_call("search_text", {"query": "str_pool_len", "path": "crates/kval_bin/src"}),
            [],
        )

        valid_micro = {
            "id": "w4_micro_1",
            "bucket": "per_micro_exec_sys",
            "per_stage": "micro_execute",
            "thinking": True,
            "messages": [
                {"role": "system", "content": build_nc_system_prompt("coder", depth=1)},
                {"role": "user", "content": "## REPO_MAP\nsrc/lib.rs: foo@4 [30L]\n\n## TASK\nMicro-task T1: fix foo."},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Inspect foo before editing.",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "search_text", "arguments": '{"query":"foo","path":"src"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "search_text", "content": "No matches"},
                {
                    "role": "assistant",
                    "reasoning_content": None,
                    "content": "STATUS: DONE\nCHANGES: Checked foo safely\nFILES: src/lib.rs\nISSUES: none",
                },
            ],
        }
        self.assertEqual(validate_sft_sample(valid_micro), [])

        missing_receipt = {
            **valid_micro,
            "messages": [*valid_micro["messages"][:-1], {"role": "assistant", "reasoning_content": None, "content": "All done."}],
        }
        self.assertTrue(any("SubTaskReceipt" in e for e in validate_sft_sample(missing_receipt)))

        bad_replan = {
            "id": "w4_replan_bad",
            "bucket": "per_replan",
            "per_stage": "replan",
            "thinking": False,
            "messages": [
                {"role": "system", "content": build_nc_system_prompt("planner")},
                {"role": "user", "content": "## REPO_MAP\nsrc/lib.rs: [10L]\n\n## TASK\nReplan"},
                {
                    "role": "assistant",
                    "reasoning_content": None,
                    "content": json.dumps({"goal": "Fix", "decisions": [], "tasks": [{"id": "T1", "title": "Fix", "target_files": ["a.rs"]}]}),
                },
            ],
        }
        self.assertTrue(any("REPLAN" in e for e in validate_sft_sample(bad_replan)))

        worker_think = {**valid_micro, "bucket": "per_subagent_worker", "thinking": True}
        self.assertTrue(any("thinking=False" in e for e in validate_sft_sample(worker_think)))

        per_orpo = {
            "id": "w4_pref_worker",
            "bucket": "per_worker_no_thinking_vs_thinking_on",
            "prompt": [
                {"role": "system", "content": build_nc_system_prompt("coder", depth=1)},
                {"role": "user", "content": "## REPO_MAP\nsrc/a.rs: [10L]\n\n## TASK\nT1: check"},
            ],
            "chosen": {"role": "assistant", "content": "STATUS: DONE\nCHANGES: Checked\nFILES: src/a.rs\nISSUES: none", "reasoning_content": None},
            "rejected": {"role": "assistant", "content": "STATUS: DONE\nCHANGES: Checked\nFILES: src/a.rs\nISSUES: none", "reasoning_content": "Long worker CoT."},
        }
        self.assertEqual(validate_orpo_pair(per_orpo), [])

    def test_wave5_no_tool_and_anti_loop_orpo_rules(self) -> None:
        coder_no_spawn_sys = build_nc_system_prompt("coder", depth=1)
        no_tool_think = {
            "id": "w5_py_think",
            "bucket": "pure_algo_python",
            "mode": "no_tool",
            "tools": [],
            "thinking": True,
            "messages": [
                {"role": "system", "content": coder_no_spawn_sys},
                {"role": "user", "content": "Write binary search."},
                {
                    "role": "assistant",
                    "reasoning_content": "Check bounds invariant.",
                    "content": "```python\ndef bs(a, x):\n    return -1\n```",
                },
            ],
        }
        self.assertEqual(validate_sft_sample(no_tool_think), [])

        no_tool_regex = {
            **no_tool_think,
            "id": "w5_js_regex",
            "thinking": False,
            "messages": [
                {"role": "system", "content": coder_no_spawn_sys},
                {"role": "user", "content": "Parse version ops."},
                {
                    "role": "assistant",
                    "reasoning_content": None,
                    "content": "Use regex:\n```js\nconst m = /^(>=|<=|>|<|=)?(.+)$/.exec(tok);\n```",
                },
            ],
        }
        self.assertEqual(validate_sft_sample(no_tool_regex), [])

        no_tool_leak = {
            **no_tool_think,
            "id": "w5_bad_leak",
            "thinking": False,
            "messages": [
                {"role": "system", "content": coder_no_spawn_sys},
                {"role": "user", "content": "Explain GIL."},
                {"role": "assistant", "reasoning_content": None, "content": "<|tool_call>call:read_symbol{}<tool_call|>"},
            ],
        }
        self.assertTrue(any("no_tool mode forbids <|tool_call>" in e for e in validate_sft_sample(no_tool_leak)))

        anti_loop_orpo = {
            "id": "w5_pref_loop",
            "bucket": "anti_loop_tool_termination",
            "prompt": [
                {"role": "system", "content": build_nc_system_prompt("coder", depth=0)},
                {"role": "user", "content": "## REPO_MAP\nweb/sync_queue.js: [20L]\n\n## TASK\nTrace retry backoff"},
            ],
            "chosen": {
                "role": "assistant",
                "content": None,
                "reasoning_content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "spawn_subagent", "arguments": '{"role":"CODER","task":"Trace retry","files":["web/sync_queue.js"]}'},
                    }
                ],
            },
            "rejected": {
                "role": "assistant",
                "content": None,
                "reasoning_content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "spawn_subagent", "arguments": '{"role":"CODER","task":"Trace retry","files":["web/sync_queue.js"]}'}},
                    {"id": "c2", "type": "function", "function": {"name": "spawn_subagent", "arguments": '{"role":"CODER","task":"Trace retry","files":["web/sync_queue.js"]}'}},
                ],
            },
        }
        self.assertEqual(validate_orpo_pair(anti_loop_orpo), [])

        clean_code_orpo = {
            "id": "w5_pref_clean",
            "bucket": "clean_code_vs_syntax_hallucination",
            "tools": [],
            "prompt": [
                {"role": "system", "content": coder_no_spawn_sys},
                {"role": "user", "content": "Write top_k_min in Python."},
            ],
            "chosen": {"role": "assistant", "content": "```python\nimport heapq\n```", "reasoning_content": None},
            "rejected": {"role": "assistant", "content": "```python\nimport heapqueue\n```", "reasoning_content": None},
        }
        self.assertEqual(validate_orpo_pair(clean_code_orpo), [])

        v21_coding_orpo = {
            "id": "v21_code_1",
            "bucket": "on_policy_coding_patch",
            "tools": [],
            "prompt": [{"role": "user", "content": "Implement dijkstra in Python."}],
            "chosen": {"role": "assistant", "content": "```python\nimport heapq\n```", "reasoning_content": None},
            "rejected": {"role": "assistant", "content": "```python\nimport heapqueue\n```", "reasoning_content": None},
        }
        self.assertEqual(validate_orpo_pair(v21_coding_orpo), [])

        rendered_no_tool = render_gemma4_training_sample(no_tool_regex["messages"], tools=None, enable_thinking=False)
        self.assertNotIn("<|tool>", rendered_no_tool["text"])

        early_returns = {
            **no_tool_think,
            "id": "w5_early_returns",
            "thinking": False,
            "messages": [
                {"role": "system", "content": coder_no_spawn_sys},
                {"role": "user", "content": "Match clauses."},
                {
                    "role": "assistant",
                    "reasoning_content": None,
                    "content": "```python\n" + "\n".join(f"if c == {i}:\n    return False" for i in range(8)) + "\n```",
                },
            ],
        }
        self.assertEqual(validate_sft_sample(early_returns), [])


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

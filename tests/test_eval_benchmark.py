"""Unit tests for the overnight benchmark harness (SSE parser, scorers, code verifier, registry)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.metrics import CompletionTelemetry, aggregate_telemetry, parse_sse_events  # noqa: E402
from eval.probes import (  # noqa: E402
    CODING_PROBES,
    extract_code_block,
    parse_integer_answer,
    score_adaptive_response,
    score_native_tool_response,
    score_per_loop_response,
    verify_code_solution,
)
from eval.suite import (  # noqa: E402
    aggregate_results,
    build_model_specs,
    refresh_live_nc_for_variant,
    render_markdown_report,
    resolve_local_gguf_path,
)


class TestEvalBenchmark(unittest.TestCase):
    def test_sse_parser_extracts_ttfb_reasoning_tool_calls_and_timings(self) -> None:
        ticks = iter([10.00, 10.05, 10.12, 10.20, 10.25])
        lines = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            b'data: {"choices":[{"delta":{"reasoning_content":"Inspect math."}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_slice","arguments":"{\\"path\\":\\"src/"}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"math.rs\\",\\"start_line\\":1,\\"end_line\\":10}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":120,"completion_tokens":22,"prompt_tokens_details":{"cached_tokens":40},"completion_tokens_details":{"reasoning_tokens":5}},"timings":{"prompt_per_second":310.5,"predicted_per_second":24.2}}\n',
            b"data: [DONE]\n",
        ]
        tel = parse_sse_events(lines, request_start_perf=10.0, clock_fn=lambda: next(ticks))
        self.assertTrue(tel.ok)
        self.assertAlmostEqual(tel.ttfb_ms or 0.0, 0.0, places=1)
        self.assertEqual(tel.reasoning_content, "Inspect math.")
        self.assertEqual(tel.reasoning_tokens, 5)
        self.assertEqual(tel.prompt_tokens, 120)
        self.assertEqual(tel.completion_tokens, 22)
        self.assertEqual(tel.cached_tokens, 40)
        self.assertAlmostEqual(tel.pp_tps or 0.0, 310.5, places=1)
        self.assertAlmostEqual(tel.tg_tps or 0.0, 24.2, places=1)
        self.assertEqual(tel.tool_calls[0]["function"]["name"], "read_slice")
        self.assertEqual(
            json.loads(tel.tool_calls[0]["function"]["arguments"]),
            {"path": "src/math.rs", "start_line": 1, "end_line": 10},
        )

    def test_aggregate_telemetry_stats(self) -> None:
        agg = aggregate_telemetry(
            [
                CompletionTelemetry(
                    ok=True, ttfb_ms=100.0, total_ms=500.0,
                    prompt_tokens=100, completion_tokens=20, reasoning_tokens=0, cached_tokens=50,
                    pp_tps=200.0, tg_tps=20.0,
                ),
                CompletionTelemetry(
                    ok=True, ttfb_ms=200.0, total_ms=700.0,
                    prompt_tokens=150, completion_tokens=30, reasoning_tokens=10, cached_tokens=60,
                    pp_tps=250.0, tg_tps=30.0,
                ),
            ]
        )
        self.assertEqual(agg["requests_ok"], 2)
        self.assertEqual(agg["prompt_tokens_total"], 250)
        self.assertEqual(agg["completion_tokens_total"], 50)
        self.assertEqual(agg["reasoning_tokens_total"], 10)
        self.assertEqual(agg["avg_ttfb_ms"], 150.0)
        self.assertEqual(agg["p95_ttfb_ms"], 200.0)
        self.assertEqual(agg["avg_tg_tps"], 25.0)

    def test_native_tool_and_per_scorers(self) -> None:
        bad_edit = CompletionTelemetry(
            ok=True,
            tool_calls=[
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({"path": "src/math.rs", "content": "bad"}),
                    },
                }
            ],
        )
        self.assertFalse(score_native_tool_response(bad_edit, expected_tool="edit_file")["passed"])

        good_edit = CompletionTelemetry(
            ok=True,
            tool_calls=[
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps(
                            {"path": "src/math.rs", "target": "price + 1", "replacement": "price * 110 / 100"}
                        ),
                    },
                }
            ],
        )
        self.assertTrue(score_native_tool_response(good_edit, expected_tool="edit_file")["passed"])

        good_receipt = CompletionTelemetry(
            ok=True,
            content="STATUS: DONE\nCHANGES: Fixed tax\nFILES: src/math.rs\nISSUES: None",
        )
        self.assertTrue(score_per_loop_response(good_receipt, "micro_execute")["passed"])

        bad_worker = CompletionTelemetry(
            ok=True,
            tool_calls=[
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "spawn_subagent",
                        "arguments": json.dumps({"role": "explorer", "task": "x", "files": []}),
                    },
                }
            ],
        )
        self.assertFalse(score_adaptive_response(bad_worker, "worker_d1")["passed"])

    def test_code_execution_verifier_and_math_parser(self) -> None:
        dijk_probe = next(p for p in CODING_PROBES if p["id"] == "code_py_dijkstra")
        sol = """```python
import heapq
def dijkstra(n, edges, src):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [-1] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if dist[v] == -1 or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
```"""
        self.assertIn("def dijkstra", extract_code_block(sol))
        self.assertTrue(verify_code_solution(dijk_probe, sol)["passed"])
        self.assertEqual(parse_integer_answer("Steps...\nANSWER: 650,190"), 650190)

    def test_model_registry_reuse_and_report_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            reused = tdir / "gemma-4-E2B-it-Q4_K_M.gguf"
            reused.write_bytes(b"0" * 2_000_000)
            specs = build_model_specs(tdir)
            self.assertEqual(resolve_local_gguf_path(specs["e2b_inst"], "Q4_K_M"), reused)

            res_dir = tdir / "results"
            res_dir.mkdir()
            (res_dir / "e2b_inst_IQ4_XS.json").write_text(
                json.dumps(
                    {
                        "model_key": "e2b_inst",
                        "display_name": "Gemma 4 E2B Instruction",
                        "quant": "IQ4_XS",
                        "size_mb": 2100.0,
                        "fits_fat32": True,
                        "cat1_vram_speed": {"ngl_used": 99, "boot_ms": 1200, "loaded_vram_mib": 2600, "peak_vram_mib": 2700},
                        "cat2_vocab_surgery": {"total_unks": 0, "avg_chars_per_token": 3.4},
                        "cat3_native_tools": {"passed": 5, "total": 6, "items": []},
                        "cat4_per_loop": {"passed": 2, "total": 3, "items": []},
                        "cat5_adaptive_thinking": {"passed": 2, "total": 3, "items": []},
                        "cat6_multimodal": {"passed": 3, "total": 4, "ngl_used": 99, "loaded_vram_mib": 3200, "peak_vram_mib": 3400, "items": []},
                        "cat7_coding_algorithms": {"passed": 5, "total": 6, "items": []},
                        "cat8_math_computation": {"thinking_on_passed": 7, "thinking_on_total": 8, "thinking_off_passed": 3, "thinking_off_total": 4},
                        "cat9_verbose_telemetry": {"prompt_tokens_total": 1000, "completion_tokens_total": 300, "reasoning_tokens_total": 50, "cached_tokens_total": 200, "avg_ttfb_ms": 110.0, "p95_ttfb_ms": 180.0, "avg_pp_tps": 140.0, "avg_tg_tps": 22.0},
                        "cat10_live_nuclear_code": {"passed": 4, "total": 5, "tasks": []},
                    }
                ),
                encoding="utf-8",
            )
            rep_md = tdir / "report.md"
            agg = aggregate_results(res_dir, rep_md)
            self.assertEqual(agg["count"], 1)
            txt = rep_md.read_text(encoding="utf-8")
            self.assertIn("Gemma 4 E2B Instruction", txt)
            self.assertIn("`IQ4_XS`", txt)

    def test_refresh_live_nc_preserves_other_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            bdir = tdir / "brain"
            bdir.mkdir()
            (bdir / "baselines" / "e2b").mkdir(parents=True)
            gguf = bdir / "baselines" / "e2b" / "gemma-4-E2B-it-IQ4_XS.gguf"
            gguf.write_bytes(b"0" * 2_000_000)

            work = tdir / "work"
            res_dir = work / "results"
            res_dir.mkdir(parents=True)
            out_file = res_dir / "e2b_inst_IQ4_XS.json"
            out_file.write_text(
                json.dumps(
                    {
                        "model_key": "e2b_inst",
                        "quant": "IQ4_XS",
                        "cat3_native_tools": {"passed": 4, "total": 6},
                        "cat10_live_nuclear_code": {"passed": 1, "total": 5},
                    }
                ),
                encoding="utf-8",
            )

            updated = refresh_live_nc_for_variant(
                model_key="e2b_inst",
                quant="IQ4_XS",
                work_root=work,
                brain_dir=bdir,
                runner_fn=lambda **_kw: {"passed": 5, "total": 5, "tasks": [{"id": "T1", "passed": True}]},
            )
            self.assertEqual(updated["cat3_native_tools"], {"passed": 4, "total": 6})
            self.assertEqual(updated["cat10_live_nuclear_code"]["passed"], 5)
            disk = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(disk["cat10_live_nuclear_code"]["passed"], 5)


if __name__ == "__main__":
    unittest.main()

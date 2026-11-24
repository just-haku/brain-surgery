"""Minimal CLI for Step 1 Gemini synthetic distillation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gemini_batch import (
    parse_and_validate_results,
    poll_and_download_batch,
    run_sync_smoke,
    submit_batch_job,
    write_batch_jsonl_by_model,
)
from .prompts import generate_seed_tasks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 1 Gemini distillation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Generate Gemini Batch JSONLs routed by Pro/Flash")
    prep.add_argument("--out-dir", required=True)
    prep.add_argument("--total-sft", type=int, default=4000)
    prep.add_argument("--total-orpo", type=int, default=600)
    prep.add_argument("--seed", type=int, default=42)

    smoke = sub.add_parser("smoke", help="Run small sync smoke test via GEMINI_API_KEY")
    smoke.add_argument("--out-jsonl", required=True)
    smoke.add_argument("--limit", type=int, default=6)
    smoke.add_argument("--seed", type=int, default=42)

    sub_job = sub.add_parser("submit", help="Submit a prepared Batch JSONL job")
    sub_job.add_argument("--jsonl", required=True)
    sub_job.add_argument("--model", required=True)
    sub_job.add_argument("--name", default="brain-surgery-step1")

    col = sub.add_parser("collect", help="Poll and download finished Batch job JSONL")
    col.add_argument("--job", required=True)
    col.add_argument("--out-jsonl", required=True)
    col.add_argument("--poll-seconds", type=int, default=30)

    val = sub.add_parser("validate", help="Parse and validate raw Gemini JSONL outputs")
    val.add_argument("--inputs", nargs="+", required=True)
    val.add_argument("--out-dir", required=True)
    val.add_argument("--tokenizer-dir", default=None)
    val.add_argument("--max-tokens", type=int, default=4096)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "prepare":
        tasks = generate_seed_tasks(
            total_sft=args.total_sft, total_orpo=args.total_orpo, seed=args.seed
        )
        paths = write_batch_jsonl_by_model(tasks, args.out_dir)
        print(json.dumps({m: str(p) for m, p in paths.items()}, indent=2))
        return 0

    if args.cmd == "smoke":
        tasks = generate_seed_tasks(total_sft=max(5, args.limit), total_orpo=2, seed=args.seed)
        out = run_sync_smoke(tasks[: args.limit], args.out_jsonl)
        print(f"Smoke output saved to {out}")
        return 0

    if args.cmd == "submit":
        job_name = submit_batch_job(args.jsonl, args.model, args.name)
        print(job_name)
        return 0

    if args.cmd == "collect":
        poll_and_download_batch(args.job, args.out_jsonl, poll_seconds=args.poll_seconds)
        return 0

    if args.cmd == "validate":
        tok = None
        if args.tokenizer_dir:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
        out_dir = Path(args.out_dir)
        report = parse_and_validate_results(
            raw_jsonl_paths=args.inputs,
            out_sft_path=out_dir / "sft_trajectories.jsonl",
            out_orpo_path=out_dir / "orpo_pairs.jsonl",
            out_rejected_path=out_dir / "rejected.jsonl",
            tokenizer=tok,
            max_tokens=args.max_tokens,
        )
        (out_dir / "distill_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

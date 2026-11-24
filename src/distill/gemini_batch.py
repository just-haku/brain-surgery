"""Gemini Batch API + sync smoke generator and result parser/validator."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any

from .formatter import render_gemma4_orpo_pair, render_gemma4_training_sample
from .nc_schema import get_nc_openai_tools
from .validator import deduplicate_samples, validate_orpo_pair, validate_sft_sample

TERMINAL_BATCH_STATES = frozenset(
    {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
)


def write_batch_jsonl_by_model(
    tasks: list[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write per-model JSONL files formatted for Gemini Batch API."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        grouped.setdefault(task["model"], []).append(task)

    paths: dict[str, Path] = {}
    for model_id, model_tasks in sorted(grouped.items()):
        safe_name = model_id.replace("/", "_")
        fpath = out / f"batch_{safe_name}.jsonl"
        with fpath.open("w", encoding="utf-8") as f:
            for t in model_tasks:
                row = {
                    "key": t["key"],
                    "request": {
                        "contents": [{"role": "user", "parts": [{"text": t["prompt_text"]}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": float(t.get("temperature", 0.4)),
                        },
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[model_id] = fpath
    return paths


def _get_genai_client(api_key: str | None = None):
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else genai.Client()


def submit_batch_job(
    jsonl_path: str | Path,
    model_id: str,
    display_name: str,
    api_key: str | None = None,
) -> str:
    """Upload batch JSONL and start a Gemini Batch API job, returning job.name."""
    client = _get_genai_client(api_key)
    try:
        uploaded = client.files.upload(file=str(jsonl_path), config={"mime_type": "jsonl"})
    except Exception:  # noqa: BLE001 - SDK signature fallback across google-genai versions
        uploaded = client.files.upload(file=str(jsonl_path))

    job = client.batches.create(
        model=model_id,
        src=uploaded.name,
        config={"display_name": display_name},
    )
    print(f"Submitted batch job {job.name} (model={model_id}, file={uploaded.name})")
    return job.name


def poll_and_download_batch(
    job_name: str,
    out_jsonl_path: str | Path,
    poll_seconds: int = 30,
    api_key: str | None = None,
) -> Path:
    """Wait for a Gemini Batch job to finish and download its output JSONL."""
    client = _get_genai_client(api_key)
    while True:
        job = client.batches.get(name=job_name)
        state_name = getattr(job.state, "name", str(job.state))
        print(f"Batch {job_name}: state={state_name}")
        if state_name in TERMINAL_BATCH_STATES:
            break
        time.sleep(poll_seconds)

    if state_name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job {job_name} ended in non-success state: {state_name}")

    dest_file = job.dest.file_name
    raw_bytes = client.files.download(file=dest_file)
    out = Path(out_jsonl_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw_bytes)
    print(f"Downloaded batch results -> {out}")
    return out


def run_sync_smoke(
    tasks: list[dict[str, Any]],
    out_jsonl_path: str | Path,
    api_key: str | None = None,
) -> Path:
    """Run a small synchronous smoke subset emitting the exact same JSONL shape as Batch API."""
    client = _get_genai_client(api_key)
    out = Path(out_jsonl_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for idx, t in enumerate(tasks, start=1):
            resp = client.models.generate_content(
                model=t["model"],
                contents=t["prompt_text"],
                config={
                    "response_mime_type": "application/json",
                    "temperature": float(t.get("temperature", 0.4)),
                },
            )
            text = getattr(resp, "text", None) or ""
            row = {
                "key": t["key"],
                "response": {
                    "candidates": [{"content": {"parts": [{"text": text}]}}]
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[smoke {idx}/{len(tasks)}] {t['key']} ({t['model']}) -> {len(text)} chars")
    return out


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _extract_payload_from_line(row: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    key = row.get("key")
    if "response" in row:
        resp = row["response"] or {}
        cands = resp.get("candidates") or []
        if not cands:
            raise ValueError(f"Empty candidates in response for key={key}")
        parts = (cands[0].get("content") or {}).get("parts") or []
        raw_text = "".join(str(p.get("text", "")) for p in parts)
        payload = json.loads(_strip_json_fences(raw_text))
        if isinstance(payload, dict) and key and "id" not in payload:
            payload["id"] = key
        return key, payload
    return key, row


def _resolve_tools(item: dict[str, Any]) -> list[dict[str, Any]] | None:
    if item.get("tools"):
        return item["tools"]
    if item.get("mode") == "nc_agent":
        return get_nc_openai_tools()
    return None


def parse_and_validate_results(
    raw_jsonl_paths: list[str | Path],
    out_sft_path: str | Path,
    out_orpo_path: str | Path,
    out_rejected_path: str | Path,
    tokenizer: Any | None = None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Parse Gemini Batch/sync JSONL outputs, validate, deduplicate, and render Gemma 4 files."""
    valid_sft_raw: list[dict[str, Any]] = []
    valid_orpo_raw: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    total_lines = 0

    for path in raw_jsonl_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total_lines += 1
            try:
                row = json.loads(line)
                if "error" in row and row["error"]:
                    raise ValueError(f"API error: {row['error']}")
                key, payload = _extract_payload_from_line(row)
            except Exception as exc:  # noqa: BLE001
                reason = f"parse_error: {exc}"
                rejection_reasons[reason] += 1
                rejected_rows.append({"line": line[:300], "errors": [reason]})
                continue

            is_orpo = "prompt_messages" in payload or (
                isinstance(key, str) and key.startswith("orpo__")
            )
            if is_orpo:
                errs = validate_orpo_pair(payload, tokenizer=tokenizer, max_tokens=max_tokens)
                if errs:
                    for e in errs:
                        rejection_reasons[e] += 1
                    rejected_rows.append({"id": payload.get("id") or key, "errors": errs})
                else:
                    valid_orpo_raw.append(payload)
            else:
                errs = validate_sft_sample(payload, tokenizer=tokenizer, max_tokens=max_tokens)
                if errs:
                    for e in errs:
                        rejection_reasons[e] += 1
                    rejected_rows.append({"id": payload.get("id") or key, "errors": errs})
                else:
                    valid_sft_raw.append(payload)

    dedup_sft = deduplicate_samples(valid_sft_raw)
    dedup_orpo = deduplicate_samples(valid_orpo_raw)

    sft_out = Path(out_sft_path)
    orpo_out = Path(out_orpo_path)
    rej_out = Path(out_rejected_path)
    for p in (sft_out, orpo_out, rej_out):
        p.parent.mkdir(parents=True, exist_ok=True)

    with sft_out.open("w", encoding="utf-8") as f:
        for sample in dedup_sft:
            bucket = str(sample.get("bucket", "unknown"))
            bucket_counts[bucket] += 1
            rendered = render_gemma4_training_sample(
                sample["messages"],
                tools=_resolve_tools(sample),
                enable_thinking=bool(sample.get("enable_thinking", True)),
            )
            record = {
                "id": sample.get("id"),
                "bucket": bucket,
                "mode": sample["mode"],
                "role": sample.get("role", "coder"),
                "enable_thinking": bool(sample.get("enable_thinking", True)),
                "text": rendered["text"],
                "assistant_spans": rendered["assistant_spans"],
                "messages": sample["messages"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with orpo_out.open("w", encoding="utf-8") as f:
        for pair in dedup_orpo:
            rendered = render_gemma4_orpo_pair(
                pair["prompt_messages"],
                pair["chosen"],
                pair["rejected"],
                tools=_resolve_tools(pair),
                enable_thinking=bool(pair.get("enable_thinking", True)),
            )
            record = {
                "id": pair.get("id"),
                "failure_type": pair["failure_type"],
                "mode": pair.get("mode", "nc_agent"),
                "enable_thinking": bool(pair.get("enable_thinking", True)),
                "prompt": rendered["prompt"],
                "chosen": rendered["chosen"],
                "rejected": rendered["rejected"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with rej_out.open("w", encoding="utf-8") as f:
        for rej in rejected_rows:
            f.write(json.dumps(rej, ensure_ascii=False) + "\n")

    return {
        "total_lines": total_lines,
        "valid_sft_before_dedup": len(valid_sft_raw),
        "kept_sft": len(dedup_sft),
        "valid_orpo_before_dedup": len(valid_orpo_raw),
        "kept_orpo": len(dedup_orpo),
        "rejected_count": len(rejected_rows),
        "sft_bucket_counts": dict(bucket_counts),
        "top_rejection_reasons": dict(rejection_reasons.most_common(10)),
    }

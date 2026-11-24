"""SFT (assistant-only loss masking) and ORPO dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty UTF-8 JSONL lines into a list of dicts."""
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def load_sft_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load pre-rendered Gemma 4 SFT JSONL records containing 'text' and 'assistant_spans'."""
    rows = load_jsonl(path)
    for idx, row in enumerate(rows):
        if "text" not in row or "assistant_spans" not in row:
            raise ValueError(f"SFT row {idx} missing required 'text' or 'assistant_spans'")
    return rows


def load_orpo_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load pre-rendered Gemma 4 ORPO JSONL records containing 'prompt', 'chosen', 'rejected'."""
    rows = load_jsonl(path)
    for idx, row in enumerate(rows):
        if not all(k in row for k in ("prompt", "chosen", "rejected")):
            raise ValueError(f"ORPO row {idx} missing 'prompt'/'chosen'/'rejected'")
    return rows


def split_train_eval(
    records: list[dict[str, Any]],
    eval_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split records into (train, eval) lists."""
    if not records:
        return [], []
    if eval_ratio <= 0.0 or len(records) < 4:
        return list(records), []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    eval_n = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * eval_ratio))))
    return shuffled[eval_n:], shuffled[:eval_n]


def build_assistant_loss_labels(
    text: str,
    assistant_spans: list[tuple[int, int]] | list[list[int]],
    tokenizer: Any,
    max_seq_length: int = 8192,
) -> dict[str, list[int]]:
    """Tokenize rendered Gemma 4 trajectory and mask loss outside assistant_spans with -100."""
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    attention_mask = list(enc.get("attention_mask") or [1] * len(input_ids))

    norm_spans = [(int(s), int(e)) for s, e in assistant_spans if int(e) > int(s)]
    labels: list[int] = []
    unmasked = 0
    for tok_id, (tok_start, tok_end) in zip(input_ids, offsets):
        in_assistant = tok_end > tok_start and any(
            tok_start < sp_end and tok_end > sp_start for sp_start, sp_end in norm_spans
        )
        if in_assistant:
            labels.append(int(tok_id))
            unmasked += 1
        else:
            labels.append(-100)

    if unmasked == 0:
        raise ValueError(
            "All tokens were masked (-100); verify assistant_spans and max_seq_length"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_sft_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 8192,
) -> list[dict[str, list[int]]]:
    """Convert pre-rendered SFT records into masked token dicts ready for Trainer."""
    return [
        build_assistant_loss_labels(
            row["text"],
            row["assistant_spans"],
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        for row in records
    ]


def make_sft_collator(pad_token_id: int):
    """Return a batch collator padding input_ids, attention_mask, and labels (-100)."""

    def _collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch_ids: list[list[int]] = []
        batch_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            batch_ids.append(f["input_ids"] + [pad_token_id] * pad_n)
            batch_mask.append(f["attention_mask"] + [0] * pad_n)
            batch_labels.append(f["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    return _collate

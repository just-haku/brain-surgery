"""Measure exact or pruned-vocab token length distributions (min/p50/p95/max) for SFT & ORPO."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct
from typing import Any

from surgery.bpe_closure import build_merge_structures, parse_merges
from surgery.vocab_filter import collect_mandatory_whitelist, select_hybrid_vocab

_GGUF_VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_CHUNK_RE = re.compile(r"\r?\n+|[^\S\n]+|▁?[A-Za-z0-9_]+|▁?[^\s▁A-Za-z0-9_]+")


def summarize_lengths(lengths: list[int], limit: int) -> dict[str, Any]:
    """Return min, p50, p95, p99, max and overflow stats against a max_seq_length limit."""
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "limit": limit,
            "exceeds_limit": 0,
            "fits_100_pct": True,
        }
    s = sorted(lengths)

    def _pct(q: float) -> int:
        idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
        return int(s[idx])

    over = sum(1 for n in s if n > limit)
    return {
        "count": len(s),
        "min": int(s[0]),
        "p50": _pct(0.50),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
        "max": int(s[-1]),
        "limit": int(limit),
        "exceeds_limit": int(over),
        "fits_100_pct": over == 0,
    }


def _load_gguf_tokenizer_tables(gguf_path: Path) -> tuple[dict[int, str], list[str], set[str]]:
    tokens: list[str] | None = None
    merges: list[str] | None = None
    ttypes: list[int] | None = None
    with gguf_path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"Invalid GGUF magic in {gguf_path}")
        _ver, _nt, n_kv = struct.unpack("<IQQ", f.read(20))

        def _read_str() -> str:
            ln = struct.unpack("<Q", f.read(8))[0]
            return f.read(ln).decode("utf-8", errors="replace")

        for _ in range(n_kv):
            key = _read_str()
            vt = struct.unpack("<I", f.read(4))[0]
            if key == "tokenizer.ggml.tokens":
                _et, ln = struct.unpack("<IQ", f.read(12))
                tokens = [_read_str() for _ in range(ln)]
            elif key == "tokenizer.ggml.merges":
                _et, ln = struct.unpack("<IQ", f.read(12))
                merges = [_read_str() for _ in range(ln)]
            elif key == "tokenizer.ggml.token_type":
                et, ln = struct.unpack("<IQ", f.read(12))
                fmt = "<" + ("i" if et == 4 else "I") * ln
                ttypes = list(struct.unpack(fmt, f.read(_GGUF_VALUE_SIZES[et] * ln)))
            else:
                if vt == 8:
                    f.seek(struct.unpack("<Q", f.read(8))[0], 1)
                elif vt == 9:
                    et, ln = struct.unpack("<IQ", f.read(12))
                    if et == 8:
                        for _ in range(ln):
                            f.seek(struct.unpack("<Q", f.read(8))[0], 1)
                    else:
                        f.seek(_GGUF_VALUE_SIZES[et] * ln, 1)
                else:
                    f.seek(_GGUF_VALUE_SIZES[vt], 1)
            if tokens is not None and merges is not None and ttypes is not None:
                break

    if tokens is None or merges is None:
        raise ValueError(f"Missing tokenizer tables in GGUF {gguf_path}")
    vocab_by_id = {idx: tok for idx, tok in enumerate(tokens)}
    specials = {"[multimodal]"}
    if ttypes is not None:
        for idx, ttype in enumerate(ttypes):
            if ttype in (3, 4):
                specials.add(tokens[idx])
    return vocab_by_id, merges, specials


class Gemma4BpeEncoder:
    """Stdlib-only Gemma 4 BPE + special-token encoder reading HF tokenizer.json or GGUF."""

    def __init__(
        self,
        vocab: dict[str, int],
        merge_pairs: list[tuple[str, str]],
        special_tokens: set[str],
        unk_token: str = "<unk>",
    ) -> None:
        self.vocab = vocab
        self.merge_rank, _ = build_merge_structures(merge_pairs)
        self.unk_token_id = vocab.get(unk_token, 3)
        ordered_specials = sorted(
            [tok for tok in special_tokens if tok in vocab],
            key=len,
            reverse=True,
        )
        self._special_re = (
            re.compile("|".join(re.escape(tok) for tok in ordered_specials))
            if ordered_specials
            else None
        )
        self._bpe_cache: dict[str, list[int]] = {}

    @classmethod
    def from_tokenizer_json(cls, path: str | Path) -> Gemma4BpeEncoder:
        """Load encoder directly from a pruned or unpruned HuggingFace tokenizer.json."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {str(k): int(v) for k, v in data["model"]["vocab"].items()}
        specials: set[str] = set()
        for entry in data.get("added_tokens") or []:
            tok = str(entry["content"])
            vocab[tok] = int(entry["id"])
            specials.add(tok)
        merge_pairs, _ = parse_merges(data["model"]["merges"])
        return cls(vocab=vocab, merge_pairs=merge_pairs, special_tokens=specials)

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str | Path,
        prune_to: int | None = None,
        sample_texts: list[str] | None = None,
    ) -> Gemma4BpeEncoder:
        """Load encoder from GGUF, optionally pruning to `prune_to` (e.g. 65536) via Step 0 filter."""
        vocab_by_id, raw_merges, specials = _load_gguf_tokenizer_tables(Path(gguf_path))
        if prune_to is not None and prune_to < len(vocab_by_id):
            whitelist = collect_mandatory_whitelist(vocab_by_id=vocab_by_id)
            code_counts: Counter[int] = Counter()
            if sample_texts:
                full_enc = cls(
                    vocab={t: i for i, t in vocab_by_id.items()},
                    merge_pairs=parse_merges(raw_merges)[0],
                    special_tokens=specials,
                )
                for txt in sample_texts[:128]:
                    for tid in full_enc.encode(txt):
                        code_counts[tid] += 1
            kept_ids, pruned_merges_raw, _ = select_hybrid_vocab(
                vocab_by_id=vocab_by_id,
                raw_merges=raw_merges,
                whitelist_ids=whitelist,
                code_counts=code_counts,
                target_size=prune_to,
            )
            vocab = {vocab_by_id[oid]: new_id for new_id, oid in enumerate(kept_ids)}
            merge_pairs, _ = parse_merges(pruned_merges_raw)
        else:
            vocab = {tok: tid for tid, tok in vocab_by_id.items()}
            merge_pairs, _ = parse_merges(raw_merges)
        return cls(vocab=vocab, merge_pairs=merge_pairs, special_tokens=specials)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._bpe_cache.get(chunk)
        if cached is not None:
            return cached

        pieces: list[str] = []
        for ch in chunk:
            if ch in self.vocab:
                pieces.append(ch)
            else:
                for b in ch.encode("utf-8"):
                    pieces.append(f"<0x{b:02X}>")

        while len(pieces) > 1:
            best_rank: int | None = None
            best_pair: tuple[str, str] | None = None
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            a, b = best_pair
            merged = a + b
            nxt: list[str] = []
            i = 0
            while i < len(pieces):
                if i + 1 < len(pieces) and pieces[i] == a and pieces[i + 1] == b:
                    nxt.append(merged)
                    i += 2
                else:
                    nxt.append(pieces[i])
                    i += 1
            pieces = nxt

        ids = [self.vocab.get(p, self.unk_token_id) for p in pieces]
        self._bpe_cache[chunk] = ids
        return ids

    def _encode_plain(self, segment: str) -> list[int]:
        if not segment:
            return []
        sp = segment.replace(" ", "▁")
        ids: list[int] = []
        for chunk in _CHUNK_RE.findall(sp):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode full wire-formatted text matching atomic Gemma 4 special tokens + BPE."""
        del add_special_tokens
        if not text:
            return []
        if self._special_re is None:
            return self._encode_plain(text)
        ids: list[int] = []
        pos = 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                ids.extend(self._encode_plain(text[pos : m.start()]))
            ids.append(self.vocab[m.group(0)])
            pos = m.end()
        if pos < len(text):
            ids.extend(self._encode_plain(text[pos:]))
        return ids


def _encode_len_and_unks(tokenizer: Any, text: str) -> tuple[int, int]:
    try:
        out = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        out = tokenizer.encode(text)
    ids = list(out.ids) if hasattr(out, "ids") else list(out)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unks = ids.count(unk_id) if unk_id is not None else 0
    return len(ids), unks


def measure_dataset_token_lengths(
    sft_jsonl: str | Path,
    orpo_jsonl: str | Path,
    *,
    tokenizer: Any | None = None,
    tokenizer_json: str | Path | None = None,
    gguf_path: str | Path | None = None,
    prune_to: int | None = None,
    sft_limit: int = 8192,
    orpo_limit: int = 4096,
    orpo_prompt_limit: int = 2560,
) -> dict[str, Any]:
    """Measure min/p50/p95/max token lengths and <unk> counts across SFT & ORPO JSONLs."""
    sft_rows = [
        json.loads(ln)
        for ln in Path(sft_jsonl).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    orpo_rows = [
        json.loads(ln)
        for ln in Path(orpo_jsonl).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    if tokenizer is not None:
        tok = tokenizer
        source_label = "provided_tokenizer"
    elif tokenizer_json is not None:
        tok = Gemma4BpeEncoder.from_tokenizer_json(tokenizer_json)
        source_label = f"tokenizer_json:{tokenizer_json}"
    elif gguf_path is not None:
        tok = Gemma4BpeEncoder.from_gguf(
            gguf_path,
            prune_to=prune_to,
            sample_texts=[r["text"] for r in sft_rows[:128]],
        )
        source_label = f"gguf:{gguf_path}(prune_to={prune_to})"
    else:
        raise ValueError("Provide one of tokenizer, tokenizer_json, or gguf_path")

    sft_lens: list[int] = []
    orpo_prompt_lens: list[int] = []
    orpo_chosen_lens: list[int] = []
    orpo_rejected_lens: list[int] = []
    unk_count = 0

    for r in sft_rows:
        n, u = _encode_len_and_unks(tok, r["text"])
        sft_lens.append(n)
        unk_count += u

    for p in orpo_rows:
        pn, pu = _encode_len_and_unks(tok, p["prompt"])
        cn, cu = _encode_len_and_unks(tok, p["prompt"] + p["chosen"])
        rn, ru = _encode_len_and_unks(tok, p["prompt"] + p["rejected"])
        orpo_prompt_lens.append(pn)
        orpo_chosen_lens.append(cn)
        orpo_rejected_lens.append(rn)
        unk_count += pu + cu + ru

    sft_stats = summarize_lengths(sft_lens, sft_limit)
    prompt_stats = summarize_lengths(orpo_prompt_lens, orpo_prompt_limit)
    chosen_stats = summarize_lengths(orpo_chosen_lens, orpo_limit)
    rejected_stats = summarize_lengths(orpo_rejected_lens, orpo_limit)

    fits_all = (
        sft_stats["fits_100_pct"]
        and prompt_stats["fits_100_pct"]
        and chosen_stats["fits_100_pct"]
        and rejected_stats["fits_100_pct"]
    )
    return {
        "tokenizer_source": source_label,
        "vocab_size": len(tok.vocab) if hasattr(tok, "vocab") else getattr(tok, "vocab_size", None),
        "unk_count": unk_count,
        "fits_100_pct": fits_all,
        "sft_text": sft_stats,
        "orpo_prompt": prompt_stats,
        "orpo_chosen_total": chosen_stats,
        "orpo_rejected_total": rejected_stats,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit exact Gemma 4 token lengths across SFT & ORPO")
    p.add_argument("--sft", default="data/distill/sft_gemma4.jsonl")
    p.add_argument("--orpo", default="data/distill/orpo_gemma4.jsonl")
    p.add_argument("--tokenizer-dir", default=None)
    p.add_argument("--tokenizer-json", default=None)
    p.add_argument("--gguf", default=None)
    p.add_argument("--prune-to", type=int, default=None)
    p.add_argument("--sft-limit", type=int, default=8192)
    p.add_argument("--orpo-limit", type=int, default=4096)
    p.add_argument("--orpo-prompt-limit", type=int, default=2560)
    p.add_argument("--out-json", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    tok = None
    if ns.tokenizer_dir:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(ns.tokenizer_dir, use_fast=True)

    report = measure_dataset_token_lengths(
        ns.sft,
        ns.orpo,
        tokenizer=tok,
        tokenizer_json=ns.tokenizer_json,
        gguf_path=ns.gguf,
        prune_to=ns.prune_to,
        sft_limit=ns.sft_limit,
        orpo_limit=ns.orpo_limit,
        orpo_prompt_limit=ns.orpo_prompt_limit,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if ns.out_json:
        Path(ns.out_json).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["fits_100_pct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

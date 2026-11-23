"""4-tier hybrid vocabulary filtering (EN + VI + Code + ASCII emoticons)."""

from __future__ import annotations

import math
import re
from typing import Any

from .bpe_closure import (
    close_and_align_vocab,
    format_merges,
    parse_merges,
    verify_merge_closure,
)

# Precomposed Vietnamese letters (NFC) across all 5 tone marks + d-stroke.
VI_CHARS: frozenset[str] = frozenset(
    "ăâđêôơư"
    "àáạảãằắặẳẵầấậẩẫ"
    "èéẹẻẽềếệểễ"
    "ìíịỉĩ"
    "òóọỏõồốộổỗờớợởỡ"
    "ùúụủũừứựửữ"
    "ỳýỵỷỹ"
    "ĂÂĐÊÔƠƯ"
    "ÀÁẠẢÃẰẮẶẲẴẦẤẬẨẪ"
    "ÈÉẸẺẼỀẾỆỂỄ"
    "ÌÍỊỈĨ"
    "ÒÓỌỎÕỒỐỘỔỖỜỚỢỞỠ"
    "ÙÚỤỦŨỪỨỰỬỮ"
    "ỲÝỴỶỸ"
)

ASCII_EMOTICON_SEEDS: tuple[str, ...] = (
    ":)",
    ":)))",
    "._.",
    ".-.",
    "=))",
    ":(",
    ":D",
    ";)",
    ":P",
    "^_^",
)

CODE_SYMBOL_SEEDS: tuple[str, ...] = (
    "==",
    "!=",
    "<=",
    ">=",
    "->",
    "=>",
    "::",
    "//",
    "/*",
    "*/",
    "**",
    "&&",
    "||",
    "<<",
    ">>",
    "+=",
    "-=",
    "```",
    '"""',
)

CRITICAL_TOKEN_STRINGS: tuple[str, ...] = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<mask>",
    "[multimodal]",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    "<|think|>",
    "<|channel>",
    "<channel|>",
    "<|turn>",
    "<turn|>",
    "<|image>",
    "<|audio>",
    "<|image|>",
    "<|audio|>",
    "<image|>",
    "<audio|>",
    "<|video|>",
    "<bbox>",
    "</bbox>",
)

_UNUSED_RE = re.compile(r"^<unused\d+>$")
_SPECIAL_TAG_RE = re.compile(r"<\|[^>\s]+\|?>|<[^|\s]+\|>")
_SP_SPACE = "\u2581"  # SentencePiece word boundary marker '▁'


def is_unused_token(token: str) -> bool:
    """Return True for Gemma reserved <unusedN> slots."""
    return bool(_UNUSED_RE.match(token))


def is_emoji_or_banned_symbol(token: str) -> bool:
    """Return True if token contains any Unicode emoji, pictograph, ZWJ, or variation selector."""
    for ch in token:
        cp = ord(ch)
        if (
            0x2600 <= cp <= 0x26FF
            or 0x2700 <= cp <= 0x27BF
            or 0x1F1E0 <= cp <= 0x1F1FF
            or 0x1F300 <= cp <= 0x1FAFF
            or cp == 0x200D
            or 0xFE00 <= cp <= 0xFE0F
        ):
            return True
    return False


def is_ascii_vi_code_candidate(token: str) -> bool:
    """Allow only ASCII, whitespace/indent, SentencePiece '▁', and Vietnamese letters."""
    if not token or is_unused_token(token) or is_emoji_or_banned_symbol(token):
        return False
    for ch in token:
        cp = ord(ch)
        if cp in (9, 10, 13) or 0x20 <= cp <= 0x7E or ch == _SP_SPACE or ch in VI_CHARS:
            continue
        return False
    return True


def has_vi_diacritic(token: str) -> bool:
    """Return True if token contains any Vietnamese diacritic or đ/Đ."""
    return any(ch in VI_CHARS for ch in token)


def _extract_ids_recursive(obj: Any, out: set[int]) -> None:
    if not isinstance(obj, dict):
        return
    for key, val in obj.items():
        if isinstance(val, dict):
            _extract_ids_recursive(val, out)
        elif key.endswith(("_token_id", "_token_index", "_token_ids")):
            if isinstance(val, int):
                out.add(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, int):
                        out.add(item)


def collect_mandatory_whitelist(
    vocab_by_id: dict[int, str],
    config: dict[str, Any] | None = None,
    tokenizer_config: dict[str, Any] | None = None,
    generation_config: dict[str, Any] | None = None,
    tokenizer_json: dict[str, Any] | None = None,
    chat_template: str | None = None,
) -> set[int]:
    """Collect all mandatory control, byte-fallback, multimodal, ASCII, and seed tokens."""
    id_by_tok = {tok: tid for tid, tok in vocab_by_id.items()}
    keep: set[int] = set()

    # Preserve non-unused system, tool, think, turn, indent, HTML, and 256 byte-fallback tokens (0..493).
    for tid in range(494):
        tok = vocab_by_id.get(tid)
        if tok is not None and not is_unused_token(tok):
            keep.add(tid)

    # Preserve all critical named tokens (including ID 5 [multimodal] & end-of-table multimodal IDs).
    for tok in CRITICAL_TOKEN_STRINGS:
        if tok in id_by_tok:
            keep.add(id_by_tok[tok])

    # Preserve added_tokens from tokenizer.json and added_tokens_decoder from tokenizer_config.json.
    if tokenizer_json:
        for item in tokenizer_json.get("added_tokens") or []:
            tid = item.get("id")
            tok = item.get("content", "")
            if isinstance(tid, int) and tid in vocab_by_id and not is_unused_token(tok):
                keep.add(tid)

    if tokenizer_config:
        decoder = tokenizer_config.get("added_tokens_decoder") or {}
        for k, item in decoder.items():
            try:
                tid = int(k)
            except ValueError:
                continue
            tok = (item or {}).get("content", "")
            if tid in vocab_by_id and not is_unused_token(tok):
                keep.add(tid)
        tpl = chat_template or tokenizer_config.get("chat_template")
        if isinstance(tpl, str):
            chat_template = tpl

    # Preserve every token ID referenced in config.json (root + text_config) and generation_config.json.
    if config:
        _extract_ids_recursive(config, keep)
    if generation_config:
        _extract_ids_recursive(generation_config, keep)

    # Auto-whitelist special tags appearing in chat_template.
    if isinstance(chat_template, str):
        for tag in _SPECIAL_TAG_RE.findall(chat_template):
            if tag in id_by_tok:
                keep.add(id_by_tok[tag])

    # Guarantee single-char printable ASCII, '▁'+ASCII, single-char VI, '▁'+VI, and emoticon/code seeds.
    for cp in range(0x20, 0x7F):
        ch = chr(cp)
        for cand in (ch, _SP_SPACE + ch):
            if cand in id_by_tok:
                keep.add(id_by_tok[cand])

    for ch in VI_CHARS:
        for cand in (ch, _SP_SPACE + ch):
            if cand in id_by_tok:
                keep.add(id_by_tok[cand])

    for seed in (*ASCII_EMOTICON_SEEDS, *CODE_SYMBOL_SEEDS):
        for cand in (seed, _SP_SPACE + seed):
            if cand in id_by_tok:
                keep.add(id_by_tok[cand])

    # Keep only IDs that actually exist in vocab_by_id.
    return {tid for tid in keep if tid in vocab_by_id}


def _normalized_log_counts(counts: dict[int, int] | None) -> dict[int, float]:
    if not counts:
        return {}
    logs = {tid: math.log1p(c) for tid, c in counts.items() if c > 0}
    max_val = max(logs.values(), default=0.0)
    if max_val <= 0.0:
        return {}
    return {tid: val / max_val for tid, val in logs.items()}


def select_hybrid_vocab(
    vocab_by_id: dict[int, str],
    raw_merges: list,
    whitelist_ids: set[int],
    en_counts: dict[int, int] | None = None,
    vi_counts: dict[int, int] | None = None,
    code_counts: dict[int, int] | None = None,
    target_size: int = 65536,
    vi_floor: int = 4000,
    code_floor: int = 16000,
) -> tuple[list[int], list, dict[str, Any]]:
    """Rank EN/VI/Code candidates, close BPE merges, align to 128, and verify zero gaps."""
    if target_size % 128 != 0:
        raise ValueError(f"target_size must be divisible by 128, got {target_size}")

    merge_pairs, is_list_format = parse_merges(raw_merges)
    vocab_set = set(vocab_by_id.values())
    id_by_tok = {tok: tid for tid, tok in vocab_by_id.items()}

    # Precompute stats for audit reporting.
    unused_total = sum(1 for tok in vocab_by_id.values() if is_unused_token(tok))
    emoji_total = sum(1 for tok in vocab_by_id.values() if is_emoji_or_banned_symbol(tok))

    eligible_ids = [
        tid
        for tid, tok in vocab_by_id.items()
        if tid not in whitelist_ids and is_ascii_vi_code_candidate(tok)
    ]

    en_norm = _normalized_log_counts(en_counts)
    vi_norm = _normalized_log_counts(vi_counts)
    code_norm = _normalized_log_counts(code_counts)

    vi_counts_safe = vi_counts or {}
    code_counts_safe = code_counts or {}

    # 1. Vietnamese diacritic floor quota (prevents English from crowding out VI syllables).
    vi_candidates = [tid for tid in eligible_ids if has_vi_diacritic(vocab_by_id[tid])]
    vi_candidates.sort(key=lambda tid: (vi_counts_safe.get(tid, 0), -tid), reverse=True)
    primary_ids: list[int] = vi_candidates[:vi_floor]
    seen_primary = set(primary_ids)

    # 2. Code floor quota.
    code_candidates = [
        tid
        for tid in eligible_ids
        if tid not in seen_primary and code_counts_safe.get(tid, 0) > 0
    ]
    code_candidates.sort(key=lambda tid: (code_counts_safe[tid], -tid), reverse=True)
    for tid in code_candidates[:code_floor]:
        primary_ids.append(tid)
        seen_primary.add(tid)

    # 3. Combined weighted ranking across remaining eligible EN + VI + Code tokens.
    remaining_ids = [tid for tid in eligible_ids if tid not in seen_primary]
    remaining_ids.sort(
        key=lambda tid: (
            0.40 * en_norm.get(tid, 0.0)
            + 0.30 * vi_norm.get(tid, 0.0)
            + 0.30 * code_norm.get(tid, 0.0),
            -tid,
        ),
        reverse=True,
    )

    whitelist_tokens = {vocab_by_id[tid] for tid in whitelist_ids}
    primary_tokens = [vocab_by_id[tid] for tid in primary_ids]
    fallback_tokens = [vocab_by_id[tid] for tid in remaining_ids]

    kept_tokens, kept_merges = close_and_align_vocab(
        whitelist_tokens=whitelist_tokens,
        primary_tokens=primary_tokens,
        fallback_ranked_tokens=fallback_tokens,
        merge_pairs=merge_pairs,
        vocab_set=vocab_set,
        target_size=target_size,
    )

    orig_merge_targets = {a + b for a, b in merge_pairs}
    unreachable = verify_merge_closure(kept_tokens, kept_merges, orig_merge_targets)
    if unreachable:
        raise RuntimeError(
            f"BPE merge-closure failed for {len(unreachable)} tokens, sample={unreachable[:5]}"
        )

    kept_old_ids = sorted(id_by_tok[tok] for tok in kept_tokens)
    kept_non_wl = [tid for tid in kept_old_ids if tid not in whitelist_ids]
    kept_vi = sum(1 for tid in kept_old_ids if has_vi_diacritic(vocab_by_id[tid]))

    # Sanity check: ensure no banned emoji or <unused> leaked unless explicitly whitelisted.
    leaked_emoji = [vocab_by_id[tid] for tid in kept_non_wl if is_emoji_or_banned_symbol(vocab_by_id[tid])]
    leaked_unused = [vocab_by_id[tid] for tid in kept_old_ids if is_unused_token(vocab_by_id[tid])]
    if leaked_emoji or leaked_unused:
        raise AssertionError(
            f"Blacklist leak detected: emoji={leaked_emoji[:5]}, unused={leaked_unused[:5]}"
        )

    report = {
        "orig_vocab_size": len(vocab_by_id),
        "new_vocab_size": len(kept_old_ids),
        "reduction_pct": round((1.0 - len(kept_old_ids) / len(vocab_by_id)) * 100.0, 2),
        "whitelist_count": len(whitelist_ids),
        "eligible_pool_count": len(eligible_ids),
        "kept_vi_diacritic_count": kept_vi,
        "pruned_unused_total": unused_total,
        "pruned_emoji_total": emoji_total,
        "merges_before": len(merge_pairs),
        "merges_after": len(kept_merges),
        "unreachable_merge_tokens": len(unreachable),
    }
    return kept_old_ids, format_merges(kept_merges, is_list_format), report

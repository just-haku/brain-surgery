"""BPE merge-closure and multiple-of-128 vocabulary alignment."""

from __future__ import annotations

from collections.abc import Iterable


def parse_merges(raw_merges: list) -> tuple[list[tuple[str, str]], bool]:
    """Parse HF tokenizer.json BPE merges from either ['a', 'b'] or 'a b' format."""
    if not raw_merges:
        return [], False
    first = raw_merges[0]
    is_list_format = isinstance(first, (list, tuple))
    pairs: list[tuple[str, str]] = []
    for idx, item in enumerate(raw_merges):
        if isinstance(item, (list, tuple)):
            if len(item) != 2:
                raise ValueError(f"Invalid list merge at index {idx}: {item!r}")
            pairs.append((str(item[0]), str(item[1])))
        elif isinstance(item, str):
            parts = item.split(" ")
            if len(parts) != 2:
                raise ValueError(f"Cannot unambiguously split string merge at index {idx}: {item!r}")
            pairs.append((parts[0], parts[1]))
        else:
            raise TypeError(f"Unsupported merge item type at index {idx}: {type(item)}")
    return pairs, is_list_format


def format_merges(pairs: Iterable[tuple[str, str]], is_list_format: bool) -> list:
    """Serialize merge pairs back to the exact source tokenizer.json format."""
    if is_list_format:
        return [[a, b] for a, b in pairs]
    return [f"{a} {b}" for a, b in pairs]


def build_merge_structures(
    pairs: list[tuple[str, str]],
) -> tuple[dict[tuple[str, str], int], dict[str, tuple[str, str]]]:
    """Build rank lookup for greedy BPE and first-parent map for fallback closure."""
    merge_rank: dict[tuple[str, str], int] = {}
    first_parent: dict[str, tuple[str, str]] = {}
    for rank, (a, b) in enumerate(pairs):
        if (a, b) not in merge_rank:
            merge_rank[(a, b)] = rank
        ab = a + b
        if ab not in first_parent:
            first_parent[ab] = (a, b)
    return merge_rank, first_parent


def initial_bpe_symbols(token: str, vocab_set: set[str]) -> list[str]:
    """Split a merged token into its base character (or byte-fallback) symbols."""
    pieces: list[str] = []
    for ch in token:
        if ch in vocab_set:
            pieces.append(ch)
        else:
            for b in ch.encode("utf-8"):
                pieces.append(f"<0x{b:02X}>")
    return pieces


def _simulate_greedy_bpe(
    pieces: list[str],
    merge_rank: dict[tuple[str, str], int],
    target: str,
    trace: set[str],
) -> list[str]:
    """Run greedy lowest-rank BPE on pieces, recording intermediate tokens in trace."""
    while len(pieces) > 1:
        best_rank = None
        best_pair = None
        for i in range(len(pieces) - 1):
            pair = (pieces[i], pieces[i + 1])
            r = merge_rank.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_pair = pair
        if best_pair is None:
            break
        a, b = best_pair
        merged = a + b
        trace.add(merged)
        new_pieces: list[str] = []
        i = 0
        while i < len(pieces):
            if i + 1 < len(pieces) and pieces[i] == a and pieces[i + 1] == b:
                new_pieces.append(merged)
                i += 2
            else:
                new_pieces.append(pieces[i])
                i += 1
        pieces = new_pieces
        if len(pieces) == 1 and pieces[0] == target:
            break
    return pieces


def _fallback_close_ancestors(
    token: str,
    first_parent: dict[str, tuple[str, str]],
    out: set[str],
    visiting: set[str],
) -> None:
    if token in out or token in visiting:
        return
    visiting.add(token)
    parent = first_parent.get(token)
    if parent is not None:
        a, b = parent
        _fallback_close_ancestors(a, first_parent, out, visiting)
        _fallback_close_ancestors(b, first_parent, out, visiting)
    visiting.remove(token)
    out.add(token)


def canonical_bpe_ancestors(
    token: str,
    merge_rank: dict[tuple[str, str], int],
    first_parent: dict[str, tuple[str, str]],
    vocab_set: set[str],
    memo: dict[str, set[str]],
) -> set[str]:
    """Return all ancestor tokens needed so greedy BPE builds `token` losslessly."""
    cached = memo.get(token)
    if cached is not None:
        return cached
    if token not in first_parent:
        res = {token}
        memo[token] = res
        return res

    pieces = initial_bpe_symbols(token, vocab_set)
    ancestors: set[str] = set(pieces)
    ancestors.add(token)
    final_pieces = _simulate_greedy_bpe(pieces, merge_rank, token, ancestors)
    if final_pieces != [token]:
        _fallback_close_ancestors(token, first_parent, ancestors, set())

    memo[token] = ancestors
    return ancestors


def close_and_align_vocab(
    whitelist_tokens: set[str],
    primary_tokens: list[str],
    fallback_ranked_tokens: list[str],
    merge_pairs: list[tuple[str, str]],
    vocab_set: set[str],
    target_size: int,
) -> tuple[set[str], list[tuple[str, str]]]:
    """Close whitelist and fill greedily up to exact `target_size` (multiple of 128)."""
    if target_size % 128 != 0:
        raise ValueError(f"target_size must be a multiple of 128, got {target_size}")

    merge_rank, first_parent = build_merge_structures(merge_pairs)
    memo: dict[str, set[str]] = {}

    kept: set[str] = set()
    for tok in whitelist_tokens:
        if tok in vocab_set:
            kept.update(canonical_bpe_ancestors(tok, merge_rank, first_parent, vocab_set, memo))

    if len(kept) > target_size:
        raise ValueError(
            f"Closed mandatory whitelist ({len(kept)} tokens) exceeds target_size={target_size}"
        )

    for pool in (primary_tokens, fallback_ranked_tokens):
        for tok in pool:
            if len(kept) >= target_size:
                break
            if tok in kept or tok not in vocab_set:
                continue
            needed = canonical_bpe_ancestors(tok, merge_rank, first_parent, vocab_set, memo)
            delta = needed - kept
            if len(kept) + len(delta) <= target_size:
                kept.update(delta)
        if len(kept) >= target_size:
            break

    # Prune merges preserving exact original relative rank order.
    kept_merges = [
        (a, b)
        for (a, b) in merge_pairs
        if a in kept and b in kept and (a + b) in kept
    ]
    return kept, kept_merges


def verify_merge_closure(
    kept_tokens: set[str],
    kept_merges: list[tuple[str, str]],
    orig_merge_targets: set[str],
) -> list[str]:
    """Verify every surviving merged token remains reachable under `kept_merges`."""
    pruned_rank, _ = build_merge_structures(kept_merges)
    direct_pairs: dict[str, list[tuple[str, str]]] = {}
    for a, b in kept_merges:
        direct_pairs.setdefault(a + b, []).append((a, b))

    unreachable: list[str] = []
    for tok in sorted(kept_tokens):
        if tok not in orig_merge_targets:
            continue
        pieces = initial_bpe_symbols(tok, kept_tokens)
        trace: set[str] = set()
        final_pieces = _simulate_greedy_bpe(pieces, pruned_rank, tok, trace)
        if final_pieces == [tok]:
            continue
        # Fallback check for tokens whose original SentencePiece merges already stalled.
        if any(a in kept_tokens and b in kept_tokens for a, b in direct_pairs.get(tok, ())):
            continue
        unreachable.append(tok)
    return unreachable

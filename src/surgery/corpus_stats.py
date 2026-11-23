"""Streaming EN + VI + Code corpus token frequency counter for Colab."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

# Built-in seed texts guarantee baseline counts for Vietnamese technical terms, code, and ASCII emoticons.
SEED_TEXTS: dict[str, tuple[str, ...]] = {
    "en": (
        "Async Python FastAPI endpoint with Pydantic validation, dependency injection, and pytest fixtures.",
        "CUDA memory management, KV cache quantization, tensor core alignment, and gradient checkpointing.",
        "Debug segmentation faults, race conditions, deadlocks, memory leaks, and SQL query plans.",
        "Happy to help with that refactoring :) looks clean and minimal ^_^",
    ),
    "vi": (
        "Phẫu thuật từ điển cho mô hình ngôn ngữ nhỏ giúp giảm mạnh bộ nhớ VRAM và tăng tốc độ suy luận cục bộ.",
        "Giải thích thuật toán tìm kiếm nhị phân, quy hoạch động, hàng đợi ưu tiên và đồ thị có hướng bằng tiếng Việt.",
        "Kiểm tra lỗi tràn bộ nhớ, con trỏ rỗng, bất đồng bộ, chỉ mục cơ sở dữ liệu và tối ưu truy vấn SQL.",
        "Chạy ổn áp rồi đó bro :))) nhìn gọn gàng sạch sẽ hẳn ra =)) không bị lỗi vặt nữa ._.",
    ),
    "code": (
        "def binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
        "#include <vector>\n#include <iostream>\nint main() {\n    std::vector<int> v{1, 2, 3};\n    for (auto x : v) std::cout << x << '\\n';\n    return 0;\n}\n",
        "export async function fetchJson<T>(url: string): Promise<T> {\n  const res = await fetch(url);\n  if (!res.ok) throw new Error(`HTTP ${res.status}`);\n  return (await res.json()) as T;\n}\n",
        "SELECT u.id, COUNT(o.id) AS total FROM users u LEFT JOIN orders o ON o.user_id = u.id GROUP BY u.id HAVING COUNT(o.id) > 0;\n",
        "<!DOCTYPE html><html><head><style>.box{display:flex;gap:8px}</style></head><body><div class=\"box\"></div></body></html>\n",
    ),
}

DEFAULT_CORPUS_SOURCES: dict[str, list[dict[str, Any]]] = {
    "en": [
        {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "text_key": "text"},
        {"path": "wikimedia/wikipedia", "name": "20231101.en", "split": "train", "text_key": "text"},
    ],
    "vi": [
        {"path": "wikimedia/wikipedia", "name": "20231101.vi", "split": "train", "text_key": "text"},
        {"path": "HuggingFaceFW/fineweb-2", "name": "vie_Latn", "split": "train", "text_key": "text"},
    ],
    "code": [
        {"path": "bigcode/the-stack-smol", "name": None, "data_dir": "data/python", "split": "train", "text_key": "content"},
        {"path": "bigcode/the-stack-smol", "name": None, "data_dir": "data/cpp", "split": "train", "text_key": "content"},
        {"path": "bigcode/the-stack-smol", "name": None, "data_dir": "data/javascript", "split": "train", "text_key": "content"},
        {"path": "bigcode/the-stack-smol", "name": None, "data_dir": "data/html", "split": "train", "text_key": "content"},
        {"path": "bigcode/the-stack-smol", "name": None, "data_dir": "data/css", "split": "train", "text_key": "content"},
        {"path": "bigcode/the-stack-smol", "name": None, "data_dir": "data/sql", "split": "train", "text_key": "content"},
        {"path": "iamtarun/python_code_instructions_18k_alpaca", "name": None, "split": "train", "text_key": "output"},
    ],
}


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    """Support both HuggingFace PreTrainedTokenizer and tokenizers.Tokenizer."""
    try:
        out = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        out = tokenizer.encode(text)
    if hasattr(out, "ids"):
        return list(out.ids)
    return list(out)


def count_corpus_tokens(
    tokenizer: Any,
    max_docs_per_lang: dict[str, int] | None = None,
    cache_path: str | Path | None = None,
    sources: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, dict[int, int]]:
    """Stream public EN/VI/Code slices and count token occurrences with JSON caching."""
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        print(f"Loaded cached corpus token counts from {cache}")
        return {
            lang: {int(tid): int(cnt) for tid, cnt in mapping.items()}
            for lang, mapping in raw.items()
        }

    from datasets import load_dataset

    budgets = {"en": 4000, "vi": 5000, "code": 4500}
    if max_docs_per_lang:
        budgets.update(max_docs_per_lang)
    src_map = sources or DEFAULT_CORPUS_SOURCES

    result: dict[str, dict[int, int]] = {}
    for lang in ("en", "vi", "code"):
        counter: Counter[int] = Counter()
        for snippet in SEED_TEXTS.get(lang, ()):
            counter.update(_encode_ids(tokenizer, snippet))

        target_docs = budgets.get(lang, 3000)
        lang_sources = src_map.get(lang, [])
        per_src_quota = max(1, target_docs // max(1, len(lang_sources)))
        total_docs = 0

        for spec in lang_sources:
            if total_docs >= target_docs:
                break
            kwargs: dict[str, Any] = {
                "path": spec["path"],
                "split": spec.get("split", "train"),
                "streaming": True,
            }
            if spec.get("name"):
                kwargs["name"] = spec["name"]
            if spec.get("data_dir"):
                kwargs["data_dir"] = spec["data_dir"]

            try:
                ds = load_dataset(**kwargs)
            except Exception as exc:  # noqa: BLE001 - network/dataset schema fallback in Colab
                print(f"Skipping corpus source {spec['path']} ({spec.get('data_dir') or spec.get('name')}): {exc}")
                continue

            text_key = spec.get("text_key", "text")
            taken = 0
            for row in ds:
                text = row.get(text_key)
                if not isinstance(text, str) or not text:
                    continue
                counter.update(_encode_ids(tokenizer, text[:4000]))
                taken += 1
                total_docs += 1
                if taken >= per_src_quota or total_docs >= target_docs:
                    break
            print(f"Corpus [{lang}] <- {spec['path']} ({spec.get('data_dir') or spec.get('name') or 'default'}): {taken} docs")

        result[lang] = dict(counter)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            lang: {str(tid): cnt for tid, cnt in counts.items()}
            for lang, counts in result.items()
        }
        cache.write_text(json.dumps(serializable), encoding="utf-8")
        print(f"Saved corpus token counts cache to {cache}")

    return result

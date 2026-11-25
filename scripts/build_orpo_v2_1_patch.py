"""Generate & strictly validate 120 on-policy ORPO pairs for Micro-ORPO v2.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from distill.nc_schema import build_nc_system_prompt, build_nc_user_prompt, format_repo_map_file
from distill.validator import convert_and_validate_openai_jsonl
from eval.metrics import LlamaServerSession, stream_chat_completion
from eval.probes import extract_code_block
from eval.suite import DEFAULT_SERVER_EXE


def _run_code_harness(lang: str, code: str, harness: str) -> tuple[bool, str]:
    if not code.strip():
        return False, "empty code"
    with tempfile.TemporaryDirectory(prefix="v21_patch_") as tmp:
        tdir = Path(tmp)
        try:
            if lang == "python":
                src = tdir / "sol.py"
                src.write_text(f"{code}\n\n{harness}\n", encoding="utf-8")
                p = subprocess.run([sys.executable, str(src)], capture_output=True, text=True, timeout=15)
                return p.returncode == 0 and "PASS" in p.stdout, (p.stderr or p.stdout)[-240:]
            if lang == "rust":
                src = tdir / "sol.rs"
                exe = tdir / ("sol.exe" if sys.platform == "win32" else "sol")
                src.write_text(f"{code}\n\n{harness}\n", encoding="utf-8")
                c = subprocess.run(["rustc", "--edition", "2021", str(src), "-o", str(exe)], capture_output=True, text=True, timeout=30)
                if c.returncode != 0:
                    return False, c.stderr[-240:]
                r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=15)
                return r.returncode == 0 and "PASS" in r.stdout, (r.stderr or r.stdout)[-240:]
            if lang == "javascript":
                src = tdir / "sol.js"
                src.write_text(f"{code}\n\n{harness}\n", encoding="utf-8")
                p = subprocess.run(["node", str(src)], capture_output=True, text=True, timeout=15)
                return p.returncode == 0 and "PASS" in p.stdout, (p.stderr or p.stdout)[-240:]
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
    return False, f"unsupported lang {lang}"


# ---------------------------------------------------------------------------
# 50 Real Executed Coding Tasks (30 Python, 10 Rust, 10 JavaScript)
# ---------------------------------------------------------------------------

CODING_TASKS: list[dict[str, str]] = [
    # --- 30 Python ---
    {
        "id": "py_01_dijkstra", "lang": "python",
        "prompt": "Write a Python function `dijkstra(n: int, edges: list[tuple[int, int, int]], src: int) -> list[int]` returning shortest distances from `src` (-1 if unreachable). Output ONLY one ```python``` block.",
        "ref": "import heapq\ndef dijkstra(n: int, edges: list[tuple[int, int, int]], src: int) -> list[int]:\n    adj = [[] for _ in range(n)]\n    for u, v, w in edges:\n        adj[u].append((v, w))\n    dist = [-1] * n\n    dist[src] = 0\n    pq = [(0, src)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in adj[u]:\n            nd = d + w\n            if dist[v] == -1 or nd < dist[v]:\n                dist[v] = nd\n                heapq.heappush(pq, (nd, v))\n    return dist",
        "bad": "import heapqueue\ndef dijkstra(n, edges, src):\n    return [0] * n",
        "harness": "assert dijkstra(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1)], 0) == [0,3,1,4]\nassert dijkstra(3, [(0,1,5)], 0) == [0,5,-1]\nprint('PASS')",
    },
    {
        "id": "py_02_lru_cache", "lang": "python",
        "prompt": "Implement `class LRUCache:` in Python using `OrderedDict` with `__init__(self, capacity: int)`, `get(self, key: int) -> int`, and `put(self, key: int, value: int) -> None`. Output ONLY one ```python``` block.",
        "ref": "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity: int) -> None:\n        self.cap = capacity\n        self.od: OrderedDict[int, int] = OrderedDict()\n    def get(self, key: int) -> int:\n        if key not in self.od:\n            return -1\n        self.od.move_to_end(key)\n        return self.od[key]\n    def put(self, key: int, value: int) -> None:\n        if key in self.od:\n            self.od.move_to_end(key)\n        self.od[key] = value\n        if len(self.od) > self.cap:\n            self.od.popitem(last=False)",
        "bad": "class LRUCache:\n    class _Node:\n        def __init__(self, k, v):\n            self.k = k\n    def __init__(self, capacity: int):\n        self.head = self._Node(0, 0)\n    def get(self, key: int) -> int:\n        return self.head.k\n    def put(self, key: int, value: int) -> None:\n        pass",
        "harness": "c = LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(3,30); assert c.get(3)==30; print('PASS')",
    },
    {
        "id": "py_03_dijkstra_k_stops", "lang": "python",
        "prompt": "Write `cheapest_within_k(n: int, flights: list[tuple[int, int, int]], src: int, dst: int, k: int) -> int` returning cheapest price using at most `k` stops (-1 if none). Output ONLY one ```python``` block.",
        "ref": "def cheapest_within_k(n: int, flights: list[tuple[int, int, int]], src: int, dst: int, k: int) -> int:\n    inf = 10**18\n    dist = [inf] * n\n    dist[src] = 0\n    for _ in range(k + 1):\n        nxt = dist[:]\n        for u, v, w in flights:\n            if dist[u] + w < nxt[v]:\n                nxt[v] = dist[u] + w\n        dist = nxt\n    return -1 if dist[dst] >= inf else dist[dst]",
        "bad": "import heapq\ndef cheapest_within_k(n, flights, src, dst, k):\n    return -1",
        "harness": "assert cheapest_within_k(4, [(0,1,100),(1,2,100),(2,0,100),(1,3,600),(2,3,200)], 0, 3, 1) == 700\nassert cheapest_within_k(3, [(0,1,100),(1,2,100),(0,2,500)], 0, 2, 0) == 500\nprint('PASS')",
    },
    {
        "id": "py_04_top_k_min", "lang": "python",
        "prompt": "Write `top_k_smallest(nums: list[int], k: int) -> list[int]` using `heapq` returning the `k` smallest values sorted ascending. Output ONLY one ```python``` block.",
        "ref": "import heapq\ndef top_k_smallest(nums: list[int], k: int) -> list[int]:\n    if k <= 0:\n        return []\n    return heapq.nsmallest(k, nums)",
        "bad": "import heapqueue\ndef top_k_smallest(nums, k):\n    return nums[:k]",
        "harness": "assert top_k_smallest([9,1,5,3,7,2], 3) == [1,2,3]\nassert top_k_smallest([1,2], 0) == []\nprint('PASS')",
    },
    {
        "id": "py_05_kth_largest", "lang": "python",
        "prompt": "Write `kth_largest(nums: list[int], k: int) -> int` using a min-heap of size `k` in Python. Output ONLY one ```python``` block.",
        "ref": "import heapq\ndef kth_largest(nums: list[int], k: int) -> int:\n    heap = nums[:k]\n    heapq.heapify(heap)\n    for x in nums[k:]:\n        if x > heap[0]:\n            heapq.heapreplace(heap, x)\n    return heap[0]",
        "bad": "def kth_largest(nums, k):\n    return sorted(nums)[k]",
        "harness": "assert kth_largest([3,2,1,5,6,4], 2) == 5\nassert kth_largest([3,2,3,1,2,4,5,5,6], 4) == 4\nprint('PASS')",
    },
    {
        "id": "py_06_coin_change", "lang": "python",
        "prompt": "Write `coin_change(coins: list[int], amount: int) -> int` returning fewest coins to sum to `amount` (-1 if impossible). Output ONLY one ```python``` block.",
        "ref": "def coin_change(coins: list[int], amount: int) -> int:\n    dp = [amount + 1] * (amount + 1)\n    dp[0] = 0\n    for a in range(1, amount + 1):\n        for c in coins:\n            if c <= a and dp[a - c] + 1 < dp[a]:\n                dp[a] = dp[a - c] + 1\n    return -1 if dp[amount] > amount else dp[amount]",
        "bad": "def coin_change(coins, amount):\n    return sum(coins) - amount",
        "harness": "assert coin_change([1,2,5], 11) == 3\nassert coin_change([2], 3) == -1\nassert coin_change([1], 0) == 0\nprint('PASS')",
    },
    {
        "id": "py_07_knapsack_01", "lang": "python",
        "prompt": "Write `knapsack_01(weights: list[int], values: list[int], cap: int) -> int` for 0/1 knapsack max value. Output ONLY one ```python``` block.",
        "ref": "def knapsack_01(weights: list[int], values: list[int], cap: int) -> int:\n    dp = [0] * (cap + 1)\n    for w, v in zip(weights, values):\n        for c in range(cap, w - 1, -1):\n            if dp[c - w] + v > dp[c]:\n                dp[c] = dp[c - w] + v\n    return dp[cap]",
        "bad": "def knapsack_01(weights, values, cap):\n    return max(values) if weights else 0",
        "harness": "assert knapsack_01([1,3,4,5], [1,4,5,7], 7) == 9\nassert knapsack_01([2,3], [4,5], 1) == 0\nprint('PASS')",
    },
    {
        "id": "py_08_lis_len", "lang": "python",
        "prompt": "Write `lis_length(nums: list[int]) -> int` computing length of longest strictly increasing subsequence in O(n log n) with `bisect`. Output ONLY one ```python``` block.",
        "ref": "from bisect import bisect_left\ndef lis_length(nums: list[int]) -> int:\n    tails: list[int] = []\n    for x in nums:\n        i = bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)",
        "bad": "def lis_length(nums):\n    return len(set(nums))",
        "harness": "assert lis_length([10,9,2,5,3,7,101,18]) == 4\nassert lis_length([7,7,7]) == 1\nprint('PASS')",
    },
    {
        "id": "py_09_edit_distance", "lang": "python",
        "prompt": "Write `edit_distance(a: str, b: str) -> int` returning Levenshtein distance between `a` and `b`. Output ONLY one ```python``` block.",
        "ref": "def edit_distance(a: str, b: str) -> int:\n    dp = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        nxt = [i] + [0] * len(b)\n        for j, cb in enumerate(b, 1):\n            nxt[j] = dp[j - 1] if ca == cb else 1 + min(dp[j - 1], dp[j], nxt[j - 1])\n        dp = nxt\n    return dp[-1]",
        "bad": "def edit_distance(a, b):\n    return abs(len(a) - len(b))",
        "harness": "assert edit_distance('horse', 'ros') == 3\nassert edit_distance('intention', 'execution') == 5\nprint('PASS')",
    },
    {
        "id": "py_10_unique_paths", "lang": "python",
        "prompt": "Write `unique_paths_grid(grid: list[list[int]]) -> int` counting paths top-left to bottom-right avoiding `1` obstacles. Output ONLY one ```python``` block.",
        "ref": "def unique_paths_grid(grid: list[list[int]]) -> int:\n    m, n = len(grid), len(grid[0])\n    dp = [0] * n\n    dp[0] = 0 if grid[0][0] == 1 else 1\n    for r in range(m):\n        for c in range(n):\n            if grid[r][c] == 1:\n                dp[c] = 0\n            elif c > 0:\n                dp[c] += dp[c - 1]\n    return dp[-1]",
        "bad": "def unique_paths_grid(grid):\n    return len(grid) + len(grid[0])",
        "harness": "assert unique_paths_grid([[0,0,0],[0,1,0],[0,0,0]]) == 2\nassert unique_paths_grid([[1,0]]) == 0\nprint('PASS')",
    },
    {
        "id": "py_11_topo_sort", "lang": "python",
        "prompt": "Write `topo_order(n: int, edges: list[tuple[int, int]]) -> list[int]` returning a valid topological order using Kahn's BFS (`[]` if cycle). Output ONLY one ```python``` block.",
        "ref": "from collections import deque\ndef topo_order(n: int, edges: list[tuple[int, int]]) -> list[int]:\n    adj = [[] for _ in range(n)]\n    indeg = [0] * n\n    for u, v in edges:\n        adj[u].append(v)\n        indeg[v] += 1\n    q = deque(i for i in range(n) if indeg[i] == 0)\n    out = []\n    while q:\n        u = q.popleft()\n        out.append(u)\n        for v in adj[u]:\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n    return out if len(out) == n else []",
        "bad": "def topo_order(n, edges):\n    return list(range(n))",
        "harness": "o = topo_order(4, [(0,1),(0,2),(1,3),(2,3)]); assert o[0] == 0 and o[-1] == 3 and len(o) == 4\nassert topo_order(2, [(0,1),(1,0)]) == []\nprint('PASS')",
    },
    {
        "id": "py_12_union_find", "lang": "python",
        "prompt": "Write `count_components(n: int, edges: list[tuple[int, int]]) -> int` using Union-Find with path compression. Output ONLY one ```python``` block.",
        "ref": "def count_components(n: int, edges: list[tuple[int, int]]) -> int:\n    p = list(range(n))\n    def find(x: int) -> int:\n        while p[x] != x:\n            p[x] = p[p[x]]\n            x = p[x]\n        return x\n    comp = n\n    for u, v in edges:\n        ru, rv = find(u), find(v)\n        if ru != rv:\n            p[ru] = rv\n            comp -= 1\n    return comp",
        "bad": "def count_components(n, edges):\n    return max(1, n - len(edges))",
        "harness": "assert count_components(5, [(0,1),(1,2),(3,4)]) == 2\nassert count_components(4, [(0,1),(1,2),(2,0)]) == 2\nprint('PASS')",
    },
    {
        "id": "py_13_bipartite", "lang": "python",
        "prompt": "Write `is_bipartite(n: int, edges: list[tuple[int, int]]) -> bool` checking undirected graph 2-colorability via BFS. Output ONLY one ```python``` block.",
        "ref": "from collections import deque\ndef is_bipartite(n: int, edges: list[tuple[int, int]]) -> bool:\n    adj = [[] for _ in range(n)]\n    for u, v in edges:\n        adj[u].append(v); adj[v].append(u)\n    col = [-1] * n\n    for s in range(n):\n        if col[s] != -1: continue\n        col[s] = 0\n        q = deque([s])\n        while q:\n            u = q.popleft()\n            for v in adj[u]:\n                if col[v] == -1:\n                    col[v] = 1 - col[u]; q.append(v)\n                elif col[v] == col[u]:\n                    return False\n    return True",
        "bad": "def is_bipartite(n, edges):\n    return True",
        "harness": "assert is_bipartite(4, [(0,1),(1,2),(2,3),(3,0)]) is True\nassert is_bipartite(3, [(0,1),(1,2),(2,0)]) is False\nprint('PASS')",
    },
    {
        "id": "py_14_trie", "lang": "python",
        "prompt": "Implement `class Trie:` with `insert(self, word: str) -> None`, `search(self, word: str) -> bool`, and `starts_with(self, prefix: str) -> bool`. Output ONLY one ```python``` block.",
        "ref": "class Trie:\n    def __init__(self) -> None:\n        self.root: dict = {}\n    def insert(self, word: str) -> None:\n        cur = self.root\n        for ch in word:\n            cur = cur.setdefault(ch, {})\n        cur['#'] = True\n    def search(self, word: str) -> bool:\n        cur = self.root\n        for ch in word:\n            if ch not in cur: return False\n            cur = cur[ch]\n        return '#' in cur\n    def starts_with(self, prefix: str) -> bool:\n        cur = self.root\n        for ch in prefix:\n            if ch not in cur: return False\n            cur = cur[ch]\n        return True",
        "bad": "class Trie:\n    def __init__(self):\n        self.words = set()\n    def insert(self, w): pass\n    def search(self, w): return False\n    def starts_with(self, p): return False",
        "harness": "t = Trie(); t.insert('apple'); assert t.search('apple') and not t.search('app') and t.starts_with('app'); print('PASS')",
    },
    {
        "id": "py_15_merge_intervals", "lang": "python",
        "prompt": "Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` merging overlapping `[start, end]` intervals sorted ascending. Output ONLY one ```python``` block.",
        "ref": "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    if not intervals:\n        return []\n    intervals = sorted(intervals)\n    out = [intervals[0][:]]\n    for s, e in intervals[1:]:\n        if s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out",
        "bad": "def merge_intervals(intervals):\n    return sorted(intervals)",
        "harness": "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[4,5],[1,4]]) == [[1,5]]\nprint('PASS')",
    },
    {
        "id": "py_16_sliding_max", "lang": "python",
        "prompt": "Write `sliding_window_max(nums: list[int], k: int) -> list[int]` in O(n) using `collections.deque`. Output ONLY one ```python``` block.",
        "ref": "from collections import deque\ndef sliding_window_max(nums: list[int], k: int) -> list[int]:\n    dq: deque[int] = deque()\n    out: list[int] = []\n    for i, x in enumerate(nums):\n        while dq and dq[0] <= i - k:\n            dq.popleft()\n        while dq and nums[dq[-1]] <= x:\n            dq.pop()\n        dq.append(i)\n        if i >= k - 1:\n            out.append(nums[dq[0]])\n    return out",
        "bad": "def sliding_window_max(nums, k):\n    return [min(nums[i:i+k]) for i in range(len(nums)-k+1)]",
        "harness": "assert sliding_window_max([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]\nprint('PASS')",
    },
    {
        "id": "py_17_daily_temps", "lang": "python",
        "prompt": "Write `daily_warmer(temps: list[int]) -> list[int]` returning days until a warmer temperature using a monotonic stack. Output ONLY one ```python``` block.",
        "ref": "def daily_warmer(temps: list[int]) -> list[int]:\n    res = [0] * len(temps)\n    st: list[int] = []\n    for i, t in enumerate(temps):\n        while st and temps[st[-1]] < t:\n            j = st.pop()\n            res[j] = i - j\n        st.append(i)\n    return res",
        "bad": "def daily_warmer(temps):\n    return [1] * len(temps)",
        "harness": "assert daily_warmer([73,74,75,71,69,72,76,73]) == [1,1,4,2,1,1,0,0]\nprint('PASS')",
    },
    {
        "id": "py_18_binary_search_rotated", "lang": "python",
        "prompt": "Write `search_rotated(nums: list[int], target: int) -> int` in O(log n) for distinct rotated sorted array (-1 if missing). Output ONLY one ```python``` block.",
        "ref": "def search_rotated(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] == target: return mid\n        if nums[lo] <= nums[mid]:\n            if nums[lo] <= target < nums[mid]: hi = mid - 1\n            else: lo = mid + 1\n        else:\n            if nums[mid] < target <= nums[hi]: lo = mid + 1\n            else: hi = mid - 1\n    return -1",
        "bad": "def search_rotated(nums, target):\n    return -1",
        "harness": "assert search_rotated([4,5,6,7,0,1,2], 0) == 4\nassert search_rotated([4,5,6,7,0,1,2], 3) == -1\nprint('PASS')",
    },
    {
        "id": "py_19_lca_bst", "lang": "python",
        "prompt": "Write `subarray_sum_k(nums: list[int], k: int) -> int` counting continuous subarrays summing to `k` in O(n). Output ONLY one ```python``` block.",
        "ref": "from collections import Counter\ndef subarray_sum_k(nums: list[int], k: int) -> int:\n    pref = Counter({0: 1})\n    cur = ans = 0\n    for x in nums:\n        cur += x\n        ans += pref[cur - k]\n        pref[cur] += 1\n    return ans",
        "bad": "def subarray_sum_k(nums, k):\n    return sum(1 for x in nums if x == k)",
        "harness": "assert subarray_sum_k([1,1,1], 2) == 2\nassert subarray_sum_k([1,2,3,-3,1,2], 3) == 6\nprint('PASS')",
    },
    {
        "id": "py_20_min_window_sub", "lang": "python",
        "prompt": "Write `min_window(s: str, t: str) -> str` returning minimum substring of `s` containing all chars of `t` with multiplicity. Output ONLY one ```python``` block.",
        "ref": "from collections import Counter\ndef min_window(s: str, t: str) -> str:\n    need = Counter(t)\n    missing = len(t)\n    best_l, best_r = 0, float('inf')\n    l = 0\n    for r, ch in enumerate(s, 1):\n        if need[ch] > 0: missing -= 1\n        need[ch] -= 1\n        while missing == 0:\n            if r - l < best_r - best_l: best_l, best_r = l, r\n            need[s[l]] += 1\n            if need[s[l]] > 0: missing += 1\n            l += 1\n    return '' if best_r == float('inf') else s[best_l:best_r]",
        "bad": "def min_window(s, t):\n    return s if t in s else ''",
        "harness": "assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'\nassert min_window('a', 'aa') == ''\nprint('PASS')",
    },
    {
        "id": "py_21_word_ladder", "lang": "python",
        "prompt": "Write `ladder_length(begin: str, end: str, words: list[str]) -> int` returning shortest single-letter transformation length (0 if unreachable). Output ONLY one ```python``` block.",
        "ref": "from collections import deque\ndef ladder_length(begin: str, end: str, words: list[str]) -> int:\n    wset = set(words)\n    if end not in wset: return 0\n    q = deque([(begin, 1)])\n    while q:\n        w, d = q.popleft()\n        if w == end: return d\n        for i in range(len(w)):\n            for c in 'abcdefghijklmnopqrstuvwxyz':\n                nw = w[:i] + c + w[i+1:]\n                if nw in wset:\n                    wset.remove(nw)\n                    q.append((nw, d + 1))\n    return 0",
        "bad": "def ladder_length(begin, end, words):\n    return 0",
        "harness": "assert ladder_length('hit', 'cog', ['hot','dot','dog','lot','log','cog']) == 5\nprint('PASS')",
    },
    {
        "id": "py_22_max_path_grid_dijkstra", "lang": "python",
        "prompt": "Write `min_cost_grid(grid: list[list[int]]) -> int` returning min sum path from `(0,0)` to `(m-1,n-1)` moving 4-directionally using `heapq`. Output ONLY one ```python``` block.",
        "ref": "import heapq\ndef min_cost_grid(grid: list[list[int]]) -> int:\n    m, n = len(grid), len(grid[0])\n    dist = [[10**18]*n for _ in range(m)]\n    dist[0][0] = grid[0][0]\n    pq = [(grid[0][0], 0, 0)]\n    while pq:\n        d, r, c = heapq.heappop(pq)\n        if (r, c) == (m - 1, n - 1): return d\n        if d > dist[r][c]: continue\n        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):\n            nr, nc = r + dr, c + dc\n            if 0 <= nr < m and 0 <= nc < n and d + grid[nr][nc] < dist[nr][nc]:\n                dist[nr][nc] = d + grid[nr][nc]\n                heapq.heappush(pq, (dist[nr][nc], nr, nc))\n    return dist[-1][-1]",
        "bad": "def min_cost_grid(grid):\n    return grid[0][0] + grid[-1][-1]",
        "harness": "assert min_cost_grid([[1,100,1],[1,1,1],[100,1,1]]) == 5\nprint('PASS')",
    },
    {
        "id": "py_23_ttl_cache", "lang": "python",
        "prompt": "Implement `class TTLCache:` in Python with `put(self, key: str, val: int, now: float, ttl: float)` and `get(self, key: str, now: float) -> int | None`. Output ONLY one ```python``` block.",
        "ref": "class TTLCache:\n    def __init__(self) -> None:\n        self._d: dict[str, tuple[int, float]] = {}\n    def put(self, key: str, val: int, now: float, ttl: float) -> None:\n        self._d[key] = (val, now + ttl)\n    def get(self, key: str, now: float) -> int | None:\n        item = self._d.get(key)\n        if item is None or now >= item[1]:\n            self._d.pop(key, None)\n            return None\n        return item[0]",
        "bad": "class TTLCache:\n    def __init__(self): self.d = {}\n    def put(self, k, v, now, ttl): self.d[k] = v\n    def get(self, k, now): return self.d.get(k)",
        "harness": "c = TTLCache(); c.put('a', 7, 10.0, 5.0); assert c.get('a', 14.9) == 7 and c.get('a', 15.0) is None; print('PASS')",
    },
    {
        "id": "py_24_median_stream", "lang": "python",
        "prompt": "Implement `class MedianFinder:` with `add_num(self, x: int) -> None` and `find_median(self) -> float` using two `heapq` heaps. Output ONLY one ```python``` block.",
        "ref": "import heapq\nclass MedianFinder:\n    def __init__(self) -> None:\n        self.lo: list[int] = []\n        self.hi: list[int] = []\n    def add_num(self, x: int) -> None:\n        heapq.heappush(self.lo, -x)\n        heapq.heappush(self.hi, -heapq.heappop(self.lo))\n        if len(self.hi) > len(self.lo):\n            heapq.heappush(self.lo, -heapq.heappop(self.hi))\n    def find_median(self) -> float:\n        if len(self.lo) > len(self.hi): return float(-self.lo[0])\n        return (-self.lo[0] + self.hi[0]) / 2.0",
        "bad": "class MedianFinder:\n    def __init__(self): self.a = []\n    def add_num(self, x): self.a.append(x)\n    def find_median(self): return float(self.a[0])",
        "harness": "m = MedianFinder(); m.add_num(1); m.add_num(2); assert m.find_median() == 1.5; m.add_num(3); assert m.find_median() == 2.0; print('PASS')",
    },
    {
        "id": "py_25_kruskal_mst", "lang": "python",
        "prompt": "Write `mst_cost(n: int, edges: list[tuple[int, int, int]]) -> int` returning Minimum Spanning Tree weight (-1 if disconnected). Output ONLY one ```python``` block.",
        "ref": "def mst_cost(n: int, edges: list[tuple[int, int, int]]) -> int:\n    p = list(range(n))\n    def find(x: int) -> int:\n        while p[x] != x:\n            p[x] = p[p[x]]\n            x = p[x]\n        return x\n    total = used = 0\n    for u, v, w in sorted(edges, key=lambda e: e[2]):\n        ru, rv = find(u), find(v)\n        if ru != rv:\n            p[ru] = rv; total += w; used += 1\n    return total if used == n - 1 else -1",
        "bad": "def mst_cost(n, edges):\n    return sum(w for _, _, w in edges)",
        "harness": "assert mst_cost(4, [(0,1,1),(1,2,2),(0,2,5),(2,3,3)]) == 6\nassert mst_cost(3, [(0,1,2)]) == -1\nprint('PASS')",
    },
    {
        "id": "py_26_scc_tarjan_or_kosaraju", "lang": "python",
        "prompt": "Write `count_sccs(n: int, edges: list[tuple[int, int]]) -> int` counting strongly connected components in a directed graph. Output ONLY one ```python``` block.",
        "ref": "def count_sccs(n: int, edges: list[tuple[int, int]]) -> int:\n    adj = [[] for _ in range(n)]\n    rev = [[] for _ in range(n)]\n    for u, v in edges:\n        adj[u].append(v); rev[v].append(u)\n    vis = [False] * n\n    order = []\n    for s in range(n):\n        if vis[s]: continue\n        st = [(s, 0)]; vis[s] = True\n        while st:\n            u, idx = st[-1]\n            if idx < len(adj[u]):\n                st[-1] = (u, idx + 1)\n                v = adj[u][idx]\n                if not vis[v]: vis[v] = True; st.append((v, 0))\n            else:\n                st.pop(); order.append(u)\n    vis = [False] * n\n    sccs = 0\n    for s in reversed(order):\n        if vis[s]: continue\n        sccs += 1; st = [s]; vis[s] = True\n        while st:\n            u = st.pop()\n            for v in rev[u]:\n                if not vis[v]: vis[v] = True; st.append(v)\n    return sccs",
        "bad": "def count_sccs(n, edges):\n    return n",
        "harness": "assert count_sccs(5, [(1,0),(0,2),(2,1),(0,3),(3,4)]) == 3\nprint('PASS')",
    },
    {
        "id": "py_27_house_robber", "lang": "python",
        "prompt": "Write `max_non_adjacent(nums: list[int]) -> int` returning maximum sum without adjacent elements. Output ONLY one ```python``` block.",
        "ref": "def max_non_adjacent(nums: list[int]) -> int:\n    p2 = p1 = 0\n    for x in nums:\n        p2, p1 = p1, max(p1, p2 + x)\n    return p1",
        "bad": "def max_non_adjacent(nums):\n    return max(sum(nums[::2]), sum(nums[1::2]))",
        "harness": "assert max_non_adjacent([2,7,9,3,1]) == 12\nassert max_non_adjacent([2,1,1,2]) == 4\nprint('PASS')",
    },
    {
        "id": "py_28_rpn_eval", "lang": "python",
        "prompt": "Write `eval_rpn(tokens: list[str]) -> int` evaluating Reverse Polish Notation with integer division truncated toward zero. Output ONLY one ```python``` block.",
        "ref": "def eval_rpn(tokens: list[str]) -> int:\n    st: list[int] = []\n    for t in tokens:\n        if t in ('+', '-', '*', '/'):\n            b, a = st.pop(), st.pop()\n            if t == '+': st.append(a + b)\n            elif t == '-': st.append(a - b)\n            elif t == '*': st.append(a * b)\n            else: st.append(int(a / b))\n        else:\n            st.append(int(t))\n    return st[0]",
        "bad": "def eval_rpn(tokens):\n    return eval(' '.join(tokens))",
        "harness": "assert eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+']) == 22\nprint('PASS')",
    },
    {
        "id": "py_29_valid_parens", "lang": "python",
        "prompt": "Write `is_balanced_brackets(s: str) -> bool` validating `()[]{}` brackets properly. Output ONLY one ```python``` block.",
        "ref": "def is_balanced_brackets(s: str) -> bool:\n    pair = {')': '(', ']': '[', '}': '{'}\n    st: list[str] = []\n    for c in s:\n        if c in pair:\n            if not st or st.pop() != pair[c]: return False\n        elif c in '([{':\n            st.append(c)\n    return not st",
        "bad": "def is_balanced_brackets(s):\n    return s.count('(') == s.count(')')",
        "harness": "assert is_balanced_brackets('([{}])') and not is_balanced_brackets('([)]'); print('PASS')",
    },
    {
        "id": "py_30_lfu_cache", "lang": "python",
        "prompt": "Implement `class LFUCache:` with `__init__(self, cap: int)`, `get(self, key: int) -> int`, `put(self, key: int, val: int) -> None` using `OrderedDict`. Output ONLY one ```python``` block.",
        "ref": "from collections import defaultdict, OrderedDict\nclass LFUCache:\n    def __init__(self, cap: int) -> None:\n        self.cap = cap\n        self.kv: dict[int, tuple[int, int]] = {}\n        self.freq: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)\n        self.min_f = 0\n    def _touch(self, k: int) -> int:\n        v, f = self.kv[k]\n        self.freq[f].pop(k)\n        if not self.freq[f] and f == self.min_f: self.min_f += 1\n        self.freq[f + 1][k] = None\n        self.kv[k] = (v, f + 1)\n        return v\n    def get(self, key: int) -> int:\n        return -1 if key not in self.kv else self._touch(key)\n    def put(self, key: int, val: int) -> None:\n        if self.cap <= 0: return\n        if key in self.kv:\n            _, f = self.kv[key]; self.kv[key] = (val, f); self._touch(key); return\n        if len(self.kv) == self.cap:\n            ev, _ = self.freq[self.min_f].popitem(last=False)\n            self.kv.pop(ev)\n        self.kv[key] = (val, 1); self.freq[1][key] = None; self.min_f = 1",
        "bad": "class LFUCache:\n    def __init__(self, cap): self.d = {}\n    def get(self, k): return self.d.get(k, -1)\n    def put(self, k, v): self.d[k] = v",
        "harness": "c = LFUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1 and c.get(3)==3; print('PASS')",
    },
    # --- 10 Rust ---
    {
        "id": "rs_01_parse_positive_sum", "lang": "rust",
        "prompt": "Write safe Rust `pub fn parse_positive_sum(input: &str) -> Result<i64, String>` using `token.parse::<i64>()` and `checked_add`, returning Err on empty, `<= 0`, bad token, or overflow (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "pub fn parse_positive_sum(input: &str) -> Result<i64, String> {\n    let mut total: i64 = 0;\n    let mut count = 0usize;\n    for tok in input.split_whitespace() {\n        let v = tok.parse::<i64>().map_err(|e| e.to_string())?;\n        if v <= 0 { return Err(\"non-positive\".into()); }\n        total = total.checked_add(v).ok_or_else(|| \"overflow\".to_string())?;\n        count += 1;\n    }\n    if count == 0 { return Err(\"empty\".into()); }\n    Ok(total)\n}",
        "bad": "pub fn parse_positive_sum(input: &str) -> Result<i64, String> {\n    let mut total: i64 = 0;\n    for tok in input.split_whitespace() {\n        let v = tok.parse<i64>().map_err(|_| \"err\")?;\n        total += v;\n    }\n    Ok(total)\n}",
        "harness": "fn main() { assert_eq!(parse_positive_sum(\"10 20 5\"), Ok(35)); assert!(parse_positive_sum(\"\").is_err()); assert!(parse_positive_sum(\"1 -2 3\").is_err()); println!(\"PASS\"); }",
    },
    {
        "id": "rs_02_merge_intervals", "lang": "rust",
        "prompt": "Write Rust `pub fn merge_intervals(mut intervals: Vec<(i32, i32)>) -> Vec<(i32, i32)>` (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "pub fn merge_intervals(mut intervals: Vec<(i32, i32)>) -> Vec<(i32, i32)> {\n    if intervals.is_empty() { return vec![]; }\n    intervals.sort_unstable();\n    let mut out = vec![intervals[0]];\n    for (s, e) in intervals.into_iter().skip(1) {\n        let last = out.last_mut().unwrap();\n        if s <= last.1 { if e > last.1 { last.1 = e; } } else { out.push((s, e)); }\n    }\n    out\n}",
        "bad": "pub fn merge_intervals(intervals: Vec<(i32, i32)>) -> Vec<(i32, i32)> { intervals }",
        "harness": "fn main() { assert_eq!(merge_intervals(vec![(4,5),(1,4)]), vec![(1,5)]); assert_eq!(merge_intervals(vec![]), vec![]); println!(\"PASS\"); }",
    },
    {
        "id": "rs_03_binary_search", "lang": "rust",
        "prompt": "Write Rust `pub fn bsearch(nums: &[i32], target: i32) -> Option<usize>` (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "pub fn bsearch(nums: &[i32], target: i32) -> Option<usize> {\n    let (mut lo, mut hi) = (0usize, nums.len());\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        if nums[mid] == target { return Some(mid); }\n        if nums[mid] < target { lo = mid + 1; } else { hi = mid; }\n    }\n    None\n}",
        "bad": "pub fn bsearch(nums: &[i32], _target: i32) -> Option<usize> { Some(nums.len()) }",
        "harness": "fn main() { assert_eq!(bsearch(&[1,3,5,7], 5), Some(2)); assert_eq!(bsearch(&[1,3,5], 2), None); println!(\"PASS\"); }",
    },
    {
        "id": "rs_04_rle_encode", "lang": "rust",
        "prompt": "Write Rust `pub fn rle_encode(s: &str) -> String` encoding consecutive ASCII chars like `aaabb -> a3b2` (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "pub fn rle_encode(s: &str) -> String {\n    let mut out = String::new();\n    let mut it = s.chars().peekable();\n    while let Some(c) = it.next() {\n        let mut cnt = 1usize;\n        while it.peek() == Some(&c) { it.next(); cnt += 1; }\n        out.push(c);\n        out.push_str(&cnt.to_string());\n    }\n    out\n}",
        "bad": "pub fn rle_encode(s: &str) -> String { s.to_string() }",
        "harness": "fn main() { assert_eq!(rle_encode(\"aaabbc\"), \"a3b2c1\"); assert_eq!(rle_encode(\"\"), \"\"); println!(\"PASS\"); }",
    },
    {
        "id": "rs_05_ring_buffer", "lang": "rust",
        "prompt": "Write Rust `pub struct RingBuf` with `new(cap: usize)`, `push(&mut self, v: i32) -> bool`, `pop(&mut self) -> Option<i32>` (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "use std::collections::VecDeque;\npub struct RingBuf { cap: usize, q: VecDeque<i32> }\nimpl RingBuf {\n    pub fn new(cap: usize) -> Self { Self { cap, q: VecDeque::with_capacity(cap) } }\n    pub fn push(&mut self, v: i32) -> bool { if self.q.len() == self.cap { return false; } self.q.push_back(v); true }\n    pub fn pop(&mut self) -> Option<i32> { self.q.pop_front() }\n}",
        "bad": "pub struct RingBuf; impl RingBuf { pub fn new(_: usize) -> Self { Self } pub fn push(&mut self, _: i32) -> bool { false } pub fn pop(&mut self) -> Option<i32> { None } }",
        "harness": "fn main() { let mut r = RingBuf::new(2); assert!(r.push(1) && r.push(2) && !r.push(3)); assert_eq!(r.pop(), Some(1)); println!(\"PASS\"); }",
    },
    {
        "id": "rs_06_prefix_sums", "lang": "rust",
        "prompt": "Write Rust `pub fn range_sums(nums: &[i32], queries: &[(usize, usize)]) -> Vec<i64>` for inclusive range sums (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "pub fn range_sums(nums: &[i32], queries: &[(usize, usize)]) -> Vec<i64> {\n    let mut pref = Vec::with_capacity(nums.len() + 1);\n    pref.push(0i64);\n    for &x in nums { pref.push(pref.last().unwrap() + i64::from(x)); }\n    queries.iter().map(|&(l, r)| pref[r + 1] - pref[l]).collect()\n}",
        "bad": "pub fn range_sums(_: &[i32], q: &[(usize, usize)]) -> Vec<i64> { vec![0; q.len()] }",
        "harness": "fn main() { assert_eq!(range_sums(&[1,2,3,4], &[(0,2),(1,3)]), vec![6,9]); println!(\"PASS\"); }",
    },
    {
        "id": "rs_07_valid_brackets", "lang": "rust",
        "prompt": "Write Rust `pub fn valid_brackets(s: &str) -> bool` checking `()[]{}` nesting (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "pub fn valid_brackets(s: &str) -> bool {\n    let mut st = Vec::new();\n    for c in s.chars() {\n        match c {\n            '(' | '[' | '{' => st.push(c),\n            ')' => if st.pop() != Some('(') { return false; },\n            ']' => if st.pop() != Some('[') { return false; },\n            '}' => if st.pop() != Some('{') { return false; },\n            _ => {}\n        }\n    }\n    st.is_empty()\n}",
        "bad": "pub fn valid_brackets(_s: &str) -> bool { false }",
        "harness": "fn main() { assert!(valid_brackets(\"([{}])\")); assert!(!valid_brackets(\"([)]\")); println!(\"PASS\"); }",
    },
    {
        "id": "rs_08_transpose", "lang": "rust",
        "prompt": "Write Rust `pub fn transpose(mat: &[Vec<i32>]) -> Vec<Vec<i32>>` (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "pub fn transpose(mat: &[Vec<i32>]) -> Vec<Vec<i32>> {\n    if mat.is_empty() || mat[0].is_empty() { return vec![]; }\n    let (r, c) = (mat.len(), mat[0].len());\n    (0..c).map(|j| (0..r).map(|i| mat[i][j]).collect()).collect()\n}",
        "bad": "pub fn transpose(mat: &[Vec<i32>]) -> Vec<Vec<i32>> { mat.to_vec() }",
        "harness": "fn main() { assert_eq!(transpose(&[vec![1,2,3],vec![4,5,6]]), vec![vec![1,4],vec![2,5],vec![3,6]]); println!(\"PASS\"); }",
    },
    {
        "id": "rs_09_top_freq", "lang": "rust",
        "prompt": "Write Rust `pub fn most_frequent(nums: &[i32]) -> Option<i32>` breaking ties by smaller integer (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "use std::collections::HashMap;\npub fn most_frequent(nums: &[i32]) -> Option<i32> {\n    let mut m: HashMap<i32, usize> = HashMap::new();\n    for &x in nums { *m.entry(x).or_insert(0) += 1; }\n    m.into_iter().max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0))).map(|p| p.0)\n}",
        "bad": "pub fn most_frequent(nums: &[i32]) -> Option<i32> { nums.first().copied() }",
        "harness": "fn main() { assert_eq!(most_frequent(&[3,1,2,2,1]), Some(1)); assert_eq!(most_frequent(&[]), None); println!(\"PASS\"); }",
    },
    {
        "id": "rs_10_semver_parse", "lang": "rust",
        "prompt": "Write Rust `pub fn cmp_semver(a: &str, b: &str) -> Result<std::cmp::Ordering, String>` comparing `major.minor.patch` numeric triples (no `fn main()`). Output ONLY one ```rust``` block.",
        "ref": "use std::cmp::Ordering;\npub fn cmp_semver(a: &str, b: &str) -> Result<Ordering, String> {\n    fn parse3(s: &str) -> Result<(u64, u64, u64), String> {\n        let ps: Vec<&str> = s.split('.').collect();\n        if ps.len() != 3 { return Err(\"bad\".into()); }\n        Ok((ps[0].parse().map_err(|_| \"bad\")?, ps[1].parse().map_err(|_| \"bad\")?, ps[2].parse().map_err(|_| \"bad\")?))\n    }\n    Ok(parse3(a)?.cmp(&parse3(b)?))\n}",
        "bad": "pub fn cmp_semver(a: &str, b: &str) -> Result<std::cmp::Ordering, String> { Ok(a.cmp(b)) }",
        "harness": "fn main() { assert_eq!(cmp_semver(\"1.10.0\", \"1.2.9\"), Ok(std::cmp::Ordering::Greater)); assert!(cmp_semver(\"1.2\", \"1.2.0\").is_err()); println!(\"PASS\"); }",
    },
    # --- 10 JavaScript ---
    {
        "id": "js_01_deepclone", "lang": "javascript",
        "prompt": "Write JS `function deepClone(value, seen = new WeakMap())` deeply cloning primitives, Arrays, plain Objects, Dates, and circular references. Output ONLY one ```javascript``` block.",
        "ref": "function deepClone(value, seen = new WeakMap()) {\n  if (value === null || typeof value !== 'object') return value;\n  if (value instanceof Date) return new Date(value.getTime());\n  if (seen.has(value)) return seen.get(value);\n  const out = Array.isArray(value) ? [] : {};\n  seen.set(value, out);\n  for (const key of Object.keys(value)) out[key] = deepClone(value[key], seen);\n  return out;\n}",
        "bad": "function deepClone(value) { return JSON.parse(JSON.stringify(value)); }",
        "harness": "const o={a:1,d:new Date(1700000000000),n:[1,{b:2}]};o.self=o;const c=deepClone(o);if(c===o||c.self!==c||!(c.d instanceof Date))throw new Error('fail');console.log('PASS');",
    },
    {
        "id": "js_02_lru_map", "lang": "javascript",
        "prompt": "Implement JS `class LRUCache` using `Map` insertion order with `get(key)` and `put(key, val)`. Output ONLY one ```javascript``` block.",
        "ref": "class LRUCache {\n  constructor(capacity) { this.cap = capacity; this.map = new Map(); }\n  get(key) {\n    if (!this.map.has(key)) return -1;\n    const v = this.map.get(key); this.map.delete(key); this.map.set(key, v); return v;\n  }\n  put(key, val) {\n    if (this.map.has(key)) this.map.delete(key);\n    this.map.set(key, val);\n    if (this.map.size > this.cap) this.map.delete(this.map.keys().next().value);\n  }\n}",
        "bad": "class LRUCache { constructor(c) { this.m = new Map(); } get(k) { return -1; } put(k, v) {} }",
        "harness": "const c=new LRUCache(2);c.put(1,1);c.put(2,2);if(c.get(1)!==1)throw 1;c.put(3,3);if(c.get(2)!==-1)throw 2;console.log('PASS');",
    },
    {
        "id": "js_03_event_emitter", "lang": "javascript",
        "prompt": "Implement JS `class Emitter` with `on(ev, fn)`, `off(ev, fn)`, and `emit(ev, ...args)`. Output ONLY one ```javascript``` block.",
        "ref": "class Emitter {\n  constructor() { this.m = new Map(); }\n  on(ev, fn) { if (!this.m.has(ev)) this.m.set(ev, []); this.m.get(ev).push(fn); }\n  off(ev, fn) { if (this.m.has(ev)) this.m.set(ev, this.m.get(ev).filter(f => f !== fn)); }\n  emit(ev, ...args) { for (const fn of [...(this.m.get(ev) || [])]) fn(...args); }\n}",
        "bad": "class Emitter { on() {} off() {} emit() {} }",
        "harness": "let s=0;const e=new Emitter();const f=x=>{s+=x};e.on('a',f);e.emit('a',3);e.off('a',f);e.emit('a',5);if(s!==3)throw 1;console.log('PASS');",
    },
    {
        "id": "js_04_merge_intervals", "lang": "javascript",
        "prompt": "Write JS `function mergeIntervals(intervals)` merging overlapping `[s, e]` intervals sorted ascending. Output ONLY one ```javascript``` block.",
        "ref": "function mergeIntervals(intervals) {\n  if (!intervals.length) return [];\n  const sorted = [...intervals].sort((a, b) => a[0] - b[0] || a[1] - b[1]);\n  const out = [[...sorted[0]]];\n  for (let i = 1; i < sorted.length; i++) {\n    const [s, e] = sorted[i], last = out[out.length - 1];\n    if (s <= last[1]) last[1] = Math.max(last[1], e); else out.push([s, e]);\n  }\n  return out;\n}",
        "bad": "function mergeIntervals(intervals) { return intervals; }",
        "harness": "const r=mergeIntervals([[4,5],[1,4]]);if(r.length!==1||r[0][0]!==1||r[0][1]!==5)throw 1;console.log('PASS');",
    },
    {
        "id": "js_05_flatten_depth", "lang": "javascript",
        "prompt": "Write JS `function flatten(arr, depth = Infinity)` recursively flattening arrays up to `depth`. Output ONLY one ```javascript``` block.",
        "ref": "function flatten(arr, depth = Infinity) {\n  const out = [];\n  for (const item of arr) {\n    if (Array.isArray(item) && depth > 0) out.push(...flatten(item, depth - 1));\n    else out.push(item);\n  }\n  return out;\n}",
        "bad": "function flatten(arr) { return arr; }",
        "harness": "const r=flatten([1,[2,[3,4]]],1);if(JSON.stringify(r)!=='[1,2,[3,4]]')throw 1;console.log('PASS');",
    },
    {
        "id": "js_06_topo_sort", "lang": "javascript",
        "prompt": "Write JS `function topoOrder(n, edges)` returning topological order array or `[]` on cycle. Output ONLY one ```javascript``` block.",
        "ref": "function topoOrder(n, edges) {\n  const adj = Array.from({ length: n }, () => []);\n  const ind = Array(n).fill(0);\n  for (const [u, v] of edges) { adj[u].push(v); ind[v]++; }\n  const q = []; for (let i = 0; i < n; i++) if (ind[i] === 0) q.push(i);\n  const out = [];\n  while (q.length) {\n    const u = q.shift(); out.push(u);\n    for (const v of adj[u]) if (--ind[v] === 0) q.push(v);\n  }\n  return out.length === n ? out : [];\n}",
        "bad": "function topoOrder(n) { return []; }",
        "harness": "if(topoOrder(3,[[0,1],[1,2]]).join(',')!=='0,1,2'||topoOrder(2,[[0,1],[1,0]]).length!==0)throw 1;console.log('PASS');",
    },
    {
        "id": "js_07_group_by", "lang": "javascript",
        "prompt": "Write JS `function groupBy(items, keyFn)` returning plain object mapping keys to arrays. Output ONLY one ```javascript``` block.",
        "ref": "function groupBy(items, keyFn) {\n  const out = {};\n  for (const it of items) { const k = String(keyFn(it)); (out[k] ||= []).push(it); }\n  return out;\n}",
        "bad": "function groupBy(items) { return { all: items }; }",
        "harness": "const g=groupBy([1,2,3,4],x=>x%2);if(g['0'].length!==2||g['1'].length!==2)throw 1;console.log('PASS');",
    },
    {
        "id": "js_08_min_heap", "lang": "javascript",
        "prompt": "Implement JS `class MinHeap` with `push(x)` and `pop()` in O(log n). Output ONLY one ```javascript``` block.",
        "ref": "class MinHeap {\n  constructor() { this.a = []; }\n  push(x) {\n    const a = this.a; a.push(x); let i = a.length - 1;\n    while (i > 0) { const p = (i - 1) >> 1; if (a[p] <= a[i]) break; [a[p], a[i]] = [a[i], a[p]]; i = p; }\n  }\n  pop() {\n    const a = this.a; if (!a.length) return undefined; const top = a[0], last = a.pop();\n    if (a.length) { a[0] = last; let i = 0; while (true) { let s = i, l = 2*i+1, r = 2*i+2; if (l < a.length && a[l] < a[s]) s = l; if (r < a.length && a[r] < a[s]) s = r; if (s === i) break; [a[i], a[s]] = [a[s], a[i]]; i = s; } }\n    return top;\n  }\n}",
        "bad": "class MinHeap { constructor() { this.a = []; } push(x) { this.a.push(x); } pop() { return this.a.pop(); } }",
        "harness": "const h=new MinHeap();[5,1,4,2].forEach(x=>h.push(x));if([h.pop(),h.pop(),h.pop(),h.pop()].join(',')!=='1,2,4,5')throw 1;console.log('PASS');",
    },
    {
        "id": "js_09_promise_pool", "lang": "javascript",
        "prompt": "Write async JS `async function runPool(tasks, limit)` running zero-arg async thunks with max concurrency `limit` and preserving order. Output ONLY one ```javascript``` block.",
        "ref": "async function runPool(tasks, limit) {\n  const out = Array(tasks.length); let idx = 0;\n  async function worker() { while (idx < tasks.length) { const i = idx++; out[i] = await tasks[i](); } }\n  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));\n  return out;\n}",
        "bad": "async function runPool(tasks) { return tasks; }",
        "harness": "(async()=>{const r=await runPool([()=>Promise.resolve(1),()=>Promise.resolve(2),()=>Promise.resolve(3)],2);if(r.join(',')!=='1,2,3')process.exit(1);console.log('PASS');})();",
    },
    {
        "id": "js_10_semver_sort", "lang": "javascript",
        "prompt": "Write JS `function compareSemver(a, b)` returning -1, 0, or 1 comparing `major.minor.patch` numerically. Output ONLY one ```javascript``` block.",
        "ref": "function compareSemver(a, b) {\n  const pa = a.split('.').map(Number), pb = b.split('.').map(Number);\n  for (let i = 0; i < 3; i++) if (pa[i] !== pb[i]) return pa[i] < pb[i] ? -1 : 1;\n  return 0;\n}",
        "bad": "function compareSemver(a, b) { return a < b ? -1 : a > b ? 1 : 0; }",
        "harness": "if(compareSemver('1.10.0','1.2.9')!==1||compareSemver('2.0.0','2.0.0')!==0)throw 1;console.log('PASS');",
    },
]
assert len(CODING_TASKS) == 50


def _tc(cid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def _gen_coding_pairs(
    server_exe: Path,
    teacher_gguf: Path,
    student_gguf: Path,
    log_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sys_no_tool = build_nc_system_prompt("coder", depth=1)
    teacher_ans: dict[str, str] = {}
    student_ans: dict[str, str] = {}
    stats = {"teacher_live_pass": 0, "teacher_ref_fallback": 0, "student_live_fail": 0, "student_bug_fallback": 0}

    with LlamaServerSession(server_exe=server_exe, model_path=teacher_gguf, ctx_size=4096, ple_cpu_offload=True, log_dir=log_dir / "teacher") as srv:
        for task in CODING_TASKS:
            tel = stream_chat_completion(
                srv.host, srv.port,
                {"messages": [{"role": "system", "content": sys_no_tool}, {"role": "user", "content": task["prompt"]}], "max_tokens": 900, "temperature": 0.1, "chat_template_kwargs": {"enable_thinking": False}},
                timeout_s=90.0,
            )
            code = extract_code_block(tel.content) if tel.ok else ""
            ok, _ = _run_code_harness(task["lang"], code, task["harness"])
            if ok:
                teacher_ans[task["id"]] = f"```{task['lang']}\n{code}\n```"
                stats["teacher_live_pass"] += 1
            else:
                ok_ref, err_ref = _run_code_harness(task["lang"], task["ref"], task["harness"])
                assert ok_ref, f"Reference solution failed {task['id']}: {err_ref}"
                teacher_ans[task["id"]] = f"```{task['lang']}\n{task['ref']}\n```"
                stats["teacher_ref_fallback"] += 1

    with LlamaServerSession(server_exe=server_exe, model_path=student_gguf, ctx_size=4096, ple_cpu_offload=True, log_dir=log_dir / "student") as srv:
        for task in CODING_TASKS:
            tel = stream_chat_completion(
                srv.host, srv.port,
                {"messages": [{"role": "system", "content": sys_no_tool}, {"role": "user", "content": task["prompt"]}], "max_tokens": 900, "temperature": 0.1, "chat_template_kwargs": {"enable_thinking": False}},
                timeout_s=90.0,
            )
            code = extract_code_block(tel.content) if tel.ok else ""
            ok, _ = _run_code_harness(task["lang"], code, task["harness"])
            if (not ok) and (tel.content or "").strip():
                student_ans[task["id"]] = tel.content.strip()
                stats["student_live_fail"] += 1
            else:
                ok_bad, _ = _run_code_harness(task["lang"], task["bad"], task["harness"])
                assert not ok_bad, f"Bad fallback unexpectedly passed {task['id']}"
                student_ans[task["id"]] = f"```{task['lang']}\n{task['bad']}\n```"
                stats["student_bug_fallback"] += 1

    pairs: list[dict[str, Any]] = []
    for idx, task in enumerate(CODING_TASKS, 1):
        pairs.append({
            "id": f"v21_code_{idx:02d}",
            "bucket": "on_policy_coding_patch",
            "mode": "no_tool",
            "tools": [],
            "thinking": False,
            "prompt": [{"role": "system", "content": sys_no_tool}, {"role": "user", "content": task["prompt"]}],
            "chosen": {"role": "assistant", "content": teacher_ans[task["id"]], "reasoning_content": None},
            "rejected": {"role": "assistant", "content": student_ans[task["id"]], "reasoning_content": None},
        })
    return pairs, stats


def _gen_live_nc_efficiency_pairs() -> list[dict[str, Any]]:
    sys_lead = build_nc_system_prompt("coder", depth=0)
    sys_worker = build_nc_system_prompt("coder", depth=1)
    pairs: list[dict[str, Any]] = []

    # 15 T1 pairs: single search_text / read_slice on Cargo.toml vs scattered reads
    t1_scenarios = [
        ("nuclear_code", "crates/nc-core/Cargo.toml", 'name = "nc-'),
        ("acme_ws", "packages/acme-api/Cargo.toml", 'name = "acme-'),
        ("kv_engine", "crates/kv-store/Cargo.toml", 'name = "kv-'),
        ("dsp_ws", "crates/dsp-fx/Cargo.toml", 'name = "dsp-'),
        ("mesh_repo", "crates/mesh-net/Cargo.toml", 'name = "mesh-'),
    ]
    for i in range(15):
        repo_name, sample_crate, name_prefix = t1_scenarios[i % len(t1_scenarios)]
        rmap = "\n".join([
            format_repo_map_file("Cargo.toml", 34, [("workspace", 1)]),
            format_repo_map_file(sample_crate, 24, [("package", 1)]),
            format_repo_map_file("src/main.rs", 80, [("main", 4)]),
        ])
        if i % 2 == 0:
            chosen_tc = _tc("c1", "read_slice", {"path": "Cargo.toml", "start_line": 1, "end_line": 34})
            rej_tcs = [_tc("r1", "list_files", {"dir": "crates"})]
            task_txt = f"[Repo {i+1:02d}] List every workspace member crate listed in `{repo_name}/Cargo.toml` verbatim."
        else:
            chosen_tc = _tc("c1", "search_text", {"query": name_prefix, "ext": "toml", "max_hits": 20})
            rej_tcs = [_tc("r1", "read_slice", {"path": sample_crate, "start_line": 1, "end_line": 24}), _tc("r2", "search_text", {"query": "edition", "ext": "toml"})]
            task_txt = f"Find all crate names under `crates/` matching prefix `{name_prefix}` in one step (index={i})."
        pairs.append({
            "id": f"v21_t1_{i+1:02d}",
            "bucket": "live_nc_step_efficiency",
            "mode": "nc_agent",
            "thinking": False,
            "prompt": [{"role": "system", "content": sys_lead}, {"role": "user", "content": build_nc_user_prompt(repo_map=rmap, task=task_txt)}],
            "chosen": {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [chosen_tc]},
            "rejected": {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": rej_tcs},
        })

    # 15 T4 pairs: distinct files/symbols/stages -> 15 unique prompts
    t4_specs = [
        ("src/math.rs", "apply_tax", "pub fn apply_tax(price: i32) -> i32 {\n    price + 1\n}", "pub fn apply_tax(price: i32) -> i32 {\n    price * 110 / 100\n}", "pub fn apply_tax(price: i32) -> i32 {\n    (price * 110 + 50) / 100\n}", "price * 110 / 100", "i32"),
        ("src/fees.rs", "add_fee", "pub fn add_fee(cents: i64) -> i64 {\n    cents\n}", "pub fn add_fee(cents: i64) -> i64 {\n    cents * 105 / 100\n}", "pub fn add_fee(cents: i64) -> f64 {\n    cents as f64 * 1.05\n}", "cents * 105 / 100", "i64"),
        ("src/clip.rs", "clamp_pct", "pub fn clamp_pct(v: i32) -> i32 {\n    v\n}", "pub fn clamp_pct(v: i32) -> i32 {\n    v.clamp(0, 100)\n}", "pub fn clamp_pct(v: i32) -> i32 {\n    if v > 100 { 99 } else { 0 }\n}", "v.clamp(0, 100)", "i32"),
        ("src/billing.rs", "net_total", "pub fn net_total(amt: i32) -> i32 {\n    amt\n}", "pub fn net_total(amt: i32) -> i32 {\n    amt.saturating_sub(50)\n}", "pub fn net_total(amt: i32) -> i32 {\n    amt - 50\n}", "amt.saturating_sub(50)", "i32"),
        ("src/rate.rs", "scale_bps", "pub fn scale_bps(v: u64) -> u64 {\n    v\n}", "pub fn scale_bps(v: u64) -> u64 {\n    v * 25 / 10_000\n}", "pub fn scale_bps(v: u64) -> f64 {\n    v as f64 * 0.0025\n}", "v * 25 / 10_000", "u64"),
    ]
    for i in range(15):
        fpath, sym, old_fn, good_fn, bad_fn, formula, rtype = t4_specs[i % len(t4_specs)]
        stage = i // len(t4_specs)  # 0, 1, 2 across 5 distinct specs -> 15 unique prompts
        rmap = format_repo_map_file(fpath, 48, [(sym, 1), ("tests", 20)])
        if stage == 0:
            # Stage A: after read_symbol, emit exact literal edit_file (no invented rounding/type mutation)
            prompt = [
                {"role": "system", "content": sys_worker},
                {"role": "user", "content": build_nc_user_prompt(repo_map=rmap, task=f"Edit `{fpath}` so `{sym}` computes `{formula}` keeping `{rtype}`.")},
                {"role": "assistant", "reasoning_content": None, "content": None, "tool_calls": [_tc("c1", "read_symbol", {"path": fpath, "symbol": sym})]},
                {"role": "tool", "tool_call_id": "c1", "name": "read_symbol", "content": old_fn},
            ]
            chosen = {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [_tc("c2", "edit_file", {"path": fpath, "target": old_fn, "replacement": good_fn})]}
            rejected = {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [_tc("r2", "edit_file", {"path": fpath, "target": old_fn, "replacement": bad_fn})]}
        elif stage == 1:
            # Stage B: after edit_file succeeds, run cargo test once
            prompt = [
                {"role": "system", "content": sys_worker},
                {"role": "user", "content": build_nc_user_prompt(repo_map=rmap, task=f"Fix `{sym}` in `{fpath}` ({formula}, keep `{rtype}`) and verify.")},
                {"role": "assistant", "reasoning_content": None, "content": None, "tool_calls": [_tc("c1", "edit_file", {"path": fpath, "target": old_fn, "replacement": good_fn})]},
                {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": f"edited {fpath}"},
            ]
            chosen = {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [_tc("c2", "run_command", {"cmd": "cargo test"})]}
            rejected = {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [_tc("r2", "read_slice", {"path": fpath, "start_line": 1, "end_line": 40})]}
        else:
            # Stage C: after cargo test passes, STOP immediately with SubTaskReceipt (never re-read file!)
            prompt = [
                {"role": "system", "content": sys_worker},
                {"role": "user", "content": build_nc_user_prompt(repo_map=rmap, task=f"Fix `{sym}` in `{fpath}` ({formula}, keep `{rtype}`) and report completion.")},
                {"role": "assistant", "reasoning_content": None, "content": None, "tool_calls": [_tc("c1", "run_command", {"cmd": "cargo test"})]},
                {"role": "tool", "tool_call_id": "c1", "name": "run_command", "content": "test result: ok. 1 passed; 0 failed"},
            ]
            chosen = {"role": "assistant", "reasoning_content": None, "content": f"STATUS: DONE\nCHANGES: Updated `{sym}` to `{formula}` keeping `{rtype}`\nFILES: {fpath}\nISSUES: none"}
            rejected = {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [_tc("r2", "read_slice", {"path": fpath, "start_line": 1, "end_line": 48})]}
        pairs.append({
            "id": f"v21_t4_{i+1:02d}",
            "bucket": "live_nc_step_efficiency",
            "mode": "nc_agent",
            "thinking": False,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return pairs


def _gen_adapt_think_pairs() -> list[dict[str, Any]]:
    sys_lead = build_nc_system_prompt("coder", depth=0)
    scenarios = [
        ("Which bug causes `apply_tax(100)` returning 101 instead of 110, and why must return type stay `i32`?", "Spot bug (`price + 1`) and abi contract (`i32` integer arithmetic).", "`apply_tax` uses `price + 1` instead of `price * 110 / 100`. Return type must stay `i32` to preserve callers and integer truncation semantics.", "Let me think carefully about every possible tax formula across regions, floats vs decimals, overflow edge cases, ABI stability, caller signatures, and refactoring options before answering anything concise here..." * 3),
        ("Why does `TokenUsageStats::reset` leave `by_model` entries empty rather than keeping stale counts?", "Explain `*self = Self::default()` clearing map.", "`reset` assigns `Self::default()`, zeroing totals and clearing `by_model`. Keeping stale map counts would misreport next session usage.", "<|tool_call>call:read_symbol{path:<|\"|>crates/nc-types/src/settings.rs<|\"|>,symbol:<|\"|>reset<|\"|>}<tool_call|>"),
        ("Why must depth-1 subagent workers omit `spawn_subagent` from their toolset?", "Keep delegation bounded to depth 1.", "Omitting `spawn_subagent` enforces single-level sequential delegation and prevents unbounded recursive agent trees. It preserves bounded VRAM and latency.", "I need to inspect the codebase first. <|tool_call>call:search_text{query:<|\"|>spawn_subagent<|\"|>}<tool_call|>"),
        ("Why does `edit_file` forbid passing whole-file `content` alongside `target`/`replacement`?", "Prevent accidental file truncation.", "Forbidding `content` enforces surgical target replacement and avoids overwriting untouched lines. Callers must pass exact `target` and `replacement`.", "Thinking very long about why search-and-replace tools differ from overwrite tools across Unix editors and AST patchers in Rust and Python..." * 3),
        ("In Gemma 4 inference, why offload `per_layer_token_embd.weight=CPU` on 4 GB VRAM?", "Free ~1 GB VRAM for KV cache.", "Offloading `per_layer_token_embd.weight` to CPU frees ~1 GB of VRAM so 16K `q8_0` KV cache and `mmproj` fit inside 4,096 MiB.", "Let me calculate tensor shapes per layer across 35 transformer blocks and multiply embedding dimension by vocabulary size carefully before responding..." * 3),
    ]
    pairs: list[dict[str, Any]] = []
    for i in range(20):
        q, rc, good, bad = scenarios[i % len(scenarios)]
        assert len(rc) <= 200
        q_var = f"[Case {i+1:02d}] {q} Answer in <= 2 sentences."
        pairs.append({
            "id": f"v21_adapt_{i+1:02d}",
            "bucket": "adapt_think_on_lead_concise",
            "mode": "no_tool",
            "tools": [],
            "thinking": True,
            "prompt": [{"role": "system", "content": sys_lead}, {"role": "user", "content": q_var}],
            "chosen": {"role": "assistant", "reasoning_content": rc, "content": good},
            "rejected": {"role": "assistant", "reasoning_content": "Long unfocused reasoning exceeding short thinking budget." * 4, "content": bad},
        })
    return pairs


def _gen_read_symbol_and_receipt_pairs() -> list[dict[str, Any]]:
    sys_worker = build_nc_system_prompt("coder", depth=1)
    pairs: list[dict[str, Any]] = []

    sym_cases = [
        ("crates/nc-types/src/settings.rs", "TokenUsageStats::reset", 273, "TokenUsageStats::reset"),
        ("crates/nc-agent/src/orchestrator.rs", "run_turn", 120, "AgentOrchestrator::run_turn"),
        ("crates/nc-agent/src/context.rs", "final_only_gbnf", 88, "final_only_gbnf"),
        ("crates/nc-core/src/ipc.rs", "serve_http", 45, "serve_http"),
        ("src/math.rs", "apply_tax", 4, "apply_tax"),
        ("crates/nc-agent/src/repomap.rs", "build_repo_map", 18, "build_repo_map"),
        ("crates/nc-agent/src/tools.rs", "execute_tool", 92, "execute_tool"),
        ("crates/nc-native/src/graph.rs", "expand_one_hop", 64, "expand_one_hop"),
        ("crates/nc-client/src/sse.rs", "parse_sse_chunk", 31, "parse_sse_chunk"),
        ("src/billing.rs", "compute_total", 15, "compute_total"),
    ]
    for i in range(10):
        fpath, sym, ln, query_name = sym_cases[i]
        rmap = format_repo_map_file(fpath, 320, [(sym, ln)])
        rej_tc = _tc("r1", "read_outline", {"path": fpath}) if i % 2 == 0 else _tc("r1", "read_slice", {"path": fpath, "start_line": 1, "end_line": 60})
        pairs.append({
            "id": f"v21_sym_{i+1:02d}",
            "bucket": "read_symbol_and_receipt_discipline",
            "mode": "nc_agent",
            "thinking": False,
            "prompt": [{"role": "system", "content": sys_worker}, {"role": "user", "content": build_nc_user_prompt(repo_map=rmap, task=f"Inspect `{query_name}` in `{fpath}` directly using `read_symbol`.")}],
            "chosen": {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [_tc("c1", "read_symbol", {"path": fpath, "symbol": sym})]},
            "rejected": {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [rej_tc]},
        })

    rcpt_cases = [
        ("src/math.rs", "Updated `apply_tax` to `price * 110 / 100` (`i32`) and `cargo test` passed"),
        ("web/app.js", "Added retry backoff cap and verified unit tests green"),
        ("crates/nc-agent/src/context.rs", "Adjusted budget reserve to 256 tokens; `cargo check` passed"),
        ("src/parser.rs", "Handled empty whitespace input safely without panicking"),
        ("crates/nc-client/src/stream.rs", "Cleared buffer after chunk flush; tests passing"),
    ]
    for i in range(10):
        fpath, chg = rcpt_cases[i % len(rcpt_cases)]
        rmap = format_repo_map_file(fpath, 120, [("handler", 12)])
        prompt = [
            {"role": "system", "content": sys_worker},
            {"role": "user", "content": build_nc_user_prompt(repo_map=rmap, task=f"Complete micro-task T{i+1} on `{fpath}` cleanly after verify succeeds. Emit ONLY 4-line SubTaskReceipt.")},
            {"role": "assistant", "reasoning_content": None, "content": None, "tool_calls": [_tc("c1", "run_command", {"cmd": "cargo test"})]},
            {"role": "tool", "tool_call_id": "c1", "name": "run_command", "content": "ok. 3 passed; 0 failed"},
        ]
        chosen = {"role": "assistant", "reasoning_content": None, "content": f"STATUS: DONE\nCHANGES: {chg}\nFILES: {fpath}\nISSUES: none"}
        rejected = {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": [_tc("r1", "read_slice", {"path": fpath, "start_line": 1, "end_line": 40})]}
        pairs.append({
            "id": f"v21_rcpt_{i+1:02d}",
            "bucket": "read_symbol_and_receipt_discipline",
            "mode": "nc_agent",
            "thinking": False,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate & validate 120 v2.1 Micro-ORPO pairs")
    parser.add_argument("--server-exe", type=Path, default=DEFAULT_SERVER_EXE)
    parser.add_argument("--teacher-gguf", type=Path, default=Path("brain/baselines/e4b/gemma-4-E4B-it-Q4_K_M.gguf"))
    parser.add_argument("--student-gguf", type=Path, default=Path("brain/agents/e4b_v2/gemma4-e4b-nc-Q4_K_M.gguf"))
    parser.add_argument("--out-orpo", type=Path, default=Path("data/distill/orpo_v2_1_patch.jsonl"))
    parser.add_argument("--reuse-coding", action="store_true")
    args = parser.parse_args()

    work_dir = Path("outputs/v2_1_gen")
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / "orpo_v2_1_raw.jsonl"
    if args.reuse_coding and raw_path.is_file():
        coding_pairs = [
            json.loads(ln)
            for ln in raw_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and json.loads(ln).get("bucket") == "on_policy_coding_patch"
        ]
        assert len(coding_pairs) == 50, f"Expected 50 cached coding pairs, got {len(coding_pairs)}"
        code_stats = {"reused_cached_coding_pairs": 50}
    else:
        coding_pairs, code_stats = _gen_coding_pairs(args.server_exe, args.teacher_gguf, args.student_gguf, work_dir / "logs")
    nc_pairs = _gen_live_nc_efficiency_pairs()
    adapt_pairs = _gen_adapt_think_pairs()
    disc_pairs = _gen_read_symbol_and_receipt_pairs()

    all_pairs = coding_pairs + nc_pairs + adapt_pairs + disc_pairs
    assert len(all_pairs) == 120, f"Expected 120 pairs, got {len(all_pairs)}"

    raw_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_pairs) + "\n", encoding="utf-8")

    report = convert_and_validate_openai_jsonl(
        raw_jsonl_paths=[raw_path],
        out_sft_path=work_dir / "unused_sft.jsonl",
        out_orpo_path=args.out_orpo,
        out_rejected_path=work_dir / "orpo_v2_1_rejected.jsonl",
        max_tokens=4096,
    )
    print(json.dumps({"coding_gen_stats": code_stats, "validator_report": report}, indent=2))
    assert report["kept_orpo"] == 120 and report["rejected_count"] == 0, "Validation failed!"

    orpo_rows = [json.loads(ln) for ln in args.out_orpo.read_text(encoding="utf-8").splitlines() if ln.strip()]
    bad_turn = sum(1 for r in orpo_rows if not r["chosen"].endswith("<turn|>\n"))
    assert bad_turn == 0, f"{bad_turn} chosen rows do not end with <turn|>\\n"
    print(f"Verified 120/120 v2.1 ORPO pairs -> {args.out_orpo} (all ending cleanly with <turn|>\\n)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

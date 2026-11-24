"""Runtime metrics, nvidia-smi VRAM monitoring, llama-server lifecycle, and SSE streaming telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import http.client
import json
import math
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import Any, Iterable
import urllib.error
import urllib.request

DEFAULT_PLE_CPU_OFFLOAD = r"per_layer_token_embd\.weight=CPU"


def query_nvidia_vram_mib() -> dict[str, int] | None:
    """Query current GPU used/total/free VRAM in MiB via nvidia-smi."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        return {
            "used_mib": int(parts[0]),
            "total_mib": int(parts[1]),
            "free_mib": int(parts[2]),
        }
    except ValueError:
        return None


class VramMonitor:
    """Background VRAM sampler polling nvidia-smi during model load and inference."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.samples_mib: list[int] = []
        self.idle_mib: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VramMonitor":
        snap = query_nvidia_vram_mib()
        if snap is not None:
            self.idle_mib = snap["used_mib"]
            self.samples_mib.append(snap["used_mib"])
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            snap = query_nvidia_vram_mib()
            if snap is not None:
                self.samples_mib.append(snap["used_mib"])

    def sample_now(self) -> int | None:
        snap = query_nvidia_vram_mib()
        if snap is None:
            return None
        self.samples_mib.append(snap["used_mib"])
        return snap["used_mib"]

    @property
    def peak_mib(self) -> int:
        return max(self.samples_mib) if self.samples_mib else 0

    @property
    def peak_delta_mib(self) -> int:
        return max(0, self.peak_mib - self.idle_mib)

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


@dataclass
class CompletionTelemetry:
    """Full verbose per-request latency, throughput, and token metrics."""

    ok: bool
    status_code: int = 200
    ttfb_ms: float | None = None
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    pp_tps: float | None = None
    tg_tps: float | None = None
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_tool_call_delta(
    acc: dict[int, dict[str, Any]],
    deltas: list[dict[str, Any]],
) -> None:
    for item in deltas:
        idx = int(item.get("index", 0))
        slot = acc.setdefault(
            idx,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if item.get("id"):
            slot["id"] = str(item["id"])
        if item.get("type"):
            slot["type"] = str(item["type"])
        fn = item.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += str(fn["name"])
        if fn.get("arguments") is not None:
            slot["function"]["arguments"] += str(fn["arguments"])


def parse_sse_events(
    lines: Iterable[bytes | str],
    *,
    request_start_perf: float | None = None,
    clock_fn: Any | None = None,
) -> CompletionTelemetry:
    """Parse OpenAI-compatible llama-server SSE stream lines into CompletionTelemetry."""
    clock = clock_fn or time.perf_counter
    t0 = request_start_perf if request_start_perf is not None else clock()
    ttfb_ms: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            evt = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if isinstance(evt, dict) and "error" in evt:
            err_obj = evt["error"]
            msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
            return CompletionTelemetry(
                ok=False,
                status_code=int(err_obj.get("code", 500)) if isinstance(err_obj, dict) and str(err_obj.get("code", "")).isdigit() else 500,
                ttfb_ms=ttfb_ms,
                total_ms=(clock() - t0) * 1000.0,
                content="".join(content_parts),
                reasoning_content="".join(reasoning_parts),
                error=msg,
            )

        if isinstance(evt.get("usage"), dict):
            usage.update(evt["usage"])
        if isinstance(evt.get("timings"), dict):
            timings.update(evt["timings"])

        choices = evt.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            continue

        rc = delta.get("reasoning_content")
        ct = delta.get("content")
        tcs = delta.get("tool_calls")
        has_token_output = bool(rc or ct or tcs)
        if has_token_output and ttfb_ms is None:
            ttfb_ms = (clock() - t0) * 1000.0

        if rc:
            reasoning_parts.append(str(rc))
        if ct:
            content_parts.append(str(ct))
        if isinstance(tcs, list) and tcs:
            _merge_tool_call_delta(tool_acc, tcs)

    total_ms = (clock() - t0) * 1000.0
    comp_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}

    prompt_tokens = int(
        usage.get("prompt_tokens")
        or ((timings.get("prompt_n") or 0) + (timings.get("cache_n") or 0))
        or 0
    )
    completion_tokens = int(
        usage.get("completion_tokens") or timings.get("predicted_n") or 0
    )
    reasoning_text = "".join(reasoning_parts)
    reasoning_tokens = int(comp_details.get("reasoning_tokens") or 0)
    if reasoning_tokens == 0 and reasoning_text:
        reasoning_tokens = max(1, (len(reasoning_text) + 3) // 4)
    cached_tokens = int(
        prompt_details.get("cached_tokens") or timings.get("cache_n") or 0
    )
    pp_tps = (
        float(timings["prompt_per_second"])
        if timings.get("prompt_per_second") is not None
        else None
    )
    tg_tps = (
        float(timings["predicted_per_second"])
        if timings.get("predicted_per_second") is not None
        else None
    )

    tool_calls = [tool_acc[k] for k in sorted(tool_acc)]
    return CompletionTelemetry(
        ok=True,
        status_code=200,
        ttfb_ms=round(ttfb_ms, 2) if ttfb_ms is not None else round(total_ms, 2),
        total_ms=round(total_ms, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        pp_tps=round(pp_tps, 2) if pp_tps is not None else None,
        tg_tps=round(tg_tps, 2) if tg_tps is not None else None,
        content="".join(content_parts),
        reasoning_content=reasoning_text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


def stream_chat_completion(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    timeout_s: float = 180.0,
) -> CompletionTelemetry:
    """Send streaming /v1/chat/completions to llama-server and capture full verbose metrics."""
    body_obj = dict(payload)
    body_obj["stream"] = True
    body_obj.setdefault("stream_options", {"include_usage": True})
    body = json.dumps(body_obj).encode("utf-8")

    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    t0 = time.perf_counter()
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            raw_err = resp.read().decode("utf-8", errors="replace")
            return CompletionTelemetry(
                ok=False,
                status_code=resp.status,
                total_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                error=f"HTTP {resp.status}: {raw_err[:500]}",
            )
        return parse_sse_events(resp, request_start_perf=t0)
    except Exception as exc:  # noqa: BLE001
        return CompletionTelemetry(
            ok=False,
            status_code=0,
            total_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            error=str(exc),
        )
    finally:
        conn.close()


def aggregate_telemetry(records: list[dict[str, Any] | CompletionTelemetry]) -> dict[str, Any]:
    """Summarize request telemetry (tokens, TTFB, throughput, failure rate)."""
    dicts = [r.to_dict() if isinstance(r, CompletionTelemetry) else dict(r) for r in records]
    ok_recs = [r for r in dicts if r.get("ok")]
    ttfbs = sorted(float(r["ttfb_ms"]) for r in ok_recs if r.get("ttfb_ms") is not None)
    totals = [float(r["total_ms"]) for r in ok_recs if r.get("total_ms") is not None]
    pps = [float(r["pp_tps"]) for r in ok_recs if r.get("pp_tps") is not None]
    tgs = [float(r["tg_tps"]) for r in ok_recs if r.get("tg_tps") is not None]

    def _p95(vals: list[float]) -> float | None:
        if not vals:
            return None
        idx = max(0, math.ceil(0.95 * len(vals)) - 1)
        return round(vals[idx], 2)

    return {
        "requests_total": len(dicts),
        "requests_ok": len(ok_recs),
        "requests_failed": len(dicts) - len(ok_recs),
        "prompt_tokens_total": sum(int(r.get("prompt_tokens") or 0) for r in ok_recs),
        "completion_tokens_total": sum(int(r.get("completion_tokens") or 0) for r in ok_recs),
        "reasoning_tokens_total": sum(int(r.get("reasoning_tokens") or 0) for r in ok_recs),
        "cached_tokens_total": sum(int(r.get("cached_tokens") or 0) for r in ok_recs),
        "avg_ttfb_ms": round(sum(ttfbs) / len(ttfbs), 2) if ttfbs else None,
        "p95_ttfb_ms": _p95(ttfbs),
        "avg_total_ms": round(sum(totals) / len(totals), 2) if totals else None,
        "avg_pp_tps": round(sum(pps) / len(pps), 2) if pps else None,
        "avg_tg_tps": round(sum(tgs) / len(tgs), 2) if tgs else None,
    }


def find_free_port() -> int:
    """Allocate a free ephemeral TCP loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _terminate_proc(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


class LlamaServerSession:
    """Managed llama-server subprocess with adaptive Vulkan VRAM fallback and VRAM telemetry."""

    def __init__(
        self,
        *,
        server_exe: Path,
        model_path: Path,
        ctx_size: int = 8192,
        ngl_candidates: tuple[int, ...] = (99, 28, 20, 12, 0),
        mmproj_path: Path | None = None,
        ple_cpu_offload: bool = True,
        threads: int = 8,
        startup_timeout_s: float = 180.0,
        log_dir: Path | None = None,
    ) -> None:
        self.server_exe = Path(server_exe).resolve()
        self.model_path = Path(model_path).resolve()
        self.ctx_size = ctx_size
        self.ngl_candidates = ngl_candidates
        self.mmproj_path = Path(mmproj_path).resolve() if mmproj_path is not None else None
        self.ple_cpu_offload = ple_cpu_offload
        self.threads = threads
        self.startup_timeout_s = startup_timeout_s
        self.log_dir = Path(log_dir).resolve() if log_dir is not None else None

        self.host = "127.0.0.1"
        self.port = 0
        self.proc: subprocess.Popen[bytes] | None = None
        self.ngl_used: int | None = None
        self.mmproj_offload_used: bool | None = None
        self.boot_ms: float = 0.0
        self.idle_vram_mib: int = 0
        self.loaded_vram_mib: int = 0
        self.peak_vram_mib: int = 0
        self._vram_monitor: VramMonitor | None = None
        self._log_file: Any = None

    def _attempt_configs(self) -> list[tuple[int, bool]]:
        if self.mmproj_path is None:
            return [(ngl, False) for ngl in self.ngl_candidates]
        configs: list[tuple[int, bool]] = []
        for mm_offload in (True, False):
            for ngl in self.ngl_candidates:
                configs.append((ngl, mm_offload))
        return configs

    def __enter__(self) -> "LlamaServerSession":
        self._vram_monitor = VramMonitor(interval_s=0.25)
        self._vram_monitor.__enter__()
        self.idle_vram_mib = self._vram_monitor.idle_mib

        errors: list[str] = []
        for ngl, mm_offload in self._attempt_configs():
            port = find_free_port()
            cmd = [
                str(self.server_exe),
                "-m",
                str(self.model_path),
                "--host",
                self.host,
                "--port",
                str(port),
                "-c",
                str(self.ctx_size),
                "-ngl",
                str(ngl),
                "-t",
                str(self.threads),
                "-fa",
                "on",
                "-ctk",
                "q8_0",
                "-ctv",
                "q8_0",
                "--cache-reuse",
                "256",
                "--jinja",
                "--reasoning",
                "auto",
                "--reasoning-format",
                "deepseek",
            ]
            if self.ple_cpu_offload:
                cmd.extend(["-ot", DEFAULT_PLE_CPU_OFFLOAD])
            if self.mmproj_path is not None:
                cmd.extend(["--mmproj", str(self.mmproj_path)])
                if not mm_offload:
                    cmd.append("--no-mmproj-offload")

            log_fh = subprocess.DEVNULL
            log_path: Path | None = None
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                mm_tag = f"_mm{int(mm_offload)}" if self.mmproj_path else ""
                log_path = self.log_dir / f"llama_{self.model_path.stem}_ngl{ngl}{mm_tag}.log"
                log_fh = log_path.open("w", encoding="utf-8", errors="replace")

            t0 = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.server_exe.parent),
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )

            health_url = f"http://{self.host}:{port}/health"
            deadline = time.monotonic() + self.startup_timeout_s
            ready = False
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(health_url, timeout=2.0) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                time.sleep(0.35)

            if ready:
                self.proc = proc
                self.port = port
                self.ngl_used = ngl
                self.mmproj_offload_used = mm_offload if self.mmproj_path else None
                self.boot_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                self._log_file = log_fh if log_fh is not subprocess.DEVNULL else None
                loaded = self._vram_monitor.sample_now()
                self.loaded_vram_mib = loaded if loaded is not None else self._vram_monitor.peak_mib
                return self

            _terminate_proc(proc)
            if log_fh is not subprocess.DEVNULL:
                log_fh.close()
            tail = ""
            if log_path is not None and log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = " | ".join(lines[-8:])
            errors.append(f"ngl={ngl},mm_offload={mm_offload}: {tail[-280:]}")
            time.sleep(0.5)

        self._vram_monitor.__exit__(None, None, None)
        self._vram_monitor = None
        raise RuntimeError(
            f"llama-server failed to start for {self.model_path.name}: {'; '.join(errors)}"
        )

    def current_peak_vram_mib(self) -> int:
        if self._vram_monitor is not None:
            self._vram_monitor.sample_now()
            self.peak_vram_mib = self._vram_monitor.peak_mib
        return self.peak_vram_mib

    def __exit__(self, *_exc: Any) -> None:
        _terminate_proc(self.proc)
        self.proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._vram_monitor is not None:
            self._vram_monitor.__exit__(None, None, None)
            self.peak_vram_mib = self._vram_monitor.peak_mib
            self._vram_monitor = None

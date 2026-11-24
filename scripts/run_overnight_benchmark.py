"""CLI entrypoint for overnight asset prep, model pulls, benchmarking, and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.probes import ensure_multimodal_assets  # noqa: E402
from eval.suite import (  # noqa: E402
    MODEL_SPECS,
    aggregate_results,
    download_agent_artifacts,
    download_baseline_artifacts,
    refresh_live_nc_for_variant,
    run_benchmark_for_variant,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Overnight benchmark & model pull CLI")
    parser.add_argument("--work-root", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--brain-dir", type=Path, default=Path("brain"))
    parser.add_argument("--prep-assets", action="store_true")
    parser.add_argument("--download-baselines", action="store_true")
    parser.add_argument("--download-agent", choices=("e2b", "e4b"), nargs="+")
    parser.add_argument("--run", choices=tuple(MODEL_SPECS.keys()), nargs="+")
    parser.add_argument("--quants", choices=("IQ4_XS", "Q4_K_M"), nargs="+", default=["IQ4_XS", "Q4_K_M"])
    parser.add_argument("--skip-live-nc", action="store_true")
    parser.add_argument("--refresh-live-nc", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--report-out", type=Path, default=Path("outputs/benchmarks/overnight_benchmark_report.md"))
    args = parser.parse_args()

    if args.prep_assets:
        assets = ensure_multimodal_assets(args.work_root / "assets")
        print(json.dumps({k: str(v) for k, v in assets.items()}, indent=2))

    if args.download_baselines:
        res = download_baseline_artifacts(brain_dir=args.brain_dir, quants=tuple(args.quants))
        print(json.dumps(res, indent=2))

    if args.download_agent:
        for var in args.download_agent:
            res = download_agent_artifacts(var, brain_dir=args.brain_dir, quants=tuple(args.quants))
            print(json.dumps(res, indent=2))

    if args.run:
        for model_key in args.run:
            for quant in args.quants:
                if args.refresh_live_nc:
                    refresh_live_nc_for_variant(
                        model_key=model_key,
                        quant=quant,
                        work_root=args.work_root,
                        brain_dir=args.brain_dir,
                    )
                else:
                    run_benchmark_for_variant(
                        model_key=model_key,
                        quant=quant,
                        work_root=args.work_root,
                        brain_dir=args.brain_dir,
                        run_live_nc=not args.skip_live_nc,
                        force=args.force,
                    )

    if args.aggregate:
        agg = aggregate_results(args.work_root / "results", args.report_out)
        print(json.dumps(agg, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

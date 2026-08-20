#!/usr/bin/env python3
"""Score model predictions in one declared evaluation view."""

from __future__ import annotations

import argparse

from colregs_framework.benchmark import BenchmarkRelease
from colregs_framework.io import iter_jsonl
from colregs_framework.scoring import score_predictions, write_score_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root")
    parser.add_argument("predictions")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--view",
        default="expert_corrected_core3",
        choices=("full_structured", "expert_corrected_core3"),
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--run-id", default="public_evaluation")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    args = parser.parse_args()

    output_scope = (
        "core_three_v1"
        if args.view == "expert_corrected_core3"
        else "gpt_fullfields_six_v1"
    )
    release = BenchmarkRelease(args.benchmark_root)
    bundle = score_predictions(
        release.references(args.split, args.view),
        list(iter_jsonl(args.predictions)),
        run_id=args.run_id,
        output_scope=output_scope,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    hashes = write_score_artifacts(bundle, args.output_dir)
    print(bundle.metrics)
    print(hashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the label-blind Symbolic COLREGs Rule Engine on a public split."""

from __future__ import annotations

import argparse
from pathlib import Path

from colregs_framework.benchmark import BenchmarkRelease
from colregs_framework.io import write_jsonl_atomic
from colregs_framework.symbolic_engine import run_symbolic_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root")
    parser.add_argument("output")
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    release = BenchmarkRelease(args.benchmark_root)
    rules = release.root / "rules" / "colregs_atomic.yaml"
    params = release.root / "rules" / "operational_params.yaml"
    priorities = release.root / "rules" / "priorities.yaml"
    predictions = [
        run_symbolic_sample(
            row,
            rules_path=rules,
            params_path=params,
            priorities_path=priorities,
        )
        for row in release.inputs(args.split)
    ]
    write_jsonl_atomic(Path(args.output), predictions, overwrite=args.overwrite)
    print(f"wrote {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

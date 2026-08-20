#!/usr/bin/env python3
"""Print a compact view of one benchmark item and one chosen reference view."""

from __future__ import annotations

import argparse
import json

from colregs_framework.benchmark import BenchmarkRelease


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument(
        "--view",
        default="expert_corrected_core3",
        choices=("full_structured", "expert_corrected_core3"),
    )
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    row = BenchmarkRelease(args.benchmark_root).joined(args.split, args.view)[args.index]
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

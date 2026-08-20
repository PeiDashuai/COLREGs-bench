#!/usr/bin/env python3
"""Validate the public benchmark release without model or API calls."""

from __future__ import annotations

import argparse
import json

from colregs_framework.benchmark import BenchmarkRelease


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    report = BenchmarkRelease(args.benchmark_root).validate(
        check_images=not args.skip_images
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

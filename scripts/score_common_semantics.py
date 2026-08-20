#!/usr/bin/env python3
"""Score predictions after both sides have been projected to common semantics."""

from __future__ import annotations

import argparse

from colregs_framework.io import iter_jsonl
from colregs_framework.symbolic_scoring import (
    score_common_semantics,
    write_common_score_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="common_semantic_evaluation")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20263816)
    args = parser.parse_args()
    bundle = score_common_semantics(
        list(iter_jsonl(args.gold)),
        list(iter_jsonl(args.prediction)),
        run_id=args.run_id,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(write_common_score_artifacts(bundle, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

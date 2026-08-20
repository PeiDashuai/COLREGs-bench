#!/usr/bin/env python3
"""Project benchmark, model, or symbolic outputs to the 39 common concepts."""

from __future__ import annotations

import argparse
from pathlib import Path

from jsonschema import Draft202012Validator

from colregs_framework.io import iter_jsonl, read_json, write_jsonl_atomic
from colregs_framework.semantic_projection import (
    assert_expert_review_complete,
    load_mapping,
    load_ontology,
    project_benchmark_row,
    project_llm_row,
    project_symbolic_row,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-space", choices=("benchmark", "symbolic_engine", "llm"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ontology = load_ontology(args.ontology)
    mapping = load_mapping(args.mapping, ontology)
    assert_expert_review_complete(mapping)
    projector = {
        "benchmark": project_benchmark_row,
        "symbolic_engine": project_symbolic_row,
        "llm": project_llm_row,
    }[args.source_space]
    validator = Draft202012Validator(read_json(args.schema))
    rows = []
    for row in iter_jsonl(args.input):
        projected = projector(row, mapping, ontology)
        validator.validate(projected)
        rows.append(projected)
    write_jsonl_atomic(args.output, rows, overwrite=args.overwrite)
    print(f"wrote {len(rows)} projected rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

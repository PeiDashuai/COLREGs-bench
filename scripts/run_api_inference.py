#!/usr/bin/env python3
"""Run traceable API inference without storing credentials in code or output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from colregs_framework.benchmark import BenchmarkRelease
from colregs_framework.inference_backend import OpenAIBackend
from colregs_framework.io import read_json, write_jsonl_atomic
from colregs_framework.parsing import parse_prediction_text
from colregs_framework.prompting import (
    build_core_three_user_prompt,
    build_fullfields_user_prompt,
)
from colregs_framework.protocols import InferenceInput


def main() -> int:
    code_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root")
    parser.add_argument("output")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--scope", default="core3", choices=("core3", "full"))
    parser.add_argument("--modality", default="text", choices=("text", "multimodal"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API credential in environment variable {args.api_key_env}")
    release = BenchmarkRelease(args.benchmark_root)
    rows = release.inputs(args.split)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        rows = rows[: args.limit]

    if args.scope == "core3":
        prompt = (code_root / "prompts" / "core_three_eval_v1.txt").read_text(encoding="utf-8").rstrip()
        schema = read_json(code_root / "schemas" / "core_three_prediction_v1.schema.json")
        vocabulary = read_json(code_root / "vocabularies" / "core_three_v2.json")
        output_scope = "core_three_v1"
    else:
        prompt = (code_root / "prompts" / "gpt_fullfields_v2.txt").read_text(encoding="utf-8").rstrip()
        schema = read_json(code_root / "schemas" / "gpt_fullfields_prediction_v2.schema.json")
        vocabulary = read_json(code_root / "vocabularies" / "fullfields_v2.json")
        output_scope = "gpt_fullfields_six_v1"

    decode = {
        "image_detail": "high",
        "max_output_tokens": 4096,
        "max_retries_per_sample": 3,
        "reasoning_effort": "low",
        "request_timeout_seconds": 180,
        "retry_backoff_seconds": [1, 2, 4],
        "service_tier": "auto",
        "store": False,
        "temperature": None,
    }
    backend = OpenAIBackend(
        args.model,
        decode_profile=decode,
        response_schema=schema,
        api_key=api_key,
        base_url=args.base_url,
    )
    output_rows = []
    for row in rows:
        payload = {"family": row["pattern"], "scene_spec": row["inputs"]["scene_spec"]}
        user_prompt = (
            build_core_three_user_prompt(payload, vocabulary["fields"])
            if args.scope == "core3"
            else build_fullfields_user_prompt(payload, vocabulary)
        )
        image_paths = ()
        if args.modality == "multimodal":
            image_paths = (
                release.image_path(row["topdown_image"]),
                release.image_path(row["radar_image"]),
            )
        response = backend(
            InferenceInput(
                sample_id=row["sample_id"],
                pattern=row["pattern"],
                system_prompt=prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
            )
        )
        parsed = parse_prediction_text(
            response.output_text,
            output_scope=output_scope,
            schema=schema,
            vocabulary=vocabulary,
        )
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "pattern": row["pattern"],
                "parse_status": parsed.status,
                "parse_repair": parsed.repair,
                "parse_error": parsed.error,
                "prediction": parsed.prediction,
                "requested_model": args.model,
                "returned_model": response.returned_model,
                "request_attempts": list(response.request_attempts),
            }
        )
    write_jsonl_atomic(args.output, output_rows, overwrite=args.overwrite)
    print(json.dumps({"rows": len(output_rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

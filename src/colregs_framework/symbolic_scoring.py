"""Scoring and paired comparisons in the frozen common semantic space."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from .hashing import sha256_file
from .io import write_json_atomic, write_jsonl_atomic
from .scoring import ScoringError, bootstrap_mean_ci, paired_sign_flip, set_metrics


COMMON_HEADS = ("rule_articles", "maneuver_allowed", "maneuver_forbidden")


def _index(rows: Iterable[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ScoringError(f"{name} row lacks sample_id")
        if sample_id in result:
            raise ScoringError(f"{name} has duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def score_common_semantics(
    gold_rows: Iterable[Mapping[str, Any]],
    prediction_rows: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260811,
) -> dict[str, Any]:
    gold = _index(gold_rows, "gold projection")
    prediction = _index(prediction_rows, "prediction projection")
    if tuple(gold) != tuple(prediction):
        missing = sorted(set(gold) - set(prediction))
        extra = sorted(set(prediction) - set(gold))
        raise ScoringError(f"projection identity/order mismatch; missing={missing[:5]}, extra={extra[:5]}")
    field_vectors: dict[str, list[float]] = defaultdict(list)
    field_exact: dict[str, list[float]] = defaultdict(list)
    family_vectors: dict[str, list[float]] = defaultdict(list)
    per_sample: list[dict[str, Any]] = []
    unsupported_gold = 0
    unsupported_prediction = 0
    source_gold = 0
    source_prediction = 0

    for sample_id, gold_row in gold.items():
        prediction_row = prediction[sample_id]
        gold_payload = gold_row.get("common_semantics")
        predicted_payload = prediction_row.get("common_semantics")
        if not isinstance(gold_payload, Mapping) or not isinstance(predicted_payload, Mapping):
            raise ScoringError(f"common semantics missing for {sample_id}")
        scores = {
            head: set_metrics(predicted_payload.get(head, []), gold_payload.get(head, []))
            for head in COMMON_HEADS
        }
        for head, item in scores.items():
            field_vectors[head].append(float(item["f1"]))
            field_exact[head].append(float(item["exact"]))
        sample_mean = mean(item["f1"] for item in scores.values())
        family = str(gold_row.get("pattern") or "unknown")
        family_vectors[family].append(sample_mean)
        gold_coverage = gold_row.get("coverage", {})
        prediction_coverage = prediction_row.get("coverage", {})
        source_gold += int(gold_coverage.get("source_token_count", 0))
        source_prediction += int(prediction_coverage.get("source_token_count", 0))
        unsupported_gold += len(gold_row.get("unsupported", []))
        unsupported_prediction += len(prediction_row.get("unsupported", []))
        per_sample.append(
            {
                "run_id": run_id,
                "sample_id": sample_id,
                "pattern": family,
                "scores": scores,
                "mean_head_f1": sample_mean,
                "all_heads_exact": float(all(item["exact"] == 1.0 for item in scores.values())),
                "gold_unsupported_count": len(gold_row.get("unsupported", [])),
                "prediction_unsupported_count": len(prediction_row.get("unsupported", [])),
            }
        )

    fields = {
        head: {
            "count": len(field_vectors[head]),
            "f1": mean(field_vectors[head]),
            "exact_rate": mean(field_exact[head]),
            "bootstrap_95_ci": bootstrap_mean_ci(
                field_vectors[head],
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=bootstrap_seed + index,
            ),
        }
        for index, head in enumerate(COMMON_HEADS)
    }
    sample_vector = [row["mean_head_f1"] for row in per_sample]
    metrics = {
        "schema_version": 1,
        "run_id": run_id,
        "sample_count": len(per_sample),
        "primary_mean_common_head_f1": mean(item["f1"] for item in fields.values()),
        "sample_mean_common_head_f1": mean(sample_vector),
        "sample_mean_bootstrap_95_ci": bootstrap_mean_ci(
            sample_vector,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=bootstrap_seed + 100,
        ),
        "all_heads_exact_rate": mean(item["all_heads_exact"] for item in per_sample),
        "field_aggregate": fields,
        "mapping_coverage": {
            "gold_unsupported_source_tokens": unsupported_gold,
            "gold_source_tokens": source_gold,
            "gold_coverage_rate": 1.0 - unsupported_gold / source_gold if source_gold else 1.0,
            "prediction_unsupported_source_tokens": unsupported_prediction,
            "prediction_source_tokens": source_prediction,
            "prediction_coverage_rate": 1.0 - unsupported_prediction / source_prediction if source_prediction else 1.0,
        },
        "statistics": {
            "bootstrap_unit": "sample",
            "bootstrap_replicates": bootstrap_replicates,
            "confidence_level": confidence_level,
            "bootstrap_seed": bootstrap_seed,
        },
    }
    by_family = {
        family: {
            "n": len(values),
            "mean_common_head_f1": mean(values),
            "bootstrap_95_ci": bootstrap_mean_ci(
                values,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=bootstrap_seed + 1000 + index,
            ),
        }
        for index, (family, values) in enumerate(sorted(family_vectors.items()))
    }
    return {"metrics": metrics, "metrics_by_family": by_family, "per_sample": per_sample}


def paired_common_comparison(
    baseline_per_sample: Iterable[Mapping[str, Any]],
    comparator_per_sample: Iterable[Mapping[str, Any]],
    *,
    baseline_name: str,
    comparator_name: str,
    replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260811,
) -> dict[str, Any]:
    baseline = _index(baseline_per_sample, baseline_name)
    comparator = _index(comparator_per_sample, comparator_name)
    if tuple(baseline) != tuple(comparator):
        raise ScoringError("paired comparison requires identical ordered sample IDs")
    left = [float(row["mean_head_f1"]) for row in baseline.values()]
    right = [float(comparator[sample_id]["mean_head_f1"]) for sample_id in baseline]
    return {
        "baseline": baseline_name,
        "comparator": comparator_name,
        "sample_count": len(left),
        **paired_sign_flip(
            left,
            right,
            replicates=replicates,
            confidence_level=confidence_level,
            seed=seed,
        ),
    }


def write_common_score_artifacts(bundle: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    write_json_atomic(root / "metrics.json", bundle["metrics"])
    write_json_atomic(root / "metrics_by_family.json", bundle["metrics_by_family"])
    write_jsonl_atomic(root / "metrics_per_sample.jsonl", bundle["per_sample"])
    return {
        "metrics_sha256": sha256_file(root / "metrics.json"),
        "metrics_by_family_sha256": sha256_file(root / "metrics_by_family.json"),
        "metrics_per_sample_sha256": sha256_file(root / "metrics_per_sample.jsonl"),
    }

"""One deterministic scorer for core-three and retained six-head predictions."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .hashing import sha256_file, sha256_json
from .io import write_json_atomic, write_jsonl_atomic
from .parsing import CORE_FIELDS, FULLFIELDS


class ScoringError(ValueError):
    """Raised when gold/prediction identity or structure is not scoreable."""


@dataclass(frozen=True)
class ScoreBundle:
    metrics: dict[str, Any]
    metrics_by_family: dict[str, Any]
    per_sample: tuple[dict[str, Any], ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _index_unique(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ScoringError(f"{label} row lacks sample_id")
        if sample_id in result:
            raise ScoringError(f"{label} contains duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _prediction_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get("prediction")
    return nested if isinstance(nested, Mapping) else row


def set_metrics(predicted: Iterable[Any], gold: Iterable[Any]) -> dict[str, Any]:
    predicted_set = {str(item) for item in predicted}
    gold_set = {str(item) for item in gold}
    true_positive = predicted_set & gold_set
    false_positive = predicted_set - gold_set
    false_negative = gold_set - predicted_set
    if not predicted_set and not gold_set:
        precision = recall = f1 = 1.0
    else:
        precision = len(true_positive) / len(predicted_set) if predicted_set else 0.0
        recall = len(true_positive) / len(gold_set) if gold_set else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": float(predicted_set == gold_set),
        "false_positive": sorted(false_positive),
        "false_negative": sorted(false_negative),
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ScoringError("cannot compute a percentile of an empty vector")
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def bootstrap_mean_ci(
    values: Sequence[float], *, replicates: int, confidence_level: float, seed: int
) -> list[float]:
    if not values or replicates <= 0 or not 0.0 < confidence_level < 1.0:
        raise ScoringError("invalid bootstrap configuration")
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates)
    )
    alpha = (1.0 - confidence_level) / 2.0
    return [_percentile(estimates, alpha), _percentile(estimates, 1.0 - alpha)]


def paired_sign_flip(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if len(baseline) != len(treatment) or not baseline:
        raise ScoringError("paired score vectors must be non-empty and equal length")
    deltas = [right - left for left, right in zip(baseline, treatment)]
    observed = mean(deltas)
    rng = random.Random(seed)
    n = len(deltas)
    bootstrap = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates)
    )
    extreme = 0
    for _ in range(replicates):
        permuted = mean(delta if rng.random() < 0.5 else -delta for delta in deltas)
        if abs(permuted) >= abs(observed) - 1e-15:
            extreme += 1
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "mean_delta": observed,
        "bootstrap_confidence_interval": [
            _percentile(bootstrap, alpha),
            _percentile(bootstrap, 1.0 - alpha),
        ],
        "paired_sign_flip_p_two_sided": (extreme + 1) / (replicates + 1),
        "replicates": replicates,
    }


def _bucket_cpa(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    if value < 200:
        return "<200"
    if value < 500:
        return "200-500"
    if value < 1000:
        return "500-1000"
    return ">=1000"


def _bucket_tcpa(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    if value < 60:
        return "<60"
    if value < 180:
        return "60-180"
    if value < 480:
        return "180-480"
    return ">=480"


def _target_index(value: Any) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    if not isinstance(value, list):
        return result
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        target_id = item.get("target_id")
        key = target_id if isinstance(target_id, str) and target_id else f"__idx_{index}"
        result[key] = item
    return result


def _target_components(value: Any, family: str) -> tuple[list[str], list[str], list[str]]:
    symbolic: list[str] = []
    numeric: list[str] = []
    rules: list[str] = []
    role_key = (
        "target_role"
        if family == "restricted_multi"
        else "intended_role" if family == "channel_crossing_impede" else None
    )
    for target_id, item in _target_index(value).items():
        symbol = {
            "target_id": target_id,
            "relative_bearing_sector": item.get("relative_bearing_sector"),
            "closing": item.get("closing"),
            "risk_of_collision": item.get("risk_of_collision"),
            "collision_imminent": item.get("collision_imminent"),
        }
        if role_key:
            symbol[role_key] = item.get(role_key)
        symbolic.append(_canonical_json(symbol))
        numeric.append(
            _canonical_json(
                {
                    "target_id": target_id,
                    "cpa_bucket": _bucket_cpa(item.get("cpa_m")),
                    "tcpa_bucket": _bucket_tcpa(item.get("tcpa_s")),
                }
            )
        )
        triggered = item.get("triggered_rule_ids")
        if isinstance(triggered, list):
            rules.extend(
                f"{target_id}|{rule}"
                for rule in sorted({rule for rule in triggered if isinstance(rule, str)})
            )
    return sorted(set(symbolic)), sorted(set(numeric)), sorted(set(rules))


def _score_target_summaries(gold: Any, predicted: Any, family: str) -> dict[str, Any]:
    gold_parts = _target_components(gold, family)
    predicted_parts = _target_components(predicted, family)
    names = ("symbolic", "numeric_bucket", "rule_attribution")
    components = {
        name: set_metrics(predicted_part, gold_part)
        for name, gold_part, predicted_part in zip(names, gold_parts, predicted_parts)
    }
    return {
        "f1": mean(item["f1"] for item in components.values()),
        "exact": float(
            sorted(_canonical_json(item) for item in predicted or [])
            == sorted(_canonical_json(item) for item in gold or [])
        ),
        "diagnostics": {name: item["f1"] for name, item in components.items()},
    }


def _ledger_components(value: Any, family: str) -> tuple[list[str], list[str]]:
    if not isinstance(value, Mapping):
        return [], []
    core: list[str] = []
    attribution: list[str] = []
    maneuver = value.get("maneuver")
    if family == "restricted_multi" and isinstance(maneuver, Mapping):
        for bucket in ("allowed", "forbidden"):
            entries = maneuver.get(bucket, [])
            if not isinstance(entries, list):
                continue
            for item in entries:
                if not isinstance(item, Mapping):
                    continue
                primitive = item.get("primitive")
                core.append(_canonical_json({"head": "maneuver", "bucket": bucket, "primitive": primitive}))
                attribution.extend(
                    f"{bucket}|{primitive}|{rule}"
                    for rule in item.get("support_rules", [])
                    if isinstance(rule, str)
                )
        summary = value.get("summary")
        if isinstance(summary, Mapping):
            core.append(_canonical_json({"head": "summary"}))
            attribution.extend(
                f"summary|{rule}"
                for rule in summary.get("support_rules", [])
                if isinstance(rule, str)
            )
        suppression = value.get("suppression")
        if isinstance(suppression, Mapping):
            for item in suppression.get("records", []):
                if isinstance(item, Mapping):
                    core.append(
                        _canonical_json(
                            {
                                "head": "suppression",
                                "suppressed_primitive": item.get("suppressed_primitive"),
                            }
                        )
                    )
    else:
        entries: list[Mapping[str, Any]] = []
        if isinstance(maneuver, list):
            entries = [item for item in maneuver if isinstance(item, Mapping)]
        elif isinstance(maneuver, Mapping):
            entries = [
                {"primitive": primitive, **dict(details)}
                for primitive, details in maneuver.items()
                if isinstance(details, Mapping)
            ]
        if family == "channel_crossing_impede" and isinstance(
            value.get("family_contract"), Mapping
        ):
            for key, item in sorted(value["family_contract"].items()):
                core.append(
                    _canonical_json({"head": "family_contract", "key": key, "value": item})
                )
        for item in entries:
            primitive = item.get("primitive")
            core.append(_canonical_json({"head": "maneuver", "primitive": primitive}))
            attribution.extend(
                f"maneuver|{primitive}|{rule}"
                for rule in item.get("support_rules", [])
                if isinstance(rule, str)
            )
    return sorted(set(core)), sorted(set(attribution))


def _score_ledger(gold: Any, predicted: Any, family: str) -> dict[str, Any]:
    gold_core, gold_attribution = _ledger_components(gold, family)
    pred_core, pred_attribution = _ledger_components(predicted, family)
    core = set_metrics(pred_core, gold_core)
    attribution = set_metrics(pred_attribution, gold_attribution)
    return {
        "f1": mean((core["f1"], attribution["f1"])),
        "exact": float(
            _canonical_json(predicted or {}) == _canonical_json(gold or {})
        ),
        "diagnostics": {
            "core_structure_f1": core["f1"],
            "support_rule_attribution_f1": attribution["f1"],
        },
    }


def _suppression_components(value: Any) -> tuple[list[str], list[str]]:
    core: list[str] = []
    metadata: list[str] = []
    if not isinstance(value, list):
        return core, metadata
    for item in value:
        if not isinstance(item, Mapping):
            continue
        core.append(
            _canonical_json(
                {
                    "suppressed_primitive": item.get("suppressed_primitive"),
                    "suppressed_rule_ids": sorted(
                        {x for x in item.get("suppressed_rule_ids", []) if isinstance(x, str)}
                    ),
                    "suppressed_rules": sorted(
                        {x for x in item.get("suppressed_rules", []) if isinstance(x, str)}
                    ),
                }
            )
        )
        metadata.append(
            _canonical_json(
                {
                    "blocking_target_id": item.get("blocking_target_id"),
                    "source_target_id": item.get("source_target_id"),
                    "reason": item.get("reason"),
                    "suppression_type": item.get("suppression_type"),
                }
            )
        )
    return sorted(set(core)), sorted(set(metadata))


def _score_suppression(gold: Any, predicted: Any) -> dict[str, Any]:
    gold_core, gold_meta = _suppression_components(gold)
    pred_core, pred_meta = _suppression_components(predicted)
    core = set_metrics(pred_core, gold_core)
    metadata = set_metrics(pred_meta, gold_meta)
    return {
        "f1": mean((core["f1"], metadata["f1"])),
        "exact": float(
            sorted(_canonical_json(item) for item in predicted or [])
            == sorted(_canonical_json(item) for item in gold or [])
        ),
        "diagnostics": {
            "core_semantics_f1": core["f1"],
            "metadata_f1": metadata["f1"],
        },
    }


def _fullfield_score(field: str, gold: Mapping[str, Any], pred: Mapping[str, Any], family: str) -> dict[str, Any]:
    if field in CORE_FIELDS:
        return set_metrics(pred.get(field, []) or [], gold.get(field, []) or [])
    if field == "target_summaries":
        return _score_target_summaries(gold.get(field, []), pred.get(field, []), family)
    if field == "primitive_ledger_summary":
        return _score_ledger(gold.get(field, {}), pred.get(field, {}), family)
    if field == "suppression_records":
        return _score_suppression(gold.get(field, []), pred.get(field, []))
    raise ScoringError(f"unknown full-fields head: {field}")


def _seed_for(base_seed: int, label: str) -> int:
    return base_seed ^ int(sha256_json(label)[:8], 16)


def score_predictions(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    output_scope: str,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260811,
) -> ScoreBundle:
    gold_index = _index_unique(gold_rows, "gold")
    prediction_index = _index_unique(prediction_rows, "predictions")
    if set(gold_index) != set(prediction_index):
        raise ScoringError(
            "prediction/gold ID mismatch: "
            f"missing={sorted(set(gold_index) - set(prediction_index))[:5]}, "
            f"extra={sorted(set(prediction_index) - set(gold_index))[:5]}"
        )
    fields = CORE_FIELDS if output_scope == "core_three_v1" else FULLFIELDS
    field_vectors: dict[str, list[float]] = {field: [] for field in fields}
    field_precision: dict[str, list[float]] = {field: [] for field in fields}
    field_recall: dict[str, list[float]] = {field: [] for field in fields}
    field_exact: dict[str, list[float]] = {field: [] for field in fields}
    false_positive = {field: Counter() for field in CORE_FIELDS}
    false_negative = {field: Counter() for field in CORE_FIELDS}
    family_vectors: dict[str, list[float]] = defaultdict(list)
    parse_counts: Counter[str] = Counter()
    per_sample: list[dict[str, Any]] = []

    for sample_id, gold_row in gold_index.items():
        labels = gold_row.get("labels")
        if not isinstance(labels, Mapping):
            raise ScoringError(f"gold labels missing for {sample_id}")
        prediction_row = prediction_index[sample_id]
        prediction = _prediction_payload(prediction_row)
        family = str(gold_row.get("pattern") or "unknown")
        scores: dict[str, Any] = {}
        applicable_fields = list(fields)
        if output_scope == "gpt_fullfields_six_v1" and family != "restricted_multi":
            applicable_fields.remove("suppression_records")
        for field in applicable_fields:
            score = (
                set_metrics(prediction.get(field, []) or [], labels.get(field, []) or [])
                if output_scope == "core_three_v1"
                else _fullfield_score(field, labels, prediction, family)
            )
            scores[field] = score
            field_vectors[field].append(float(score["f1"]))
            field_exact[field].append(float(score["exact"]))
            field_precision[field].append(float(score.get("precision", score["f1"])))
            field_recall[field].append(float(score.get("recall", score["f1"])))
            if field in CORE_FIELDS:
                false_positive[field].update(score.get("false_positive", []))
                false_negative[field].update(score.get("false_negative", []))
        sample_mean = mean(score["f1"] for score in scores.values())
        family_vectors[family].append(sample_mean)
        parse_status = str(prediction_row.get("parse_status") or "legacy_unrecorded")
        parse_counts[parse_status] += 1
        per_sample.append(
            {
                "run_id": run_id,
                "sample_id": sample_id,
                "pattern": family,
                "parse_status": parse_status,
                "scores": scores,
                "mean_field_f1": sample_mean,
                "all_applicable_exact": float(all(score["exact"] == 1.0 for score in scores.values())),
            }
        )

    aggregate_fields: dict[str, Any] = {}
    for field in fields:
        vector = field_vectors[field]
        if not vector:
            continue
        aggregate_fields[field] = {
            "count": len(vector),
            "precision": mean(field_precision[field]),
            "recall": mean(field_recall[field]),
            "f1": mean(vector),
            "exact_rate": mean(field_exact[field]),
            "bootstrap_95_ci": bootstrap_mean_ci(
                vector,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=_seed_for(bootstrap_seed, f"field:{field}"),
            ),
        }
        if field in CORE_FIELDS:
            aggregate_fields[field]["top_false_positive_labels"] = false_positive[
                field
            ].most_common(20)
            aggregate_fields[field]["top_false_negative_labels"] = false_negative[
                field
            ].most_common(20)
    primary = mean(item["f1"] for item in aggregate_fields.values())
    sample_vector = [row["mean_field_f1"] for row in per_sample]
    metrics = {
        "schema_version": 1,
        "run_id": run_id,
        "output_scope": output_scope,
        "sample_count": len(per_sample),
        "primary_mean_field_f1": primary,
        "sample_mean_field_f1": mean(sample_vector),
        "sample_mean_field_f1_bootstrap_95_ci": bootstrap_mean_ci(
            sample_vector,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=_seed_for(bootstrap_seed, "sample_mean"),
        ),
        "all_applicable_exact_rate": mean(row["all_applicable_exact"] for row in per_sample),
        "field_aggregate": aggregate_fields,
        "parse_status_counts": dict(sorted(parse_counts.items())),
        "statistics": {
            "bootstrap_unit": "sample",
            "bootstrap_replicates": bootstrap_replicates,
            "confidence_level": confidence_level,
            "bootstrap_seed": bootstrap_seed,
            "training_seed_variance_claim_allowed": False,
        },
    }
    by_family: dict[str, Any] = {}
    for family, vector in sorted(family_vectors.items()):
        by_family[family] = {
            "n": len(vector),
            "mean_field_f1": mean(vector),
            "bootstrap_95_ci": bootstrap_mean_ci(
                vector,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=_seed_for(bootstrap_seed, f"family:{family}"),
            ),
        }
    return ScoreBundle(metrics, by_family, tuple(per_sample))


def write_score_artifacts(bundle: ScoreBundle, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    write_json_atomic(root / "metrics.json", bundle.metrics)
    write_json_atomic(root / "metrics_by_family.json", bundle.metrics_by_family)
    write_jsonl_atomic(root / "metrics_per_sample.jsonl", bundle.per_sample)
    return {
        "metrics_sha256": sha256_file(root / "metrics.json"),
        "metrics_by_family_sha256": sha256_file(root / "metrics_by_family.json"),
        "metrics_per_sample_sha256": sha256_file(root / "metrics_per_sample.jsonl"),
    }

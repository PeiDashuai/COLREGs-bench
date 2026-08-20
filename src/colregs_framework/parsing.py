"""Strict, syntax-only parsing for contract-bound model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .io import JsonFormatError, strict_json_loads


CORE_FIELDS = ("triggered_rules", "maneuver_allowed", "maneuver_forbidden")
FULLFIELDS = CORE_FIELDS + (
    "target_summaries",
    "primitive_ledger_summary",
    "suppression_records",
)


@dataclass(frozen=True)
class ParseResult:
    prediction: dict[str, Any]
    status: str
    repair: str | None
    error: str | None


def empty_prediction(output_scope: str) -> dict[str, Any]:
    if output_scope == "core_three_v1":
        return {field: [] for field in CORE_FIELDS}
    if output_scope == "gpt_fullfields_six_v1":
        return {
            "triggered_rules": [],
            "maneuver_allowed": [],
            "maneuver_forbidden": [],
            "target_summaries": [],
            "primitive_ledger_summary": {},
            "suppression_records": [],
        }
    raise ValueError(f"unknown output scope: {output_scope}")


def _strip_single_code_fence(text: str) -> tuple[str, str | None]:
    value = text.strip().lstrip("\ufeff")
    if not value.startswith("```") or not value.endswith("```"):
        return value, None
    first_break = value.find("\n")
    if first_break < 0:
        return value, None
    opening = value[:first_break].strip().lower()
    if opening not in {"```", "```json"}:
        return value, None
    return value[first_break + 1 : -3].strip(), "single_json_code_fence"


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
            if depth < 0:
                return None
    return None


def _load_object(text: str) -> tuple[dict[str, Any], str | None]:
    stripped, fence_repair = _strip_single_code_fence(text)
    try:
        value = strict_json_loads(stripped, source="model output")
        repair = fence_repair
    except JsonFormatError as direct_error:
        candidate = _first_balanced_object(stripped)
        if candidate is None or candidate == stripped:
            raise direct_error
        value = strict_json_loads(candidate, source="extracted model output")
        repair = "balanced_json_object_extraction"
    if not isinstance(value, dict):
        raise JsonFormatError("model output JSON root must be an object")
    return value, repair


def _validate_schema(value: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise JsonFormatError("jsonschema is required to validate predictions") from exc
    errors = sorted(
        Draft202012Validator(dict(schema)).iter_errors(dict(value)),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        preview = []
        for error in errors[:5]:
            location = "/".join(str(part) for part in error.absolute_path) or "<root>"
            preview.append(f"{location}: {error.message}")
        raise JsonFormatError("prediction schema violation: " + "; ".join(preview))


def _validate_closed_vocabulary(
    value: Mapping[str, Any], vocabulary: Mapping[str, Any]
) -> None:
    fields = vocabulary.get("fields")
    if not isinstance(fields, Mapping):
        raise JsonFormatError("vocabulary artifact lacks fields")
    violations: list[str] = []
    for field in CORE_FIELDS:
        allowed_values = fields.get(field)
        if not isinstance(allowed_values, list) or not all(
            isinstance(item, str) for item in allowed_values
        ):
            raise JsonFormatError(f"vocabulary field is invalid: {field}")
        allowed = set(allowed_values)
        predicted = value.get(field, [])
        if isinstance(predicted, list):
            violations.extend(
                f"{field}:{item}" for item in predicted if item not in allowed
            )
    if violations:
        raise JsonFormatError(
            "prediction contains out-of-vocabulary labels: " + ", ".join(violations[:10])
        )


def _canonicalize(value: Mapping[str, Any], output_scope: str) -> dict[str, Any]:
    result = dict(value)
    for field in CORE_FIELDS:
        result[field] = sorted(set(result[field]))
    if output_scope == "gpt_fullfields_six_v1":
        result["target_summaries"] = list(result["target_summaries"])
        result["primitive_ledger_summary"] = dict(result["primitive_ledger_summary"])
        result["suppression_records"] = list(result["suppression_records"])
    return result


def parse_prediction_text(
    text: str,
    *,
    output_scope: str,
    schema: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
) -> ParseResult:
    """Parse one output without inventing, dropping, or mapping semantic labels."""

    if not isinstance(text, str):
        return ParseResult(
            empty_prediction(output_scope), "failed", None, "model output is not text"
        )
    try:
        value, repair = _load_object(text)
        _validate_schema(value, schema)
        _validate_closed_vocabulary(value, vocabulary)
        prediction = _canonicalize(value, output_scope)
    except (JsonFormatError, KeyError, TypeError, ValueError) as exc:
        return ParseResult(empty_prediction(output_scope), "failed", None, str(exc))
    return ParseResult(
        prediction,
        "repaired" if repair else "valid",
        repair,
        None,
    )

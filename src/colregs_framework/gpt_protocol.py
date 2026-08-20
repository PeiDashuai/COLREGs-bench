"""Frozen GPT full-fields run matrix and provider-compatible strict schema."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator


GPT_RUN_ORDER = (
    "gpt54_text_fullfields",
    "gpt54_multimodal_fullfields",
    "gpt54mini_text_fullfields",
    "gpt54mini_multimodal_fullfields",
)


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": dict(properties),
    }


def _string_enum(values: Iterable[str]) -> dict[str, Any]:
    items = sorted(set(values))
    if not items:
        raise ValueError("string enum cannot be empty")
    return {"type": "string", "enum": items}


def _string_array(values: Iterable[str]) -> dict[str, Any]:
    return {"type": "array", "items": _string_enum(values)}


def _control(rules: list[str], *, urgency: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "polarities": _string_array(("allow", "forbid")),
        "support_rules": _string_array(rules),
    }
    if urgency:
        properties["observed_urgency"] = _string_enum(("low", "medium", "high"))
    return _object(properties)


def build_strict_fullfields_schema(vocabulary: Mapping[str, Any]) -> dict[str, Any]:
    """Build one strict schema from the train/validation-frozen vocabulary."""

    fields = vocabulary.get("fields")
    nested = vocabulary.get("diagnostic_nested_string_vocabulary")
    if not isinstance(fields, Mapping) or not isinstance(nested, Mapping):
        raise ValueError("full-fields vocabulary lacks top-level or nested vocabularies")
    rules = [str(item) for item in fields["triggered_rules"]]
    allowed = [str(item) for item in fields["maneuver_allowed"]]
    forbidden = [str(item) for item in fields["maneuver_forbidden"]]
    nested_target = [str(item) for item in nested["target_summaries"]]
    nested_ledger = [str(item) for item in nested["primitive_ledger_summary"]]
    nested_suppression = [str(item) for item in nested["suppression_records"]]
    target_ids = tuple(f"t{index}" for index in range(1, 7))
    target_roles = tuple(f"target_{index}" for index in range(1, 7))
    bearings = ("ahead", "astern", "port", "starboard")
    intended_roles = tuple(
        value
        for value in nested_target
        if value.endswith("_target") and value not in target_roles
    )

    target_base: dict[str, Any] = {
        "target_id": _string_enum(target_ids),
        "triggered_rule_ids": _string_array(rules),
        "relative_bearing_sector": _string_enum(bearings),
        "cpa_m": {"type": "number"},
        "tcpa_s": {"type": "number"},
        "risk_of_collision": {"type": "boolean"},
        "closing": {"type": "boolean"},
        "collision_imminent": {"type": "boolean"},
    }
    target_summary = {
        "anyOf": [
            _object(target_base),
            _object({**target_base, "intended_role": _string_enum(intended_roles)}),
            _object({**target_base, "target_role": _string_enum(target_roles)}),
        ]
    }

    suppression_record = _object(
        {
            "record_id": _string_enum(
                value for value in nested_suppression if value.startswith("supp_")
            ),
            "suppression_type": _string_enum(
                value
                for value in nested_suppression
                if value in {"action_conflict", "area_override", "hierarchy_override", "mixed"}
            ),
            "source_target_id": _string_enum((*target_ids, "unknown")),
            "blocking_target_id": _string_enum((*target_ids, "unknown")),
            "suppressed_rule_ids": _string_array(rules),
            "suppressed_rules": _string_array(rules),
            "suppressed_primitive": _string_enum(
                value
                for value in nested_suppression
                if value not in rules
                and value
                not in {
                    "action_conflict",
                    "area_override",
                    "hierarchy_override",
                    "mixed",
                    "supp_001",
                    "unknown",
                    "COLREG_MULTI_TARGET_ACTION_CONFLICT suppressed under global multi-target resolution.",
                }
            ),
            "reason": _string_enum(
                value
                for value in nested_suppression
                if value in {"action_conflict", "area_override", "hierarchy_override", "mixed"}
            ),
            "reason_text": _string_enum(
                value for value in nested_suppression if value.endswith("resolution.")
            ),
        }
    )
    action_common = {
        "primitive": _string_enum((*allowed, *forbidden, *nested_ledger)),
        "support_rules": _string_array(rules),
    }
    action_item = {
        "anyOf": [
            _object(action_common),
            _object({**action_common, "targets": _string_array(target_ids)}),
        ]
    }
    restricted_ledger = _object(
        {
            "maneuver": _object(
                {
                    "allowed": {"type": "array", "items": action_item},
                    "forbidden": {"type": "array", "items": action_item},
                }
            ),
            "suppression": _object(
                {
                    "records": {"type": "array", "items": suppression_record},
                    "support_rules": _string_array(rules),
                }
            ),
            "summary": _object(
                {
                    "support_rules": _string_array(rules),
                    "target_ids": _string_array(target_ids),
                    "suppression_count": {"type": "integer", "minimum": 0},
                }
            ),
        }
    )
    channel_ledger = _object(
        {
            "maneuver": _object(
                {
                    "CHANNEL_CONTEXT_CONTROL": _control(rules),
                    "COLLISION_RISK_CONTROL": _control(rules),
                }
            ),
            "family_contract": _object(
                {
                    "semantic_mode": _string_enum(
                        (
                            "keep_starboard_dominant",
                            "mixed_semantics",
                            "not_to_impede",
                            "ordinary_crossing_in_channel",
                        )
                    ),
                    "occupancy_pattern": _string_enum(
                        (
                            "low_occupancy",
                            "opposing_flow",
                            "single_occupancy",
                            "third_vessel_blocking",
                        )
                    ),
                    "ownship_channel_position": _string_enum(
                        (
                            "crossing_across",
                            "entering_from_outside",
                            "inside_boundary",
                            "inside_center",
                        )
                    ),
                    "observed_crossing_angle_band": _string_enum(
                        ("small", "medium", "large")
                    ),
                }
            ),
        }
    )
    responsibility_ledger = _object(
        {
            "maneuver": _object(
                {
                    "ROLE_RESPONSIBILITY_CONTROL": _control(rules),
                    "URGENCY_AWARE_ACTION_SCALING": _control(rules, urgency=True),
                }
            )
        }
    )
    tss_ledger = _object(
        {
            "maneuver": _object(
                {
                    "CROSSING_CONTROL": _control(rules),
                    "TSS_RULE_VARIANT": _control(rules),
                }
            )
        }
    )
    schema = _object(
        {
            "triggered_rules": _string_array(rules),
            "maneuver_allowed": _string_array(allowed),
            "maneuver_forbidden": _string_array(forbidden),
            "target_summaries": {"type": "array", "items": target_summary},
            "primitive_ledger_summary": {
                "anyOf": [
                    channel_ledger,
                    responsibility_ledger,
                    restricted_ledger,
                    tss_ledger,
                ]
            },
            "suppression_records": {"type": "array", "items": suppression_record},
        }
    )
    schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "gpt_fullfields_prediction_v2.schema.json",
            "title": "Strict six-head full-fields benchmark prediction",
        }
    )
    Draft202012Validator.check_schema(schema)
    return schema


def assert_all_objects_strict(schema: Mapping[str, Any]) -> None:
    failures: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if value.get("type") == "object":
                properties = value.get("properties")
                required = value.get("required")
                if value.get("additionalProperties") is not False:
                    failures.append(f"{path}:additionalProperties")
                if not isinstance(properties, Mapping) or set(required or ()) != set(properties):
                    failures.append(f"{path}:required")
            for key, child in value.items():
                visit(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")

    visit(schema, "$")
    if failures:
        raise ValueError(f"provider-strict object audit failed: {failures[:10]}")


def fullfields_gold_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    labels = row.get("labels")
    if not isinstance(labels, Mapping):
        raise ValueError(f"row lacks labels: {row.get('sample_id')}")
    return {
        "triggered_rules": labels.get("triggered_rules", []),
        "maneuver_allowed": labels.get("maneuver_allowed", []),
        "maneuver_forbidden": labels.get("maneuver_forbidden", []),
        "target_summaries": labels.get("target_summaries", []),
        "primitive_ledger_summary": labels.get("primitive_ledger_summary", {}),
        "suppression_records": labels.get("suppression_records", []),
    }


def validate_fullfields_rows(
    rows: Iterable[Mapping[str, Any]], schema: Mapping[str, Any]
) -> list[dict[str, Any]]:
    validator = Draft202012Validator(dict(schema))
    failures: list[dict[str, Any]] = []
    for row in rows:
        errors = sorted(
            validator.iter_errors(fullfields_gold_payload(row)),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            failures.append(
                {
                    "sample_id": row.get("sample_id"),
                    "errors": [error.message for error in errors[:5]],
                }
            )
    return failures

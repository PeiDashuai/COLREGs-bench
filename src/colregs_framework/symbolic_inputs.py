"""Label-blind materialization for the symbolic COLREGs baseline.

The materialized rows intentionally contain only sample identity, the public
family name used for reporting, and observable scene-state fields.  Gold labels,
generator debug fields, role hints, and precomputed geometry never cross this
boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import sha256_file, sha256_ordered_ids, verify_file_sha256
from .io import iter_jsonl, write_json_atomic, write_jsonl_atomic


SCENE_FIELDS = (
    "domain",
    "visibility",
    "snapshot_time_s",
    "duration_s",
    "area_context",
    "ownship",
    "targets",
)
AREA_FIELDS = (
    "area_type",
    "channel_context",
    "tss_context",
    "tss_lane_heading_deg",
    "channel_heading_deg",
    "lane_count",
    "lane_width_m",
    "lane_half_total_width_m",
    "channel_width_m",
    "channel_half_width_m",
)
VESSEL_FIELDS = (
    "id",
    "target_id",
    "vessel_class",
    "nav_status",
    "length_m",
    "x_m",
    "y_m",
    "heading_deg",
    "speed_mps",
)
FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "labels",
        "seed",
        "debug_native_geometry",
        "intended_role",
        "role_hint",
        "is_effective_target",
        "local_geometry",
        "local_priority",
        "relation_type",
        "relative_bearing_sector",
        "role_sector",
        "target_role_graph",
    }
)


class SymbolicInputError(ValueError):
    """Raised when a materialized split would violate the label-blind boundary."""


def _select(source: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {key: source[key] for key in fields if key in source and source[key] is not None}


def sanitize_vessel(value: Mapping[str, Any]) -> dict[str, Any]:
    vessel = _select(value, VESSEL_FIELDS)
    required = ("vessel_class", "x_m", "y_m", "heading_deg", "speed_mps")
    missing = [key for key in required if key not in vessel]
    if missing:
        raise SymbolicInputError(f"vessel lacks observable fields: {missing}")
    return vessel


def sanitize_scene_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SymbolicInputError("scene_spec must be an object")
    result = _select(value, SCENE_FIELDS)
    area = value.get("area_context")
    ownship = value.get("ownship")
    targets = value.get("targets")
    if not isinstance(area, Mapping) or not isinstance(ownship, Mapping):
        raise SymbolicInputError("scene_spec requires area_context and ownship objects")
    if not isinstance(targets, list) or not targets:
        raise SymbolicInputError("scene_spec requires at least one target")
    result["area_context"] = _select(area, AREA_FIELDS)
    result["ownship"] = sanitize_vessel(ownship)
    result["targets"] = [
        sanitize_vessel(item) for item in targets if isinstance(item, Mapping)
    ]
    if len(result["targets"]) != len(targets):
        raise SymbolicInputError("every target must be an object")
    return result


def find_forbidden_fields(value: Any, *, prefix: str = "$") -> tuple[str, ...]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if key in FORBIDDEN_FIELD_NAMES or key.startswith("debug_"):
                hits.append(child_path)
            hits.extend(find_forbidden_fields(child, prefix=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(find_forbidden_fields(child, prefix=f"{prefix}[{index}]"))
    return tuple(hits)


def materialize_inputs_only_split(
    source_path: str | Path,
    output_path: str | Path,
    *,
    expected_count: int,
    expected_sha256: str,
    benchmark_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create one label-blind split and return an auditable manifest."""

    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    benchmark = Path(benchmark_root).resolve()
    verify_file_sha256(source, expected_sha256)
    try:
        output.relative_to(benchmark)
    except ValueError:
        pass
    else:
        raise SymbolicInputError("inputs-only output must be outside the frozen benchmark")

    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for row_number, row in enumerate(iter_jsonl(source), start=1):
        sample_id = row.get("sample_id")
        pattern = row.get("pattern")
        inputs = row.get("inputs")
        scene = inputs.get("scene_spec") if isinstance(inputs, Mapping) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise SymbolicInputError(f"missing sample_id at row {row_number}")
        if not isinstance(pattern, str) or not pattern:
            raise SymbolicInputError(f"missing pattern at row {row_number}")
        materialized = {
            "sample_id": sample_id,
            "pattern": pattern,
            "inputs": {"scene_spec": sanitize_scene_spec(scene)},
        }
        forbidden = find_forbidden_fields(materialized)
        if forbidden:
            raise SymbolicInputError(
                f"forbidden fields crossed boundary at row {row_number}: {forbidden}"
            )
        rows.append(materialized)
        ids.append(sample_id)

    if len(rows) != expected_count:
        raise SymbolicInputError(
            f"row count mismatch: expected {expected_count}, observed {len(rows)}"
        )
    if len(set(ids)) != len(ids):
        raise SymbolicInputError("duplicate sample_id in source split")

    write_jsonl_atomic(output, rows, overwrite=overwrite)
    manifest = {
        "schema_version": 1,
        "boundary": "observable_scene_fields_only",
        "source_path": source.as_posix(),
        "source_sha256": expected_sha256,
        "output_path": output.as_posix(),
        "output_sha256": sha256_file(output),
        "row_count": len(rows),
        "ordered_sample_ids_sha256": sha256_ordered_ids(ids),
        "forbidden_field_hits": [],
        "source_benchmark_unchanged": sha256_file(source) == expected_sha256,
    }
    write_json_atomic(
        output.with_suffix(output.suffix + ".manifest.json"),
        manifest,
        overwrite=overwrite,
    )
    return manifest

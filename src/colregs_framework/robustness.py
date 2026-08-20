"""Fixed robustness subset, deterministic perturbations, and diagnostic scoring."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import LoadedContract
from .datasets import observable_scene_spec
from .hashing import sha256_file, sha256_json, sha256_ordered_ids, sha256_text
from .io import iter_jsonl, read_json, write_json_atomic, write_jsonl_atomic
from .prompting import build_core_three_user_prompt, canonical_prompt_text
from .protocols import EvaluationSample, ProtocolAssets, ProtocolError, load_frozen_test_rows


CONDITION_IDS = (
    "clean_text",
    "clean_multimodal",
    "degraded_text",
    "degraded_text_multimodal",
    "image_only",
    "bounded_numeric_noise",
    "prompt_paraphrase",
    "contradiction",
)
MULTIMODAL_CONDITIONS = {
    "clean_multimodal",
    "degraded_text_multimodal",
    "image_only",
}
REMOVED_GEOMETRY_FIELDS = {
    "x_m",
    "y_m",
    "heading_deg",
    "speed_mps",
    "cpa_m",
    "tcpa_s",
    "relative_bearing_deg",
    "relative_course_deg",
    "relative_speed_mps",
    "distance_from_ownship_m",
}
NOISE_BOUNDS = {
    "x_m": 50.0,
    "y_m": 50.0,
    "heading_deg": 3.0,
    "speed_mps": 0.3,
}
ABSTAIN_TOKEN = "ABSTAIN_INVALID_INPUT"


class RobustnessError(ValueError):
    """Raised when a frozen robustness artifact or transformation drifts."""


@dataclass(frozen=True)
class RobustnessArtifacts:
    subset_manifest: Path
    transform_manifest: Path
    protocol_manifest: Path


def _stable_key(seed: int, sample_id: str) -> str:
    return sha256_json({"sample_id": sample_id, "selection_seed": seed})


def _gold_set_size(row: Mapping[str, Any]) -> int:
    labels = row.get("labels")
    if not isinstance(labels, Mapping):
        raise RobustnessError(f"labels missing for {row.get('sample_id')}")
    return sum(
        len(labels.get(field, []))
        for field in ("triggered_rules", "maneuver_allowed", "maneuver_forbidden")
    )


def _selection_stratum(row: Mapping[str, Any]) -> tuple[int, int, int]:
    scene = row.get("inputs", {}).get("scene_spec", {})
    targets = scene.get("targets", []) if isinstance(scene, Mapping) else []
    area = scene.get("area_context", {}) if isinstance(scene, Mapping) else {}
    if not isinstance(targets, list) or not isinstance(area, Mapping):
        raise RobustnessError(f"invalid selection metadata for {row.get('sample_id')}")
    nonempty_area_fields = sum(value not in (None, False, "", [], {}) for value in area.values())
    complexity = len(targets) + nonempty_area_fields + _gold_set_size(row)
    return len(targets), _gold_set_size(row), complexity


def select_robustness_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_counts: Mapping[str, int],
    selection_seed: int,
) -> list[Mapping[str, Any]]:
    """Select quotas with deterministic round-robin coverage across exact strata."""

    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("pattern") or "")].append(row)
    selected_ids: set[str] = set()
    for family, quota in family_counts.items():
        candidates = by_family.get(family, [])
        if len(candidates) < quota:
            raise RobustnessError(f"family {family} has {len(candidates)} rows, needs {quota}")
        if quota == len(candidates):
            chosen = candidates
        else:
            groups: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
            for row in candidates:
                groups[_selection_stratum(row)].append(row)
            queues = {
                key: sorted(
                    group,
                    key=lambda item: (_stable_key(selection_seed, str(item["sample_id"])), item["sample_id"]),
                )
                for key, group in groups.items()
            }
            chosen = []
            keys = sorted(queues)
            while len(chosen) < quota:
                progressed = False
                for key in keys:
                    if queues[key] and len(chosen) < quota:
                        chosen.append(queues[key].pop(0))
                        progressed = True
                if not progressed:
                    raise RobustnessError(f"selection exhausted early for {family}")
        selected_ids.update(str(row["sample_id"]) for row in chosen)
    ordered = [row for row in rows if str(row["sample_id"]) in selected_ids]
    expected = sum(family_counts.values())
    if len(ordered) != expected or len(selected_ids) != expected:
        raise RobustnessError("selected robustness IDs are not unique and complete")
    actual = Counter(str(row["pattern"]) for row in ordered)
    if dict(actual) != dict(family_counts):
        raise RobustnessError(f"family quota drift: {dict(actual)} != {dict(family_counts)}")
    return ordered


def _drop_geometry(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _drop_geometry(child)
            for key, child in value.items()
            if key not in REMOVED_GEOMETRY_FIELDS
        }
    if isinstance(value, list):
        return [_drop_geometry(child) for child in value]
    return deepcopy(value)


def _noise_delta(seed: int, sample_id: str, vessel_path: str, field: str, bound: float) -> float:
    digest = sha256_json(
        {"seed": seed, "sample_id": sample_id, "vessel_path": vessel_path, "field": field}
    )
    unit = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return round((2.0 * unit - 1.0) * bound, 6)


def _noise_scene(
    scene: Mapping[str, Any], sample_id: str, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = deepcopy(dict(scene))
    deltas: list[dict[str, Any]] = []
    vessels: list[tuple[str, dict[str, Any]]] = []
    ownship = result.get("ownship")
    if isinstance(ownship, dict):
        vessels.append(("ownship", ownship))
    targets = result.get("targets")
    if isinstance(targets, list):
        vessels.extend(
            (f"targets[{index}]", vessel)
            for index, vessel in enumerate(targets)
            if isinstance(vessel, dict)
        )
    for vessel_path, vessel in vessels:
        for field, bound in NOISE_BOUNDS.items():
            original = vessel.get(field)
            if not isinstance(original, (int, float)) or isinstance(original, bool):
                continue
            delta = _noise_delta(seed, sample_id, vessel_path, field, bound)
            perturbed = float(original) + delta
            if field == "speed_mps":
                perturbed = max(0.0, perturbed)
            elif field == "heading_deg":
                perturbed %= 360.0
            perturbed = round(perturbed, 6)
            vessel[field] = perturbed
            deltas.append(
                {
                    "path": f"scene_spec.{vessel_path}.{field}",
                    "original": original,
                    "delta": delta,
                    "perturbed": perturbed,
                    "bound": bound,
                }
            )
    return result, deltas


def _image_only_prompt(vocabulary_fields: Mapping[str, Any]) -> str:
    return build_core_three_user_prompt(
        {"scene_spec": "Use only the supplied top-down and radar images."},
        vocabulary_fields,
    )


def _relative_images(row: Mapping[str, Any]) -> tuple[str, str]:
    inputs = row.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RobustnessError(f"inputs missing for {row.get('sample_id')}")
    topdown = row.get("topdown_image") or inputs.get("topdown_image")
    radar = row.get("radar_image") or inputs.get("radar_image")
    if not isinstance(topdown, str) or not isinstance(radar, str):
        raise RobustnessError(f"image paths missing for {row.get('sample_id')}")
    return topdown.replace("\\", "/"), radar.replace("\\", "/")


def transform_row(
    contract: LoadedContract,
    row: Mapping[str, Any],
    assets: ProtocolAssets,
    *,
    condition_id: str,
    paraphrase_prompt: str,
) -> tuple[EvaluationSample, dict[str, Any]]:
    if condition_id not in CONDITION_IDS:
        raise RobustnessError(f"unknown robustness condition: {condition_id}")
    sample_id = str(row["sample_id"])
    pattern = str(row["pattern"])
    inputs = row.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("scene_spec"), Mapping):
        raise RobustnessError(f"scene_spec missing for {sample_id}")
    vocabulary_fields = assets.vocabulary.get("fields")
    if not isinstance(vocabulary_fields, Mapping):
        raise RobustnessError("core vocabulary lacks fields")
    clean_scene = observable_scene_spec(contract, inputs["scene_spec"])
    payload: Mapping[str, Any] | None = {"family": pattern, "scene_spec": clean_scene}
    system_prompt = assets.prompt
    transformation: dict[str, Any] = {"condition_id": condition_id}
    if condition_id in {"degraded_text", "degraded_text_multimodal"}:
        payload = {"family": pattern, "scene_spec": _drop_geometry(clean_scene)}
        transformation["removed_fields"] = sorted(REMOVED_GEOMETRY_FIELDS)
    elif condition_id == "image_only":
        payload = None
    elif condition_id == "bounded_numeric_noise":
        noisy, deltas = _noise_scene(
            clean_scene,
            sample_id,
            int(contract.data["robustness"]["numeric_noise_seed"]),
        )
        payload = {"family": pattern, "scene_spec": noisy}
        transformation.update(
            {
                "deltas": deltas,
                "reference_policy": "fixed_latent_scene_reference",
                "gold_stability_under_noise": "not_recomputed_no_frozen_oracle",
            }
        )
    elif condition_id == "prompt_paraphrase":
        system_prompt = paraphrase_prompt
    elif condition_id == "contradiction":
        payload = {
            "family": pattern,
            "scene_spec": clean_scene,
            "input_consistency_check": {
                "field": "visibility",
                "status": "explicit_contradiction",
                "reported_values": ["in_sight", "restricted"],
            },
        }
        system_prompt = (
            assets.prompt
            + "\nThe input includes an explicit consistency check. If it reports contradictory "
            + f"values, return exactly {ABSTAIN_TOKEN} and no JSON or commentary."
        )
        transformation["contradiction"] = dict(payload["input_consistency_check"])
    user_prompt = (
        _image_only_prompt(vocabulary_fields)
        if payload is None
        else build_core_three_user_prompt(payload, vocabulary_fields)
    )
    image_relpaths = _relative_images(row) if condition_id in MULTIMODAL_CONDITIONS else ()
    record = {
        "sample_id": sample_id,
        "pattern": pattern,
        "condition_id": condition_id,
        "source_row_sha256": sha256_json(dict(row)),
        "payload_sha256": sha256_json(payload),
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "image_paths": list(image_relpaths),
        "transformation": transformation,
    }
    sample = EvaluationSample(
        sample_id=sample_id,
        pattern=pattern,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=tuple(Path(path) for path in image_relpaths),
        source=row,
    )
    return sample, record


def _write_or_verify_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RobustnessError(f"existing frozen JSON differs: {path}")
        return
    write_json_atomic(path, value)


def _write_or_verify_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        if list(iter_jsonl(path)) != list(records):
            raise RobustnessError(f"existing frozen JSONL differs: {path}")
        return
    write_jsonl_atomic(path, records)


def materialize_robustness_protocol(
    contract: LoadedContract,
    benchmark_root: str | Path,
    code_root: str | Path,
    output_dir: str | Path,
    assets: ProtocolAssets,
) -> RobustnessArtifacts:
    root = Path(code_root).resolve()
    output = Path(output_dir).resolve()
    rows = load_frozen_test_rows(contract, benchmark_root)
    spec = contract.data["robustness"]
    selected = select_robustness_rows(
        rows,
        family_counts=spec["family_counts"],
        selection_seed=int(spec["selection_seed"]),
    )
    paraphrase_path = root / contract.data["prompt_contracts"]["robustness_paraphrase_v1"]["path"]
    paraphrase_prompt = canonical_prompt_text(paraphrase_path.read_text(encoding="utf-8"))
    subset_path = output / "robustness_subset_manifest.json"
    transform_path = output / "robustness_transform_manifest.jsonl"
    protocol_path = output / "robustness_protocol_manifest.json"
    subset = {
        "schema_version": 1,
        "subset_id": spec["subset_id"],
        "source_freeze_id": contract.data["benchmark"]["freeze_id"],
        "source_test_sha256": contract.data["benchmark"]["evaluator_view"]["test"]["sha256"],
        "selection_seed": spec["selection_seed"],
        "selection_method": "exact_stratum_round_robin_then_source_order_v1",
        "stratum_fields": ["target_count", "core_three_gold_set_size", "complexity_proxy"],
        "sample_count": len(selected),
        "family_counts": dict(Counter(str(row["pattern"]) for row in selected)),
        "ordered_sample_ids": [str(row["sample_id"]) for row in selected],
        "ordered_sample_id_sha256": sha256_ordered_ids(str(row["sample_id"]) for row in selected),
        "selection_rows": [
            {
                "sample_id": row["sample_id"],
                "pattern": row["pattern"],
                "stratum": list(_selection_stratum(row)),
                "stable_selection_key": _stable_key(int(spec["selection_seed"]), str(row["sample_id"])),
            }
            for row in selected
        ],
    }
    transform_records: list[dict[str, Any]] = []
    for condition_id in CONDITION_IDS:
        for row in selected:
            _, record = transform_row(
                contract,
                row,
                assets,
                condition_id=condition_id,
                paraphrase_prompt=paraphrase_prompt,
            )
            transform_records.append(record)
    _write_or_verify_json(subset_path, subset)
    _write_or_verify_jsonl(transform_path, transform_records)
    protocol = {
        "schema_version": 1,
        "subset_id": spec["subset_id"],
        "condition_ids": list(CONDITION_IDS),
        "subset_size": len(selected),
        "transformation_row_count": len(transform_records),
        "subset_manifest_path": subset_path.name,
        "subset_manifest_sha256": sha256_file(subset_path),
        "transform_manifest_path": transform_path.name,
        "transform_manifest_sha256": sha256_file(transform_path),
        "robustness_implementation_sha256": sha256_file(Path(__file__)),
        "paraphrase_prompt_sha256": sha256_file(paraphrase_path),
        "numeric_noise_seed": spec["numeric_noise_seed"],
        "numeric_noise_bounds": dict(NOISE_BOUNDS),
        "new_run_count": len(spec.get("runs", [])),
        "expected_new_inference_calls": spec["expected_new_inference_calls"],
    }
    _write_or_verify_json(protocol_path, protocol)
    return RobustnessArtifacts(subset_path, transform_path, protocol_path)


def load_robustness_samples(
    contract: LoadedContract,
    benchmark_root: str | Path,
    code_root: str | Path,
    assets: ProtocolAssets,
    *,
    condition_id: str,
) -> list[EvaluationSample]:
    root = Path(code_root).resolve()
    spec = contract.data["robustness"]
    protocol_path = root / spec["protocol_manifest_path"]
    if sha256_file(protocol_path) != spec["protocol_manifest_sha256"]:
        raise RobustnessError("robustness protocol manifest hash drift")
    protocol = read_json(protocol_path)
    subset_path = protocol_path.parent / protocol["subset_manifest_path"]
    transform_path = protocol_path.parent / protocol["transform_manifest_path"]
    if sha256_file(subset_path) != protocol["subset_manifest_sha256"]:
        raise RobustnessError("robustness subset manifest hash drift")
    if sha256_file(transform_path) != protocol["transform_manifest_sha256"]:
        raise RobustnessError("robustness transform manifest hash drift")
    subset = read_json(subset_path)
    ordered_ids = subset["ordered_sample_ids"]
    rows_by_id = {row["sample_id"]: row for row in load_frozen_test_rows(contract, benchmark_root)}
    selected = [rows_by_id[sample_id] for sample_id in ordered_ids]
    paraphrase_path = root / contract.data["prompt_contracts"]["robustness_paraphrase_v1"]["path"]
    paraphrase = canonical_prompt_text(paraphrase_path.read_text(encoding="utf-8"))
    expected_records = {
        (row["condition_id"], row["sample_id"]): row for row in iter_jsonl(transform_path)
    }
    samples: list[EvaluationSample] = []
    benchmark = Path(benchmark_root).resolve()
    for row in selected:
        sample, record = transform_row(
            contract,
            row,
            assets,
            condition_id=condition_id,
            paraphrase_prompt=paraphrase,
        )
        if expected_records.get((condition_id, sample.sample_id)) != record:
            raise RobustnessError(f"transformation drift for {condition_id}/{sample.sample_id}")
        image_paths = tuple((benchmark / path).resolve() for path in record["image_paths"])
        for image in image_paths:
            try:
                image.relative_to(benchmark)
            except ValueError as exc:
                raise ProtocolError(f"robustness image escapes benchmark root: {image}") from exc
            if not image.is_file():
                raise ProtocolError(f"robustness image missing: {image}")
        samples.append(
            EvaluationSample(
                sample_id=sample.sample_id,
                pattern=sample.pattern,
                system_prompt=sample.system_prompt,
                user_prompt=sample.user_prompt,
                image_paths=image_paths,
                source=sample.source,
            )
        )
    return samples


def classify_contradiction_output(text: str) -> tuple[str, str, str | None]:
    normalized = text.strip()
    if normalized == ABSTAIN_TOKEN:
        return "invalid", "abstained_invalid", None
    if normalized.startswith("{") and normalized.endswith("}"):
        return "unknown", "unsupported_deterministic_output", None
    return "unknown", "failed", "contradiction response is neither exact abstention nor JSON"

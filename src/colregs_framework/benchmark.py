"""Reader and integrity checks for the public COLREGs benchmark release."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Iterator, Mapping


EXPECTED_SPLIT_COUNTS = {"train": 1190, "validation": 254, "test": 256}
EXPECTED_FAMILY_COUNTS = {
    "restricted_multi": 500,
    "responsibility_mix": 550,
    "channel_crossing_impede": 550,
    "tss_crossing": 100,
}
CORE_FIELDS = ("triggered_rules", "maneuver_allowed", "maneuver_forbidden")
EVALUATION_VIEWS = ("full_structured", "expert_corrected_core3")


class BenchmarkError(ValueError):
    """Raised when a release is incomplete or internally inconsistent."""


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BenchmarkError(f"{source}:{line_number} is not a JSON object")
            yield value


class BenchmarkRelease:
    """Access input records and the release's explicitly separated evaluation views."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not (self.root / "release.json").is_file():
            raise BenchmarkError(f"not a benchmark release root: {self.root}")

    @staticmethod
    def normalize_split(split: str) -> str:
        value = "validation" if split == "val" else split
        if value not in EXPECTED_SPLIT_COUNTS:
            raise BenchmarkError(f"unknown split: {split}")
        return value

    def input_path(self, split: str) -> Path:
        return self.root / "data" / f"{self.normalize_split(split)}.jsonl"

    def reference_path(self, split: str, view: str) -> Path:
        split = self.normalize_split(split)
        if view == "full_structured":
            return self.root / "evaluation" / view / f"{split}.jsonl"
        if view == "expert_corrected_core3" and split == "test":
            return self.root / "evaluation" / view / "test.jsonl"
        raise BenchmarkError(f"view {view!r} is not defined for split {split!r}")

    def inputs(self, split: str) -> list[dict[str, Any]]:
        return list(iter_jsonl(self.input_path(split)))

    def references(self, split: str, view: str) -> list[dict[str, Any]]:
        return list(iter_jsonl(self.reference_path(split, view)))

    def joined(self, split: str, view: str) -> list[dict[str, Any]]:
        inputs = self.inputs(split)
        references = self.references(split, view)
        reference_by_id = _unique_index(references, f"{view} references")
        if {row["sample_id"] for row in inputs} != set(reference_by_id):
            raise BenchmarkError(f"input/reference ID mismatch for {split}/{view}")
        result: list[dict[str, Any]] = []
        for row in inputs:
            merged = dict(row)
            merged["labels"] = reference_by_id[row["sample_id"]]["labels"]
            result.append(merged)
        return result

    def image_path(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or PureWindowsPath(relative).is_absolute():
            raise BenchmarkError(f"image path must be relative: {relative}")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise BenchmarkError(f"image path escapes release root: {relative}") from exc
        return resolved

    def validate(self, *, check_images: bool = True) -> dict[str, Any]:
        all_ids: set[str] = set()
        family_counts: Counter[str] = Counter()
        split_counts: dict[str, int] = {}
        image_count = 0
        errors: list[str] = []

        for split, expected in EXPECTED_SPLIT_COUNTS.items():
            rows = self.inputs(split)
            split_counts[split] = len(rows)
            if len(rows) != expected:
                errors.append(f"{split}_count:{len(rows)}!={expected}")
            ids = [str(row.get("sample_id") or "") for row in rows]
            if len(ids) != len(set(ids)) or "" in ids:
                errors.append(f"{split}_duplicate_or_empty_id")
            overlap = all_ids & set(ids)
            if overlap:
                errors.append(f"cross_split_overlap:{sorted(overlap)[:3]}")
            all_ids.update(ids)
            for row in rows:
                family_counts[str(row.get("pattern"))] += 1
                if "labels" in row:
                    errors.append(f"labels_cross_input_boundary:{row.get('sample_id')}")
                for field in ("topdown_image", "radar_image"):
                    value = row.get(field)
                    if not isinstance(value, str):
                        errors.append(f"missing_{field}:{row.get('sample_id')}")
                        continue
                    image_count += 1
                    if check_images and not self.image_path(value).is_file():
                        errors.append(f"missing_image:{value}")
            full = self.references(split, "full_structured")
            if [row.get("sample_id") for row in rows] != [row.get("sample_id") for row in full]:
                errors.append(f"full_structured_order:{split}")

        if dict(family_counts) != EXPECTED_FAMILY_COUNTS:
            errors.append(f"family_counts:{dict(family_counts)}")

        core = self.references("test", "expert_corrected_core3")
        test_ids = [row.get("sample_id") for row in self.inputs("test")]
        if [row.get("sample_id") for row in core] != test_ids:
            errors.append("expert_corrected_core3_order")
        for row in core:
            labels = row.get("labels")
            if not isinstance(labels, Mapping) or tuple(labels) != CORE_FIELDS:
                errors.append(f"expert_core3_fields:{row.get('sample_id')}")

        ontology = json.loads(
            (self.root / "evaluation" / "common_semantic" / "ontology.json").read_text(
                encoding="utf-8"
            )
        )
        concept_count = sum(len(values) for values in ontology["heads"].values())
        mappings: list[Mapping[str, Any]] = []
        for name in ("benchmark_mapping.json", "symbolic_engine_mapping.json"):
            value = json.loads(
                (self.root / "evaluation" / "common_semantic" / name).read_text(
                    encoding="utf-8"
                )
            )
            mappings.extend(value["entries"])
        clear_count = sum(item.get("review_status") != "unsupported" for item in mappings)
        unsupported_count = len(mappings) - clear_count
        if (concept_count, len(mappings), clear_count, unsupported_count) != (39, 183, 137, 46):
            errors.append(
                "common_semantic_counts:"
                f"{concept_count}/{len(mappings)}/{clear_count}/{unsupported_count}"
            )

        report = {
            "status": "PASS" if not errors else "FAIL",
            "split_counts": split_counts,
            "total_samples": len(all_ids),
            "family_counts": dict(sorted(family_counts.items())),
            "referenced_images": image_count,
            "common_semantic": {
                "concepts": concept_count,
                "mappings": len(mappings),
                "scored_mappings": clear_count,
                "excluded_ambiguous_mappings": unsupported_count,
            },
            "errors": errors[:50],
        }
        if errors:
            raise BenchmarkError(json.dumps(report, ensure_ascii=False))
        return report


def _unique_index(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise BenchmarkError(f"{label} contains an invalid or duplicate sample_id")
        result[sample_id] = row
    return result

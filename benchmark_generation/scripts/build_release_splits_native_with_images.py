#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

FAMILIES = [
    "tss_crossing",
    "responsibility_mix",
    "channel_crossing_impede",
    "restricted_multi",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def family_stratum_key(row: Dict[str, Any]) -> str:
    fs = get_nested(row, "generation_meta.family_semantics.family_stratum", {}) or {}
    return json.dumps(fs, sort_keys=True, ensure_ascii=False)


def quotas(n: int) -> Tuple[int, int, int]:
    if n <= 1:
        return (n, 0, 0)
    if n == 2:
        return (1, 0, 1)
    train = int(round(n * 0.70))
    val = int(round(n * 0.15))
    test = n - train - val
    if val == 0:
        val = 1
        train -= 1
    if test == 0:
        test = 1
        train -= 1
    if train <= 0:
        train = max(1, n - val - test)
    return train, val, test


def stable_shuffle(rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    keyed = []
    for row in rows:
        key = hashlib.sha256(f"{seed}|{row.get('sample_id')}".encode("utf-8")).hexdigest()
        keyed.append((key, row))
    return [r for _, r in sorted(keyed, key=lambda x: x[0])]


def stratified_split_family(rows: List[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    train_q, val_q, test_q = quotas(len(rows))
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[family_stratum_key(row)].append(row)

    bucket_items = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for k, v in bucket_items:
        buckets[k] = stable_shuffle(v, seed)

    splits = {"train": [], "val": [], "test": []}
    caps = {"train": train_q, "val": val_q, "test": test_q}

    order = ["train", "val", "test"]
    oi = 0
    for _, bucket in bucket_items:
        if not bucket:
            continue
        for _ in range(min(3, len(bucket))):
            placed = False
            for _j in range(3):
                s = order[oi % 3]
                oi += 1
                if len(splits[s]) < caps[s]:
                    splits[s].append(bucket.pop(0))
                    placed = True
                    break
            if not placed or not bucket:
                break

    remaining: List[Dict[str, Any]] = []
    for _, bucket in bucket_items:
        remaining.extend(bucket)
    remaining = stable_shuffle(remaining, seed + 999)
    for row in remaining:
        for s in ["train", "val", "test"]:
            if len(splits[s]) < caps[s]:
                splits[s].append(row)
                break

    assert sum(len(v) for v in splits.values()) == len(rows)
    return splits


def make_meta_index_row(row: Dict[str, Any], split: str) -> Dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "pattern": row.get("pattern"),
        "split": split,
        "generator_version": row.get("generator_version"),
        "family_stratum": get_nested(row, "generation_meta.family_semantics.family_stratum", {}),
    }


def infer_image_paths(root: Path, pattern: str, sample_id: str) -> Tuple[Optional[str], Optional[str]]:
    release_root = root / "release"
    top = release_root / "images_v3" / pattern / f"{sample_id}_topdown.png"
    rad = release_root / "images_v3" / pattern / f"{sample_id}_radar.png"
    top_rel = str(top.relative_to(root)) if top.exists() else None
    rad_rel = str(rad.relative_to(root)) if rad.exists() else None
    return top_rel, rad_rel


def load_rendered_image_index(root: Path) -> Dict[str, Dict[str, str]]:
    """Best-effort loader for release/manifests/rendered_image_index_v3.jsonl.
    Supports multiple possible schemas and falls back to path inference if needed.
    """
    idx_path = root / "release" / "manifests" / "rendered_image_index_v3.jsonl"
    mapping: Dict[str, Dict[str, str]] = {}
    if not idx_path.exists():
        return mapping
    for row in read_jsonl(idx_path):
        sample_id = row.get("sample_id") or row.get("id")
        if not sample_id:
            continue
        top = (
            row.get("topdown_image")
            or row.get("topdown_path")
            or get_nested(row, "images.topdown")
            or get_nested(row, "paths.topdown")
        )
        rad = (
            row.get("radar_image")
            or row.get("radar_path")
            or get_nested(row, "images.radar")
            or get_nested(row, "paths.radar")
        )
        # Normalize to root-relative if absolute and under root.
        def norm(p: Any) -> Optional[str]:
            if not p:
                return None
            sp = str(p)
            pp = Path(sp)
            if pp.is_absolute():
                try:
                    return str(pp.relative_to(root))
                except Exception:
                    return sp
            return sp
        mapping[str(sample_id)] = {
            "topdown_image": norm(top) or None,
            "radar_image": norm(rad) or None,
        }
    return mapping


def inject_image_paths(row: Dict[str, Any], root: Path, image_index: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    sample_id = str(out.get("sample_id"))
    pattern = str(out.get("pattern"))

    top = out.get("topdown_image") or get_nested(out, "inputs.topdown_image")
    rad = out.get("radar_image") or get_nested(out, "inputs.radar_image")

    # Prefer existing values, else manifest index, else deterministic inference.
    if not top or not rad:
        idx = image_index.get(sample_id, {})
        top = top or idx.get("topdown_image")
        rad = rad or idx.get("radar_image")
    if not top or not rad:
        itop, irad = infer_image_paths(root, pattern, sample_id)
        top = top or itop
        rad = rad or irad

    out["topdown_image"] = top
    out["radar_image"] = rad
    out.setdefault("inputs", {})
    out["inputs"]["topdown_image"] = top
    out["inputs"]["radar_image"] = rad
    return out


def validate_images(rows: List[Dict[str, Any]], root: Path) -> Tuple[int, List[Dict[str, Any]]]:
    missing: List[Dict[str, Any]] = []
    ok = 0
    for row in rows:
        top = row.get("topdown_image") or get_nested(row, "inputs.topdown_image")
        rad = row.get("radar_image") or get_nested(row, "inputs.radar_image")
        top_ok = bool(top and (root / str(top)).exists())
        rad_ok = bool(rad and (root / str(rad)).exists())
        if top_ok and rad_ok:
            ok += 1
        else:
            if len(missing) < 20:
                missing.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "pattern": row.get("pattern"),
                        "topdown_image": top,
                        "radar_image": rad,
                        "topdown_exists": top_ok,
                        "radar_exists": rad_ok,
                    }
                )
    return ok, missing


def load_source_rows(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    accepted_dir = root / "accepted_pool"
    if accepted_dir.exists():
        fam_rows = {
            fam: read_jsonl(accepted_dir / fam / "accepted_candidates.jsonl") for fam in FAMILIES
        }
        if any(fam_rows.values()):
            return fam_rows

    # Fallback to already-released all samples, grouped by family.
    all_release = root / "release" / "all" / "all_release_samples.jsonl"
    if all_release.exists():
        rows = read_jsonl(all_release)
        grouped: Dict[str, List[Dict[str, Any]]] = {fam: [] for fam in FAMILIES}
        for row in rows:
            fam = str(row.get("pattern"))
            grouped.setdefault(fam, []).append(row)
        return grouped

    raise FileNotFoundError(
        f"Could not find accepted_pool or release/all/all_release_samples.jsonl under {root}"
    )


def to_evaluator_view_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "pattern": row.get("pattern"),
        "split": row.get("split"),
        "topdown_image": row.get("topdown_image"),
        "radar_image": row.get("radar_image"),
        "inputs": row.get("inputs", {}),
        "labels": row.get("labels", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build native release splits and preserve topdown/radar image paths in every emitted sample."
    )
    ap.add_argument("--root", required=True, help="benchmark generation work directory")
    ap.add_argument("--seed", type=int, default=20260326)
    ap.add_argument("--release-name", default="benchmark_native_release_v1")
    ap.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Do not fail if some samples are missing topdown/radar files. By default the script fails.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    release_root = root / "release"
    image_index = load_rendered_image_index(root)
    family_rows = load_source_rows(root)

    family_split_counts: Dict[str, Dict[str, int]] = {}
    combined: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for fam, rows in family_rows.items():
        splits = stratified_split_family(rows, args.seed + sum(ord(c) for c in fam))
        family_split_counts[fam] = {k: len(v) for k, v in splits.items()}
        family_split_counts[fam]["total"] = len(rows)

        for split, rs in splits.items():
            for row in rs:
                row2 = inject_image_paths(row, root, image_index)
                row2.setdefault("generation_meta", {}).setdefault("quality_flags", {})[
                    "accepted_into_release_split"
                ] = True
                row2["split"] = split
                combined[split].append(row2)

    all_rows = combined["train"] + combined["val"] + combined["test"]
    ok_count, missing_examples = validate_images(all_rows, root)
    if missing_examples and not args.allow_missing_images:
        raise ValueError(
            "Missing topdown/radar image files for some released samples. "
            f"Validated ok={ok_count}/{len(all_rows)}. Examples: {missing_examples}"
        )

    # Write full release rows and meta index.
    write_jsonl(release_root / "all" / "all_release_samples.jsonl", all_rows)
    write_jsonl(
        release_root / "all" / "all_release_meta_index.jsonl",
        [make_meta_index_row(r, r.get("split", "")) for r in all_rows],
    )

    # Write splits + meta index.
    for split in ["train", "val", "test"]:
        write_jsonl(release_root / "splits" / f"{split}.jsonl", combined[split])
        write_jsonl(
            release_root / "splits" / f"{split}_meta_index.jsonl",
            [make_meta_index_row(r, split) for r in combined[split]],
        )

    # Write evaluator_view preserving image paths in inputs/top-level.
    ev_all = [to_evaluator_view_row(r) for r in all_rows]
    write_jsonl(release_root / "evaluator_view" / "all.jsonl", ev_all)
    for split in ["train", "val", "test"]:
        write_jsonl(
            release_root / "evaluator_view" / f"{split}.jsonl",
            [to_evaluator_view_row(r) for r in combined[split]],
        )

    split_counts = {k: len(v) for k, v in combined.items()}
    manifests = {
        "release_manifest": {
            "release_name": args.release_name,
            "total_samples": len(all_rows),
            "family_counts": {fam: len(rows) for fam, rows in family_rows.items()},
            "split_counts": split_counts,
            "image_validation": {
                "ok_count": ok_count,
                "total": len(all_rows),
                "all_present": ok_count == len(all_rows),
            },
        },
        "split_manifest": {
            "seed": args.seed,
            "policy": "familywise_stratified_70_15_15_with_nonempty_guards",
            "family_split_counts": family_split_counts,
            "image_source": {
                "manifest_index_used": bool(image_index),
                "fallback_pattern": "release/images_v3/{pattern}/{sample_id}_{view}.png",
            },
        },
    }
    write_json(release_root / "manifests" / "release_manifest.json", manifests["release_manifest"])
    write_json(release_root / "manifests" / "split_manifest.json", manifests["split_manifest"])

    print(
        json.dumps(
            {
                "release_name": args.release_name,
                "total_samples": len(all_rows),
                "split_counts": split_counts,
                "family_split_counts": family_split_counts,
                "image_validation": {
                    "ok_count": ok_count,
                    "total": len(all_rows),
                    "all_present": ok_count == len(all_rows),
                    "missing_examples": missing_examples,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

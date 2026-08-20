#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

FAMILIES = ["tss_crossing", "responsibility_mix", "channel_crossing_impede", "restricted_multi"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def structure_signature(row: Dict[str, Any]) -> str:
    payload = {
        "pattern": row.get("pattern"),
        "family_latent": row.get("family_latent", {}),
        "family_stratum": get_nested(row, "generation_meta.family_semantics.family_stratum", {}) or {},
        "triggered_rules": sorted(get_nested(row, "labels.triggered_rules", []) or []),
        "suppression_records": get_nested(row, "labels.suppression_records", []) or [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def rounded_geometry_signature(row: Dict[str, Any]) -> str:
    own = get_nested(row, "scene_spec.ownship", {}) or {}
    tgts = get_nested(row, "scene_spec.targets", []) or []
    def round_ent(ent: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "x": round(float(ent.get("x_m", 0.0)) / 50.0),
            "y": round(float(ent.get("y_m", 0.0)) / 50.0),
            "hdg": round(float(ent.get("heading_deg", 0.0)) / 15.0),
            "spd": round(float(ent.get("speed_mps", 0.0)) / 1.0),
        }
    payload = {
        "pattern": row.get("pattern"),
        "own": round_ent(own),
        "targets": sorted([round_ent(t) for t in tgts], key=lambda x: json.dumps(x, sort_keys=True)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def summarize_family(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sid_counter = Counter(r.get("sample_id") for r in rows)
    struct_counter = Counter(structure_signature(r) for r in rows)
    geom_counter = Counter(rounded_geometry_signature(r) for r in rows)

    dup_sample_ids = {k: v for k, v in sid_counter.items() if k and v > 1}
    dup_struct = {k: v for k, v in struct_counter.items() if v > 1}
    dup_geom = {k: v for k, v in geom_counter.items() if v > 1}

    return {
        "row_count": len(rows),
        "duplicate_sample_id_count": sum(v - 1 for v in dup_sample_ids.values()),
        "duplicate_structure_signature_count": sum(v - 1 for v in dup_struct.values()),
        "duplicate_geometry_signature_count": sum(v - 1 for v in dup_geom.values()),
        "unique_structure_signature_count": len(struct_counter),
        "unique_geometry_signature_count": len(geom_counter),
        "duplicate_sample_ids_head": list(dup_sample_ids.items())[:10],
        "duplicate_structure_signatures_head": list(dup_struct.items())[:10],
        "duplicate_geometry_signatures_head": list(dup_geom.items())[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build duplicate QA report for current native release.")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = read_jsonl(root / "release" / "all" / "all_release_samples.jsonl")
    report = {
        "families": {fam: summarize_family([r for r in rows if r.get("pattern") == fam]) for fam in FAMILIES},
        "overall": summarize_family(rows),
    }
    out = root / "reports" / "duplicate_reports" / "duplicate_qa_report.json"
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

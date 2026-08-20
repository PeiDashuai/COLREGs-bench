#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FAMILIES = [
    "tss_crossing",
    "responsibility_mix",
    "channel_crossing_impede",
    "restricted_multi",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit structured labels for all benchmark families.")
    p.add_argument("--root", type=str, required=True)
    p.add_argument(
        "--source-mode",
        choices=["auto", "release_all", "accepted_pool"],
        default="auto",
        help="Input source selection. auto prefers release/all/all_release_samples.jsonl when available.",
    )
    p.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Output JSON report path. Default: reports/release_reports/structured_label_audit_report.json",
    )
    p.add_argument(
        "--failures-jsonl",
        type=str,
        default=None,
        help="Optional JSONL dump of failing samples.",
    )
    p.add_argument("--max-failures-head", type=int, default=20)
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(root: Path, source_mode: str) -> Tuple[List[Dict[str, Any]], str]:
    release_path = root / "release/all/all_release_samples.jsonl"
    if source_mode in {"auto", "release_all"} and release_path.exists():
        return read_jsonl(release_path), str(release_path)
    if source_mode == "release_all":
        raise FileNotFoundError(release_path)

    rows: List[Dict[str, Any]] = []
    used = []
    for fam in FAMILIES:
        p = root / "accepted_pool" / fam / "accepted_candidates.jsonl"
        if p.exists():
            fam_rows = read_jsonl(p)
            rows.extend(fam_rows)
            used.append(str(p))
    if not rows:
        raise FileNotFoundError("No accepted_pool JSONL files found.")
    return rows, ",".join(used)


def is_nonempty_text(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def get_explanation_text(step: Dict[str, Any]) -> str:
    for key in ("text", "summary", "note"):
        val = step.get(key)
        if is_nonempty_text(val):
            return str(val).strip()
    return ""


def flatten_support_rules(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "support_rules" and isinstance(v, list):
                out.extend([str(x) for x in v if isinstance(x, (str, int, float))])
            else:
                out.extend(flatten_support_rules(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(flatten_support_rules(v))
    return out


def get_generation_meta_family_semantics(row: Dict[str, Any]) -> Dict[str, Any]:
    gm = row.get("generation_meta")
    if isinstance(gm, dict):
        fam = gm.get("family_semantics")
        if isinstance(fam, dict):
            return fam
    return {}


def infer_observed_status_combo(targets: List[Dict[str, Any]]) -> Optional[str]:
    classes = {str(t.get("vessel_class")) for t in targets if isinstance(t, dict) and t.get("vessel_class") is not None}
    ordinary = {"power_driven", "sailing"}
    special = {"fishing", "ram", "nuc", "constrained_by_draft"}
    if not classes:
        return None
    if classes <= ordinary and classes == {"power_driven", "sailing"}:
        return "power_sailing"
    if any(c in special for c in classes):
        return "mixed_multi_status"
    if classes == {"power_driven"}:
        return "power_only"
    if classes == {"sailing"}:
        return "sailing_only"
    return "mixed_multi_status"


def collect_stage_names(explanation_steps: List[Dict[str, Any]]) -> List[str]:
    return [str(s.get("stage")) for s in explanation_steps if isinstance(s, dict) and s.get("stage") is not None]


def check_basic_label_shape(labels: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in [
        "triggered_rules",
        "target_summaries",
        "primitive_ledger_summary",
        "suppression_records",
        "explanation_steps",
        "maneuver_allowed",
        "maneuver_forbidden",
    ]:
        if key not in labels:
            errors.append(f"missing_labels_{key}")
    if not isinstance(labels.get("triggered_rules"), list) or not labels.get("triggered_rules"):
        errors.append("triggered_rules_empty_or_invalid")
    if not isinstance(labels.get("target_summaries"), list) or not labels.get("target_summaries"):
        errors.append("target_summaries_empty_or_invalid")
    if not isinstance(labels.get("explanation_steps"), list) or not labels.get("explanation_steps"):
        errors.append("explanation_steps_empty_or_invalid")
    return errors


def audit_target_summaries(row: Dict[str, Any], labels: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    tss = labels.get("target_summaries") or []
    trig = set(map(str, labels.get("triggered_rules") or []))
    seen_ids = set()
    for idx, ts in enumerate(tss, 1):
        if not isinstance(ts, dict):
            errors.append("target_summary_non_dict")
            continue
        tid = ts.get("target_id")
        if not is_nonempty_text(tid):
            errors.append("target_summary_missing_target_id")
        else:
            if tid in seen_ids:
                errors.append("target_summary_duplicate_target_id")
            seen_ids.add(tid)
        if not is_nonempty_text(ts.get("relative_bearing_sector")):
            errors.append("target_summary_missing_bearing_sector")
        if not isinstance(ts.get("triggered_rule_ids"), list) or not ts.get("triggered_rule_ids"):
            errors.append("target_summary_missing_triggered_rule_ids")
        else:
            ts_rules = [str(x) for x in ts.get("triggered_rule_ids")]
            if not set(ts_rules).issubset(trig):
                warnings.append("target_summary_rule_not_subset_of_triggered_rules")
        for key in ("cpa_m", "tcpa_s"):
            val = ts.get(key)
            if not isinstance(val, (int, float)):
                errors.append(f"target_summary_invalid_{key}")
            elif float(val) < 0:
                errors.append(f"target_summary_negative_{key}")
        for key in ("risk_of_collision", "collision_imminent", "closing"):
            if key in ts and not isinstance(ts.get(key), bool):
                errors.append(f"target_summary_invalid_{key}")
    scene_targets = row.get("scene_spec", {}).get("targets")
    if isinstance(scene_targets, list) and scene_targets:
        scene_count = len(scene_targets)
        if len(tss) != scene_count:
            warnings.append("target_summary_count_mismatch_scene_targets")
    return errors, warnings


def audit_maneuvers(labels: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    allowed = labels.get("maneuver_allowed") or []
    forbidden = labels.get("maneuver_forbidden") or []
    if not isinstance(allowed, list):
        errors.append("maneuver_allowed_invalid")
        allowed = []
    if not isinstance(forbidden, list):
        errors.append("maneuver_forbidden_invalid")
        forbidden = []
    if set(map(str, allowed)) & set(map(str, forbidden)):
        errors.append("maneuver_allow_forbid_overlap")
    if not allowed and not forbidden:
        errors.append("maneuver_sets_both_empty")
    return errors


def audit_primitive_ledger(labels: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    ledger = labels.get("primitive_ledger_summary")
    if not isinstance(ledger, dict) or not ledger:
        errors.append("primitive_ledger_summary_invalid")
        return errors, warnings
    support = set(flatten_support_rules(ledger))
    trig = set(map(str, labels.get("triggered_rules") or []))
    if support and not support.issubset(trig):
        warnings.append("primitive_ledger_support_rules_not_subset_of_triggered_rules")
    if not support:
        warnings.append("primitive_ledger_support_rules_empty")
    return errors, warnings


def extract_suppressed_from_records(records: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for key in ("suppressed_rules", "suppressed_rule_ids", "rules_suppressed"):
            val = rec.get(key)
            if isinstance(val, list):
                out.extend([str(x) for x in val])
        # fallback: if a record points to a single rule field
        for key in ("suppressed_rule", "rule_id", "rule"):
            val = rec.get(key)
            if isinstance(val, str):
                out.append(val)
    return out


def audit_suppression(labels: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    suppressed_rules = [str(x) for x in (labels.get("suppressed_rules") or [])]
    records = labels.get("suppression_records") or []
    if not isinstance(records, list):
        errors.append("suppression_records_invalid")
        return errors, warnings
    from_records = extract_suppressed_from_records(records)
    if suppressed_rules and records and set(suppressed_rules) != set(from_records):
        warnings.append("suppressed_rules_not_matching_suppression_records")
    if suppressed_rules and not records:
        warnings.append("suppressed_rules_present_but_records_empty")
    if records and not suppressed_rules and not from_records:
        warnings.append("suppression_records_present_without_explicit_suppressed_rules")
    return errors, warnings


def audit_explanation(row: Dict[str, Any], labels: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    steps = labels.get("explanation_steps") or []
    stages = collect_stage_names(steps)
    texts = [get_explanation_text(s) for s in steps if isinstance(s, dict)]
    if not stages:
        errors.append("explanation_stages_missing")
        return errors, warnings
    for s, t in zip(stages, texts):
        if not t:
            errors.append(f"explanation_stage_empty_text:{s}")

    family = str(row.get("pattern") or row.get("family") or "")
    joined = " ".join(texts).lower()
    stage_set = set(stages)
    if family == "restricted_multi":
        if not any(x in stage_set for x in ["global_stage", "global_resolution", "conflict_and_suppression"]):
            errors.append("restricted_multi_missing_global_explanation_stage")
        if "global" not in joined and "retained" not in joined and "suppression" not in joined:
            errors.append("restricted_multi_global_explanation_not_evident")
    elif family == "responsibility_mix":
        if "urgency" not in stage_set and "urgency" not in joined:
            errors.append("responsibility_mix_missing_urgency_explanation")
    elif family == "channel_crossing_impede":
        trig = set(map(str, labels.get("triggered_rules") or []))
        if not any(r.startswith("COLREG_R09") for r in trig):
            errors.append("channel_crossing_impede_missing_rule9_family")
        if "channel" not in joined:
            warnings.append("channel_crossing_impede_channel_not_mentioned_in_explanation")
    elif family == "tss_crossing":
        trig = set(map(str, labels.get("triggered_rules") or []))
        if not any(r.startswith("COLREG_R10") for r in trig):
            errors.append("tss_crossing_missing_rule10_family")
        if "tss" not in joined and "rule 10" not in joined and "traffic separation" not in joined:
            warnings.append("tss_crossing_tss_not_mentioned_in_explanation")
    return errors, warnings


def audit_generation_meta(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    fam = get_generation_meta_family_semantics(row)
    if fam:
        if fam.get("family_invariant_pass") is False:
            errors.append("generation_meta_family_invariant_fail")
        la = fam.get("latent_alignment")
        if isinstance(la, dict):
            for k, v in la.items():
                if isinstance(v, bool) and not v:
                    errors.append(f"latent_alignment_fail:{k}")
    else:
        warnings.append("generation_meta_family_semantics_missing")
    return errors, warnings


def audit_family_specific_status(row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    family = str(row.get("pattern") or row.get("family") or "")
    latent = row.get("family_latent") or {}
    scene_targets = row.get("scene_spec", {}).get("targets") or []
    if family == "responsibility_mix":
        latent_combo = latent.get("target_status_combo")
        if latent_combo == "mixed_multi_status" and isinstance(scene_targets, list) and scene_targets:
            observed = infer_observed_status_combo(scene_targets)
            if observed != "mixed_multi_status":
                errors.append("responsibility_mix_target_status_combo_not_realized")
    return errors


def audit_row(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    labels = row.get("labels")
    if not isinstance(labels, dict):
        return ["labels_missing_or_invalid"], warnings

    errors.extend(check_basic_label_shape(labels))
    e, w = audit_target_summaries(row, labels)
    errors.extend(e)
    warnings.extend(w)
    errors.extend(audit_maneuvers(labels))
    e, w = audit_primitive_ledger(labels)
    errors.extend(e)
    warnings.extend(w)
    e, w = audit_suppression(labels)
    errors.extend(e)
    warnings.extend(w)
    e, w = audit_explanation(row, labels)
    errors.extend(e)
    warnings.extend(w)
    e, w = audit_generation_meta(row)
    errors.extend(e)
    warnings.extend(w)
    errors.extend(audit_family_specific_status(row))

    return sorted(set(errors)), sorted(set(warnings))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows, source = load_rows(root, args.source_mode)

    report_json = Path(args.report_json) if args.report_json else root / "reports/release_reports/structured_label_audit_report.json"
    failures_jsonl = Path(args.failures_jsonl) if args.failures_jsonl else root / "reports/release_reports/structured_label_audit_failures.jsonl"

    family_stats: Dict[str, Dict[str, Any]] = {fam: {
        "row_count": 0,
        "pass_count": 0,
        "fail_count": 0,
        "warning_count": 0,
        "error_histogram": Counter(),
        "warning_histogram": Counter(),
    } for fam in FAMILIES}
    overall_errors = Counter()
    overall_warnings = Counter()
    failures: List[Dict[str, Any]] = []

    for row in rows:
        fam = str(row.get("pattern") or row.get("family") or "unknown")
        if fam not in family_stats:
            family_stats[fam] = {
                "row_count": 0, "pass_count": 0, "fail_count": 0, "warning_count": 0,
                "error_histogram": Counter(), "warning_histogram": Counter(),
            }
        family_stats[fam]["row_count"] += 1
        errors, warnings = audit_row(row)
        if errors:
            family_stats[fam]["fail_count"] += 1
            for e in errors:
                family_stats[fam]["error_histogram"][e] += 1
                overall_errors[e] += 1
            failures.append({
                "sample_id": row.get("sample_id"),
                "pattern": fam,
                "errors": errors,
                "warnings": warnings,
            })
        else:
            family_stats[fam]["pass_count"] += 1
        if warnings:
            family_stats[fam]["warning_count"] += 1
            for w in warnings:
                family_stats[fam]["warning_histogram"][w] += 1
                overall_warnings[w] += 1

    total_rows = len(rows)
    total_fail = len(failures)
    total_pass = total_rows - total_fail
    report = {
        "source": source,
        "row_count": total_rows,
        "pass_count": total_pass,
        "fail_count": total_fail,
        "warning_row_count": sum(1 for fam in family_stats.values() if fam["warning_count"] > 0),
        "family_stats": {
            fam: {
                "row_count": stats["row_count"],
                "pass_count": stats["pass_count"],
                "fail_count": stats["fail_count"],
                "warning_count": stats["warning_count"],
                "error_histogram": dict(stats["error_histogram"]),
                "warning_histogram": dict(stats["warning_histogram"]),
            }
            for fam, stats in family_stats.items()
        },
        "overall_error_histogram": dict(overall_errors),
        "overall_warning_histogram": dict(overall_warnings),
        "failures_head": failures[: args.max_failures_head],
    }
    write_json(report_json, report)
    write_jsonl(failures_jsonl, failures)
    print(f"Rows audited: {total_rows}")
    print(f"Pass: {total_pass}")
    print(f"Fail: {total_fail}")
    print(f"Report: {report_json}")
    print(f"Failures JSONL: {failures_jsonl}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

FAMILIES = ['tss_crossing','responsibility_mix','channel_crossing_impede','restricted_multi']


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def get_nested(d: Dict[str, Any], path: str, default: Any=None) -> Any:
    cur: Any = d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        x=v.strip().lower()
        if x in {'true','1','yes','y'}:
            return True
        if x in {'false','0','no','n'}:
            return False
    return None


def latent_alignment_pass(row: Dict[str, Any]) -> bool:
    gm=row.get('generation_meta',{})
    fs=gm.get('family_semantics',{})
    la=fs.get('latent_alignment')
    if isinstance(la, dict):
        if 'pass' in la:
            return as_bool(la.get('pass')) is True
        # if all explicit *_match are True and no errors field
        flags=[v for k,v in la.items() if k.endswith('_match') or k.endswith('_present')]
        if flags and all(as_bool(x) is True for x in flags):
            return True
    qf=gm.get('quality_flags',{})
    return as_bool(qf.get('passes_latent_alignment')) is True


def family_invariant_pass(row: Dict[str, Any]) -> bool:
    gm=row.get('generation_meta',{})
    fs=gm.get('family_semantics',{})
    qf=gm.get('quality_flags',{})
    return (
        as_bool(fs.get('family_invariant_pass')) is True
        or as_bool(qf.get('passes_family_invariants')) is True
        or as_bool(qf.get('passes_invariant_gate')) is True
    )


def release_ready(row: Dict[str, Any]) -> bool:
    gm=row.get('generation_meta',{})
    qf=gm.get('quality_flags',{})
    if as_bool(qf.get('passes_release_readiness')) is True:
        return True
    # fallback for older families
    if family_invariant_pass(row) and latent_alignment_pass(row):
        reasons = qf.get('rejection_reasons') or []
        return len(reasons) == 0
    return False


def family_stratum(row: Dict[str, Any]) -> Dict[str, Any]:
    return get_nested(row, 'generation_meta.family_semantics.family_stratum', {}) or {}


def signature_string(row: Dict[str, Any]) -> str:
    fs = get_nested(row, 'generation_meta.family_semantics', {}) or {}
    sig = get_nested(row, 'generation_meta.signatures', {}) or {}
    res = get_nested(row, 'generation_meta.resolution_structure', {}) or {}
    parts = {
        'pattern': row.get('pattern'),
        'family_stratum': family_stratum(row),
        'rule_signature': fs.get('rule_signature') or get_nested(sig, 'rule_signature.triggered_rules_sorted', []),
        'suppression_signature': fs.get('suppression_signature') or res.get('suppression_type') or get_nested(sig, 'resolution_signature.suppression_type'),
    }
    return json.dumps(parts, ensure_ascii=False, sort_keys=True)


def signature_hash(row: Dict[str, Any]) -> str:
    return hashlib.sha256(signature_string(row).encode('utf-8')).hexdigest()


def accept_family(root: Path, family: str, dedupe_on_signature: bool) -> Dict[str, Any]:
    ann_path = root/'annotated_pool'/family/'candidates_annotated.jsonl'
    rows = read_jsonl(ann_path)
    accepted=[]
    meta=[]
    seen_ids=set()
    seen_sig=set()
    rej=Counter()
    dup=Counter()

    for row in rows:
        sid=row.get('sample_id')
        if sid in seen_ids:
            rej['duplicate_sample_id'] += 1
            continue
        seen_ids.add(sid)
        if not family_invariant_pass(row):
            rej['family_invariant_fail'] += 1
            continue
        if not latent_alignment_pass(row):
            rej['latent_alignment_fail'] += 1
            continue
        if not release_ready(row):
            rej['release_readiness_fail'] += 1
            continue
        sig = signature_hash(row)
        if dedupe_on_signature and sig in seen_sig:
            dup['duplicate_signature'] += 1
            continue
        seen_sig.add(sig)
        row.setdefault('generation_meta', {}).setdefault('quality_flags', {})['accepted_into_candidate_pool'] = True
        row['generation_meta']['quality_flags']['accepted_into_release_split'] = False
        accepted.append(row)
        meta.append({
            'sample_id': sid,
            'pattern': row.get('pattern'),
            'generator_version': row.get('generator_version'),
            'family_stratum': family_stratum(row),
            'signature_hash': sig,
        })

    out_path = root/'accepted_pool'/family/'accepted_candidates.jsonl'
    idx_path = root/'accepted_pool'/family/'accepted_meta_index.jsonl'
    write_jsonl(out_path, accepted)
    write_jsonl(idx_path, meta)

    report = {
        'family': family,
        'total_rows': len(rows),
        'accepted_count': len(accepted),
        'rejected_count': len(rows)-len(accepted),
        'rejection_histogram': dict(rej),
        'dedupe_histogram': dict(dup),
        'accepted_file': str(out_path),
        'accepted_meta_index_file': str(idx_path),
    }
    write_json(root/'reports'/'acceptance_reports'/f'{family}_acceptance_report.json', report)
    return report


def main() -> int:
    ap=argparse.ArgumentParser(description='Acceptance gate for current native benchmark outputs.')
    ap.add_argument('--root', required=True)
    ap.add_argument('--family', choices=FAMILIES+['all'], default='all')
    ap.add_argument('--dedupe-on-signature', action='store_true')
    args=ap.parse_args()
    root=Path(args.root)
    fams=FAMILIES if args.family=='all' else [args.family]
    reports=[]
    for fam in fams:
        rep=accept_family(root, fam, args.dedupe_on_signature)
        reports.append(rep)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    summary={'families': reports}
    write_json(root/'reports'/'acceptance_reports'/'overall_acceptance_summary.json', summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

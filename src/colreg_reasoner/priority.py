from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from importlib.resources import files as pkg_files
from typing import Dict, Optional, Tuple

import yaml

from colreg_kernel.rules import RuleSpec


@dataclass(frozen=True)
class PriorityBucket:
    name: str
    priority: int
    articles: Tuple[str, ...]


@dataclass(frozen=True)
class PriorityPolicy:
    version: int
    buckets: Tuple[PriorityBucket, ...]


def default_priorities_path() -> str:
    # packaged default
    return str(pkg_files("colreg_reasoner").joinpath("rules/priorities.yaml"))


@lru_cache(maxsize=4)
def load_priority_policy(path: str | None = None) -> PriorityPolicy:
    path = path or default_priorities_path()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    version = int(data.get("policy_version", 1))
    buckets_raw: Dict[str, dict] = data.get("buckets", {}) or {}
    buckets = []
    for name, spec in buckets_raw.items():
        buckets.append(
            PriorityBucket(
                name=name,
                priority=int(spec.get("priority", 0)),
                articles=tuple(spec.get("articles", []) or []),
            )
        )
    # Sort descending by priority for deterministic ranking
    buckets = sorted(buckets, key=lambda b: (-b.priority, b.name))
    return PriorityPolicy(version=version, buckets=tuple(buckets))


def bucket_for_article(article: str, policy: PriorityPolicy) -> Optional[PriorityBucket]:
    for b in policy.buckets:
        if article in b.articles:
            return b
    return None


def rank_rule(rule: RuleSpec, policy: PriorityPolicy) -> int:
    b = bucket_for_article(rule.article, policy)
    # Unknown articles get lowest priority
    return b.priority if b else 0

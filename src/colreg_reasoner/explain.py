from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ExplanationStep:
    stage: str  # e.g., domain/responsibility/encounter/area/signals
    applied_rules: Tuple[str, ...]
    suppressed_rules: Tuple[str, ...]
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class ExplanationChain:
    steps: Tuple[ExplanationStep, ...]

    def to_dict(self) -> Dict:
        return {
            "steps": [
                {
                    "stage": s.stage,
                    "applied_rules": list(s.applied_rules),
                    "suppressed_rules": list(s.suppressed_rules),
                    "notes": list(s.notes),
                }
                for s in self.steps
            ]
        }

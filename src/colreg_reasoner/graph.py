from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from colreg_kernel.features import Features
from colreg_kernel.rules import RuleSpec

from colreg_reasoner.pairwise import pair_relation, PairRelation


@dataclass(frozen=True)
class RuleEdge:
    a: str
    b: str
    relation: PairRelation
    reason: str


@dataclass(frozen=True)
class RuleGraph:
    """Lightweight graph representation: edges + adjacency for visualization / curriculum metrics."""
    nodes: Tuple[str, ...]
    edges: Tuple[RuleEdge, ...]
    adjacency: Dict[str, Tuple[str, ...]]  # node -> neighbors (undirected)


def build_rule_graph(
    rules: Sequence[RuleSpec],
    features: Features,
) -> RuleGraph:
    rlist = list(rules)
    nodes = tuple(sorted({r.id for r in rlist}))
    edges: List[RuleEdge] = []
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    # Deterministic ordering
    rlist = sorted(rlist, key=lambda r: r.id)
    for i in range(len(rlist)):
        for j in range(i + 1, len(rlist)):
            ra, rb = rlist[i], rlist[j]
            pr = pair_relation(ra, rb, features, assume_applicable_true=False)
            if pr.relation in {PairRelation.CONFLICT, PairRelation.TENSION, PairRelation.COMPATIBLE}:
                edges.append(RuleEdge(a=ra.id, b=rb.id, relation=pr.relation, reason=pr.reason.value))
                adj[ra.id].append(rb.id)
                adj[rb.id].append(ra.id)
    adjacency = {k: tuple(sorted(v)) for k, v in adj.items()}
    edges = sorted(edges, key=lambda e: (e.a, e.b, e.relation.value))
    return RuleGraph(nodes=nodes, edges=tuple(edges), adjacency=adjacency)

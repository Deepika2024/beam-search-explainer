"""Explicit search tree used by the tree-based demonstrations.

Every node carries a heuristic score ``h`` (an *estimate of the remaining cost to
reach a goal*: **lower is better**) and the cost of the edge from its parent.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TreeNode:
    id: int
    label: str
    h: float
    parent: Optional[int]
    depth: int
    edge_cost: float = 1.0
    is_goal: bool = False
    children: List[int] = field(default_factory=list)


class SearchTree:
    """A rooted tree stored as ``id -> TreeNode``. Node ids are assigned in insertion order."""

    def __init__(self, name: str = "", description: str = "") -> None:
        self.nodes: Dict[int, TreeNode] = {}
        self.root: Optional[int] = None
        self.name = name
        self.description = description

    # ---- construction -------------------------------------------------
    def add_node(self, label: str, h: float, parent: Optional[int] = None,
                 edge_cost: float = 1.0, is_goal: bool = False) -> int:
        nid = len(self.nodes)
        if parent is None:
            if self.root is not None:
                raise ValueError("tree already has a root")
            self.root = nid
            depth = 0
        else:
            if parent not in self.nodes:
                raise KeyError(f"unknown parent {parent}")
            depth = self.nodes[parent].depth + 1
        self.nodes[nid] = TreeNode(nid, label, float(h), parent, depth, float(edge_cost), is_goal)
        if parent is not None:
            self.nodes[parent].children.append(nid)
        return nid

    # ---- queries ------------------------------------------------------
    def children(self, nid: int) -> List[int]:
        return self.nodes[nid].children

    def label(self, nid: int) -> str:
        return self.nodes[nid].label

    def path_to(self, nid: int) -> List[int]:
        """Node ids from the root down to ``nid``."""
        out: List[int] = []
        cur: Optional[int] = nid
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out[::-1]

    def path_cost(self, nid: int) -> float:
        """Sum of edge costs along the unique root -> nid path (g-value)."""
        return sum(self.nodes[n].edge_cost for n in self.path_to(nid)[1:])

    def goals(self) -> List[int]:
        return [n.id for n in self.nodes.values() if n.is_goal]

    def optimal_cost(self) -> Optional[float]:
        """Cheapest root->goal path cost, found by looking at *every* goal (exhaustive)."""
        costs = [self.path_cost(g) for g in self.goals()]
        return min(costs) if costs else None

    def max_depth(self) -> int:
        return max(n.depth for n in self.nodes.values()) if self.nodes else 0

    def max_width(self) -> int:
        """Largest number of nodes on any single depth level."""
        counts: Dict[int, int] = {}
        for n in self.nodes.values():
            counts[n.depth] = counts.get(n.depth, 0) + 1
        return max(counts.values()) if counts else 0

    def __len__(self) -> int:
        return len(self.nodes)

    # ---- editing ------------------------------------------------------
    def set_heuristics(self, mapping: Dict[int, float]) -> None:
        for nid, h in mapping.items():
            if nid in self.nodes:
                self.nodes[nid].h = float(h)

    def copy(self) -> "SearchTree":
        return copy.deepcopy(self)

    def to_networkx(self):
        """Directed graph view (edges parent -> child) for analysis with NetworkX."""
        import networkx as nx

        g = nx.DiGraph()
        for n in self.nodes.values():
            g.add_node(n.id, label=n.label, h=n.h, depth=n.depth)
            if n.parent is not None:
                g.add_edge(n.parent, n.id, cost=n.edge_cost)
        return g

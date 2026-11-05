"""Hand-designed and random search trees + the default maze.

All heuristic values are *inputs* to the algorithm. Nothing here stores an
algorithm result: the lab always runs the real search on these trees.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from algorithms.grid import Grid
from algorithms.tree import SearchTree


def N(label: str, h: float, *children: dict, goal: bool = False, cost: float = 1.0) -> dict:
    """Tiny spec builder:  N("A", 5, N("A1", 3), N("A2", 4))."""
    return {"label": label, "h": h, "children": list(children), "goal": goal, "cost": cost}


def build_tree(spec: dict, name: str = "", description: str = "") -> SearchTree:
    t = SearchTree(name, description)

    def rec(node: dict, parent: Optional[int]) -> None:
        nid = t.add_node(node["label"], node["h"], parent, node["cost"], node["goal"])
        for c in node["children"]:
            rec(c, nid)

    rec(spec, None)
    return t


def demo_tree() -> SearchTree:
    """Standard teaching tree: 3 levels below the root, goal G at depth 3 (h = estimated cost to goal)."""
    spec = N("S", 9,
             N("A", 5,
               N("A1", 4, N("A1a", 3), N("A1b", 6)),
               N("A2", 7, N("A2a", 7), N("A2b", 9)),
               N("A3", 6, N("A3a", 5), N("A3b", 8))),
             N("B", 3,
               N("B1", 2, N("G", 0, goal=True), N("B1b", 4)),
               N("B2", 5, N("B2a", 6), N("B2b", 5)),
               N("B3", 4, N("B3a", 2), N("B3b", 7))),
             N("C", 6,
               N("C1", 8, N("C1a", 9), N("C1b", 8)),
               N("C2", 3, N("C2a", 2), N("C2b", 4)),
               N("C3", 7, N("C3a", 6), N("C3b", 7))))
    return build_tree(spec, "Standard demo tree",
                      "A balanced tree where a good heuristic leads towards the goal G.")


def failure_tree() -> SearchTree:
    """Deceptive heuristic: the only goal hangs below nodes that look *worse* than a dead-end branch.

    Goal G sits under C -> C2. C2 is only the 5th best node at depth 2, so a beam
    narrower than 5 prunes it; with k = 1 the search follows the 'attractive' branch A into a dead end.
    """
    spec = N("S", 8,
             N("A", 2,
               N("A1", 1, N("A1a", 1.5), N("A1b", 2)),
               N("A2", 2, N("A2a", 2.5), N("A2b", 3))),
             N("B", 3,
               N("B1", 2.5, N("B1a", 2), N("B1b", 3)),
               N("B2", 3, N("B2a", 3), N("B2b", 4))),
             N("C", 4,
               N("C1", 3.5),
               N("C2", 3.4, N("G", 0, goal=True), N("C2b", 4))),
             N("D", 7, N("D1", 5), N("D2", 6)))
    return build_tree(spec, "Failure demo: a narrow beam discards the only path",
                      "The goal G is reachable only through C -> C2, which a narrow beam prunes.")


def cost_trap_tree() -> SearchTree:
    """A wider beam reaches a goal sooner - but via an expensive edge, so the result is worse than k = 1."""
    spec = N("S", 8,
             N("A", 2,
               N("A1", 1, N("G1", 0, goal=True), N("A1b", 5)),
               N("A2", 4, N("A2a", 6))),
             N("B", 3,
               N("G2", 0, goal=True, cost=10),
               N("B2", 6)),
             N("C", 5, N("C1", 4, N("C1a", 5))))
    return build_tree(spec, "Cost trap: a wider beam is not better",
                      "k = 1 follows A -> A1 -> G1 (cost 3). With k >= 2 the goal G2 (cost 11) enters the beam first.")


def random_tree(branching: int = 3, depth: int = 4, seed: int = 7, noise: float = 1.0,
                max_nodes: int = 400) -> SearchTree:
    """Random tree with one goal at maximum depth.

    h(n) = (number of edges between n and the goal) + uniform noise in [-noise, +noise].
    noise = 0 gives a perfect heuristic; larger noise makes the heuristic misleading.
    """
    if branching < 1 or depth < 1:
        raise ValueError("branching and depth must be >= 1")
    total = sum(branching ** i for i in range(depth + 1))
    if total > max_nodes:
        raise ValueError(f"Tree would have {total} nodes (limit {max_nodes}). Reduce branching or depth.")
    rng = np.random.default_rng(seed)
    t = SearchTree(f"Random tree (b={branching}, d={depth}, seed={seed}, noise={noise:g})",
                   "One goal at the deepest level; heuristic = hop distance + noise.")
    root = t.add_node("S", 0.0)
    frontier = [root]
    for d in range(1, depth + 1):
        nxt = []
        for parent in frontier:
            for _ in range(branching):
                cost = float(rng.integers(1, 4))
                nxt.append(t.add_node(f"{d}.{len(nxt)}", 0.0, parent, edge_cost=cost))
        frontier = nxt
    goal = int(rng.choice(frontier))
    t.nodes[goal].is_goal = True
    import networkx as nx
    dist = nx.single_source_shortest_path_length(t.to_networkx().to_undirected(), goal)
    for nid, node in t.nodes.items():
        if nid == goal:
            node.h = 0.0
        else:
            node.h = max(0.1, round(float(dist[nid]) + float(rng.uniform(-noise, noise)), 1))
    return t


TREE_SCENARIOS: Dict[str, Callable[[], SearchTree]] = {
    "Standard demo tree": demo_tree,
    "Failure demo: narrow beam loses the goal": failure_tree,
    "Cost trap: wider beam is not better": cost_trap_tree,
}


def default_grid() -> Grid:
    """12 x 12 maze: a wall (gap at the bottom) plus a small pocket in front of it.

    The Manhattan heuristic pulls the beam straight into the pocket and the wall,
    so k = 1 fails, k = 2 finds a long detour, and k >= 3 finds the shortest path.
    """
    obstacles = {(r, 6) for r in range(0, 10)} | {(4, 5), (6, 5), (4, 4), (6, 4)}
    return Grid(12, 12, start=(5, 2), goal=(5, 10), obstacles=obstacles)


def uniform_tree(branching: int, depth: int, seed: int = 0) -> SearchTree:
    """Complete b-ary tree of the given depth with random heuristic values and *no goal*.

    Used to compare the closed-form node counts of the theory module with a real run.
    """
    total = sum(branching ** i for i in range(depth + 1))
    if total > 2000:
        raise ValueError("Tree too large to materialise.")
    rng = np.random.default_rng(seed)
    t = SearchTree(f"Uniform tree b={branching}, d={depth}")
    frontier = [t.add_node("S", 0.0)]
    for d in range(1, depth + 1):
        nxt = []
        for p in frontier:
            for _ in range(branching):
                nxt.append(t.add_node(f"{d}.{len(nxt)}", round(float(rng.uniform(0, 10)), 1), p))
        frontier = nxt
    return t

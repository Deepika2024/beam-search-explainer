"""Breadth-First Search on explicit trees and grids (baseline for the comparison lab).

BFS keeps *every* node of the current frontier, so it never discards a path -
it is complete (finite branching) and optimal for unit step costs - but its
memory grows like b^d.
"""
from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Deque, Dict, List, Optional, Set

from algorithms.common import SearchSummary
from algorithms.grid import Cell, Grid
from algorithms.tree import SearchTree


def bfs_tree(tree: SearchTree) -> SearchSummary:
    """BFS from the root; the goal test happens when a node is removed from the queue."""
    t0 = perf_counter()
    root = tree.root
    queue: Deque[int] = deque([root])
    expanded: List[int] = []
    generated: Set[int] = {root}
    max_frontier = 1
    goal: Optional[int] = None
    n_generated = 0
    while queue:
        n = queue.popleft()
        if tree.nodes[n].is_goal:
            goal = n
            break
        expanded.append(n)
        for c in tree.children(n):                 # FIFO queue => shallowest nodes first
            queue.append(c)
            generated.add(c)
            n_generated += 1
        max_frontier = max(max_frontier, len(queue))
    found = goal is not None
    return SearchSummary(
        algorithm="Breadth-First Search", found=found, path=tree.path_to(goal) if found else [],
        cost=tree.path_cost(goal) if found else None, expanded=len(expanded), generated=n_generated,
        pruned=0, max_frontier=max_frontier, time_us=(perf_counter() - t0) * 1e6,
        expanded_order=expanded, generated_ids=generated)


def bfs_grid(grid: Grid) -> SearchSummary:
    """BFS on a grid: returns a shortest path (unit step cost) if one exists."""
    errs = grid.validate()
    if errs:
        raise ValueError(" ".join(errs))
    t0 = perf_counter()
    start, goal = grid.start, grid.goal
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    queue: Deque[Cell] = deque([start])
    expanded: List[Cell] = []
    max_frontier = 1
    n_generated = 0
    found = False
    while queue:
        cur = queue.popleft()
        if cur == goal:
            found = True
            break
        expanded.append(cur)
        for nb in grid.neighbors(cur):
            if nb not in parent:
                parent[nb] = cur
                queue.append(nb)
                n_generated += 1
        max_frontier = max(max_frontier, len(queue))
    path: List[Cell] = []
    if found:
        c: Optional[Cell] = goal
        while c is not None:
            path.append(c)
            c = parent[c]
        path.reverse()
    return SearchSummary(
        algorithm="Breadth-First Search", found=found, path=path,
        cost=float(len(path) - 1) if found else None, expanded=len(expanded), generated=n_generated,
        max_frontier=max_frontier, time_us=(perf_counter() - t0) * 1e6, expanded_order=expanded,
        generated_ids=set(parent.keys()))

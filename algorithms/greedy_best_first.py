"""Greedy Best-First Search: always expand the single open node with the best h(n).

Unlike beam search it never discards a generated node (the open list is
unbounded), so on a finite space with a closed set it will eventually try
everything - complete, but not optimal, and the open list can grow large.
"""
from __future__ import annotations

import heapq
from itertools import count
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

from algorithms.common import SearchSummary
from algorithms.grid import Cell, Grid
from algorithms.tree import SearchTree


def greedy_tree(tree: SearchTree) -> SearchSummary:
    t0 = perf_counter()
    root = tree.root
    tie = count()                                   # insertion counter => stable tie-breaking
    heap: List[Tuple[float, int, int]] = [(tree.nodes[root].h, next(tie), root)]
    expanded: List[int] = []
    generated: Set[int] = {root}
    n_generated = 0
    max_frontier = 1
    goal: Optional[int] = None
    while heap:
        _, _, n = heapq.heappop(heap)
        if tree.nodes[n].is_goal:
            goal = n
            break
        expanded.append(n)
        for c in tree.children(n):
            heapq.heappush(heap, (tree.nodes[c].h, next(tie), c))
            generated.add(c)
            n_generated += 1
        max_frontier = max(max_frontier, len(heap))
    found = goal is not None
    return SearchSummary(
        algorithm="Greedy Best-First", found=found, path=tree.path_to(goal) if found else [],
        cost=tree.path_cost(goal) if found else None, expanded=len(expanded), generated=n_generated,
        max_frontier=max_frontier, time_us=(perf_counter() - t0) * 1e6, expanded_order=expanded,
        generated_ids=generated)


def greedy_grid(grid: Grid) -> SearchSummary:
    errs = grid.validate()
    if errs:
        raise ValueError(" ".join(errs))
    t0 = perf_counter()
    start, goal = grid.start, grid.goal
    tie = count()
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    heap: List[Tuple[int, int, Cell]] = [(grid.manhattan(start, goal), next(tie), start)]
    expanded: List[Cell] = []
    max_frontier = 1
    n_generated = 0
    found = False
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur == goal:
            found = True
            break
        expanded.append(cur)
        for nb in grid.neighbors(cur):
            if nb not in parent:
                parent[nb] = cur
                heapq.heappush(heap, (grid.manhattan(nb, goal), next(tie), nb))
                n_generated += 1
        max_frontier = max(max_frontier, len(heap))
    path: List[Cell] = []
    if found:
        c: Optional[Cell] = goal
        while c is not None:
            path.append(c)
            c = parent[c]
        path.reverse()
    return SearchSummary(
        algorithm="Greedy Best-First", found=found, path=path,
        cost=float(len(path) - 1) if found else None, expanded=len(expanded), generated=n_generated,
        max_frontier=max_frontier, time_us=(perf_counter() - t0) * 1e6, expanded_order=expanded,
        generated_ids=set(parent.keys()))

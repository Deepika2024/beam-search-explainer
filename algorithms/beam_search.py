"""Beam Search on explicit trees and on grids.

The algorithm is *level synchronous*:

    B(0)   = { start }
    C(t+1) = children of every node in B(t)          (candidate set)
    B(t+1) = TopK( C(t+1), k )                       (best k by heuristic score)

Everything the UI shows is produced by really running the algorithm: when
``record=True`` the function stores a snapshot ("step") after each phase so the
interface can replay the run forwards and backwards without re-computing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, TypeVar

from algorithms.common import SearchSummary
from algorithms.grid import Cell, Grid
from algorithms.tree import SearchTree

T = TypeVar("T")

# --- node status labels (shared with the visualisation layer) --------------
UNSEEN, PENDING, EXPANDING, BEAM, EXPLORED, PRUNED, PATH = (
    "unseen", "pending", "expanding", "beam", "explored", "pruned", "path")

# --- phases of one beam level ----------------------------------------------
INIT, EXPAND, GENERATE, SELECT, CHECK, DONE = "init", "expand", "generate", "select", "check", "done"

PHASE_TITLES = {
    INIT: "Initialise", EXPAND: "Expand the beam", GENERATE: "Generate candidates",
    SELECT: "Select top-k and prune", CHECK: "Goal test", DONE: "Finished",
}

PSEUDOCODE: List[str] = [
    "BEAM-SEARCH(start, k, h, is_goal)",                                   # 0
    "    beam <- [ start ]",                                               # 1
    "    while beam is not empty:",                                        # 2
    "        candidates <- [ ]",                                           # 3
    "        for each node n in beam:                # expand whole beam", # 4
    "            candidates <- candidates + children(n)",                  # 5
    "        if candidates is empty: return FAILURE",                      # 6
    "        sort candidates by h(n)                 # best score first",  # 7
    "        beam <- first k candidates              # B(t+1)=TopK(C(t+1),k)",  # 8
    "        discard (prune) every other candidate",                       # 9
    "        if some node in beam is a goal: return its path",             # 10
    "    return FAILURE",                                                  # 11
]


def top_k(candidates: Sequence[T], k: int, key: Callable[[T], float]) -> Tuple[List[T], List[T]]:
    """TopK(C, k): return ``(kept, dropped)``.

    ``sorted`` is stable, so candidates with equal scores keep their generation
    order. If there are fewer than ``k`` candidates all of them are kept.
    """
    ranked = sorted(candidates, key=key)
    return ranked[:k], ranked[k:]


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("Beam width k must be a positive integer (k >= 1).")


# ============================================================================
# Tree version
# ============================================================================
@dataclass
class BeamStep:
    """Snapshot of the search after one phase."""
    index: int
    phase: str
    depth: int                       # depth of the nodes currently in / being expanded from the beam
    beam: List[int]
    expanded: List[int]              # nodes expanded at this level
    candidates: List[int]            # candidates generated at this level
    selected: List[int]
    pruned: List[int]
    ranks: Dict[int, int]            # candidate id -> rank (1 = best)
    reasons: Dict[int, str]          # candidate id -> why it was kept / pruned
    cutoff: Optional[float]          # h of the k-th best candidate
    status: Dict[int, str]           # node id -> status label for drawing
    stats: Dict[str, int]            # cumulative counters
    lines: List[int]                 # pseudocode lines being executed
    message: str
    goal: Optional[int] = None
    failed: bool = False
    path: Optional[List[int]] = None


@dataclass
class BeamResult:
    tree: SearchTree
    k: int
    lower_is_better: bool
    steps: List[BeamStep]
    found: bool
    goal: Optional[int]
    path: List[int]
    cost: Optional[float]
    stats: Dict[str, float]
    expanded_order: List[int]
    generated_ids: Set[int]
    pruned_ids: Set[int]
    failure_reason: str = ""
    lost_goals: List[Tuple[int, int]] = field(default_factory=list)  # (goal, first pruned ancestor)

    def summary(self) -> SearchSummary:
        return SearchSummary(
            algorithm=f"Beam Search (k={self.k})", found=self.found, path=list(self.path), cost=self.cost,
            expanded=int(self.stats["expanded"]), generated=int(self.stats["generated"]),
            pruned=int(self.stats["pruned"]), max_frontier=int(self.stats["max_beam"]),
            time_us=float(self.stats.get("time_us", 0.0)), expanded_order=list(self.expanded_order),
            generated_ids=set(self.generated_ids), pruned_ids=set(self.pruned_ids),
            note=self.failure_reason)


def _lab(tree: SearchTree, ids: Sequence[int]) -> str:
    return ", ".join(tree.label(i) for i in ids) if ids else "none"


def _lost_goals(tree: SearchTree, pruned_all: Set[int]) -> List[Tuple[int, int]]:
    """Goals that still exist in the tree but whose path was cut by pruning."""
    lost = []
    for g in tree.goals():
        for n in tree.path_to(g):
            if n in pruned_all:
                lost.append((g, n))
                break
    return lost


def beam_search(tree: SearchTree, k: int, lower_is_better: bool = True,
                max_depth: Optional[int] = None, record: bool = True) -> BeamResult:
    """Run beam search from the root of ``tree``.

    * ``k`` - beam width.
    * ``lower_is_better`` - True for heuristics that estimate *cost to goal*
      (the default in this lab); False if a higher score means "more promising".
    * A node is goal-tested when it enters the beam, so a goal can only be found
      if it *survives* pruning. This is what makes beam search incomplete.
    * ``record`` - store step snapshots (needed by the UI, skipped when timing).
    """
    _check_k(k)
    if tree.root is None:
        raise ValueError("The search tree is empty.")
    t0 = perf_counter()
    nodes = tree.nodes
    root = tree.root
    sign = 1.0 if lower_is_better else -1.0

    def key(n: int) -> float:                     # f(n) = h(n), sign-flipped if higher is better
        return sign * nodes[n].h

    word = "lowest" if lower_is_better else "highest"
    steps: List[BeamStep] = []
    counters = {"expanded": 0, "generated": 0, "pruned": 0, "max_beam": 1}
    explored: Set[int] = set()
    pruned_all: Set[int] = set()
    generated_ids: Set[int] = {root}
    expanded_order: List[int] = []
    beam: List[int] = [root]
    level = 0
    goal: Optional[int] = None
    failure = ""

    def snap(phase: str, *, expanding: Sequence[int] = (), pending: Sequence[int] = (),
             expanded: Sequence[int] = (), cands: Sequence[int] = (), selected: Sequence[int] = (),
             dropped: Sequence[int] = (), ranks: Optional[Dict[int, int]] = None,
             reasons: Optional[Dict[int, str]] = None, cutoff: Optional[float] = None,
             message: str = "", lines: Sequence[int] = (), goal_id: Optional[int] = None,
             failed: bool = False, path: Optional[List[int]] = None) -> None:
        if not record:
            return
        exp, pend, bm, pth = set(expanding), set(pending), set(beam), set(path or [])
        status: Dict[int, str] = {}
        for nid in nodes:                          # precedence: path > pruned > expanding > beam > pending > explored
            if nid in pth:
                s = PATH
            elif nid in pruned_all:
                s = PRUNED
            elif nid in exp:
                s = EXPANDING
            elif nid in bm:
                s = BEAM
            elif nid in pend:
                s = PENDING
            elif nid in explored:
                s = EXPLORED
            else:
                s = UNSEEN
            status[nid] = s
        steps.append(BeamStep(
            index=len(steps), phase=phase, depth=level, beam=list(beam), expanded=list(expanded),
            candidates=list(cands), selected=list(selected), pruned=list(dropped), ranks=dict(ranks or {}),
            reasons=dict(reasons or {}), cutoff=cutoff, status=status,
            stats={**counters, "beam_size": len(beam), "depth": level}, lines=list(lines),
            message=message, goal=goal_id, failed=failed, path=path))

    # ---- INIT -----------------------------------------------------------
    snap(INIT, lines=[0, 1], message=f"Initialise the beam with the start node {tree.label(root)}. The beam holds 1 node.")
    if nodes[root].is_goal:
        goal = root
    while goal is None:
        if max_depth is not None and level >= max_depth:
            failure = f"Stopped at the maximum depth ({max_depth}) without reaching a goal."
            snap(DONE, lines=[11], failed=True, message=failure)
            break

        # ---- EXPAND: every node in the beam is expanded -------------------
        expanded = list(beam)
        counters["expanded"] += len(expanded)
        expanded_order.extend(expanded)
        snap(EXPAND, expanding=expanded, expanded=expanded, lines=[2, 4],
             message=f"Expand every node in the beam: {_lab(tree, expanded)}.")

        # ---- GENERATE: C(t+1) = all children of the beam ------------------
        candidates = [c for n in expanded for c in tree.children(n)]
        counters["generated"] += len(candidates)
        generated_ids.update(candidates)
        if not candidates:
            explored.update(beam)
            beam = []
            failure = ("The beam ran out of candidates: every node in it is a leaf (dead end), "
                       "and beam search cannot backtrack.")
            lost = _lost_goals(tree, pruned_all)
            if lost:
                g, a = lost[0]
                failure += (f" Goal {tree.label(g)} exists, but its ancestor {tree.label(a)} "
                            f"was pruned at depth {nodes[a].depth}.")
            snap(DONE, expanding=(), expanded=expanded, lines=[6], failed=True, message=failure)
            break
        scored = ", ".join(f"{tree.label(c)} (h={nodes[c].h:g})" for c in candidates)
        snap(GENERATE, expanding=expanded, pending=candidates, expanded=expanded, cands=candidates,
             lines=[3, 5], message=f"{len(candidates)} candidate(s) generated with heuristic scores: {scored}.")

        # ---- SELECT: B(t+1) = TopK(C(t+1), k); everything else is pruned ---
        selected, dropped = top_k(candidates, k, key)
        ranked = selected + dropped
        ranks = {n: i + 1 for i, n in enumerate(ranked)}
        cutoff = nodes[selected[-1]].h
        reasons: Dict[int, str] = {}
        for n in selected:
            reasons[n] = f"rank #{ranks[n]} of {len(ranked)} (within the top {k})"
        for n in dropped:
            h = nodes[n].h
            if h == cutoff:
                reasons[n] = (f"tied with the cutoff score h={h:g}, but ranked #{ranks[n]} of {len(ranked)}; "
                              f"ties go to the node generated first, and only {k} fit in the beam")
            else:
                reasons[n] = (f"h={h:g} ranks #{ranks[n]} of {len(ranked)}; the beam keeps only the {k} "
                              f"{word}-scoring candidate(s) (cutoff h={cutoff:g})")
        explored.update(beam)
        pruned_all.update(dropped)
        counters["pruned"] += len(dropped)
        beam = list(selected)
        level += 1
        counters["max_beam"] = max(counters["max_beam"], len(beam))
        snap(SELECT, expanded=expanded, cands=candidates, selected=selected, dropped=dropped, ranks=ranks,
             reasons=reasons, cutoff=cutoff, lines=[7, 8, 9],
             message=(f"Keep the top {len(selected)} ({word} h): {_lab(tree, selected)}. "
                      f"Prune {len(dropped)}: {_lab(tree, dropped)}."))

        # ---- CHECK: goal test on the new beam -----------------------------
        goals = [n for n in selected if nodes[n].is_goal]
        if goals:
            goal = goals[0]
            snap(CHECK, expanded=expanded, cands=candidates, selected=selected, dropped=dropped, ranks=ranks,
                 reasons=reasons, cutoff=cutoff, lines=[10], goal_id=goal,
                 message=f"Goal {tree.label(goal)} is in the beam, so the search stops.")
        else:
            snap(CHECK, expanded=expanded, cands=candidates, selected=selected, dropped=dropped, ranks=ranks,
                 reasons=reasons, cutoff=cutoff, lines=[10],
                 message=f"No goal in the beam at depth {level}. Continue with the next level.")

    found = goal is not None
    path = tree.path_to(goal) if found else []
    cost = tree.path_cost(goal) if found else None
    if found:
        snap(DONE, lines=[10], goal_id=goal, path=path,
             message=(f"Goal found. Path: {' -> '.join(tree.label(n) for n in path)}  (cost {cost:g})."))
    lost = [] if found else _lost_goals(tree, pruned_all)
    if not found and not failure:
        failure = "Search ended without reaching a goal."
    stats: Dict[str, float] = {**counters, "retained": len(beam), "depth": level,
                               "time_us": (perf_counter() - t0) * 1e6}
    return BeamResult(tree=tree, k=k, lower_is_better=lower_is_better, steps=steps, found=found, goal=goal,
                      path=path, cost=cost, stats=stats, expanded_order=expanded_order,
                      generated_ids=generated_ids, pruned_ids=pruned_all, failure_reason=failure,
                      lost_goals=lost)


# ============================================================================
# Grid (maze) version
# ============================================================================
@dataclass
class GridStep:
    index: int
    level: int
    beam: List[Cell]
    expanded: List[Cell]
    candidates: List[Cell]
    selected: List[Cell]
    pruned: List[Cell]
    explored: FrozenSet[Cell]        # every cell expanded so far
    pruned_all: FrozenSet[Cell]      # every cell pruned so far
    path: Optional[List[Cell]]
    stats: Dict[str, int]
    message: str


@dataclass
class GridBeamResult:
    grid: Grid
    k: int
    steps: List[GridStep]
    found: bool
    path: List[Cell]
    expanded_order: List[Cell]
    stats: Dict[str, float]
    failure_reason: str = ""

    @property
    def path_length(self) -> Optional[int]:
        return len(self.path) - 1 if self.found else None

    def summary(self) -> SearchSummary:
        return SearchSummary(
            algorithm=f"Beam Search (k={self.k})", found=self.found, path=list(self.path),
            cost=float(self.path_length) if self.found else None, expanded=int(self.stats["expanded"]),
            generated=int(self.stats["generated"]), pruned=int(self.stats["pruned"]),
            max_frontier=int(self.stats["max_beam"]), time_us=float(self.stats.get("time_us", 0.0)),
            expanded_order=list(self.expanded_order), note=self.failure_reason)


def beam_search_grid(grid: Grid, k: int, record: bool = True) -> GridBeamResult:
    """Beam search on a grid with the Manhattan-distance heuristic (lower is better).

    Cells that have been generated once are *closed*: they are never generated
    again. Consequently a cell that was pruned stays discarded for the rest of
    the run - exactly the information loss that makes beam search incomplete.
    """
    _check_k(k)
    errs = grid.validate()
    if errs:
        raise ValueError(" ".join(errs))
    t0 = perf_counter()
    start, goal = grid.start, grid.goal
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    closed: Set[Cell] = {start}
    beam: List[Cell] = [start]
    explored: Set[Cell] = set()
    pruned_all: Set[Cell] = set()
    counters = {"expanded": 0, "generated": 0, "pruned": 0, "max_beam": 1}
    order: List[Cell] = []
    steps: List[GridStep] = []
    level = 0
    found = start == goal
    failure = ""

    def snap(expanded: Sequence[Cell], cands: Sequence[Cell], selected: Sequence[Cell],
             dropped: Sequence[Cell], message: str, path: Optional[List[Cell]] = None) -> None:
        if record:
            steps.append(GridStep(
                index=len(steps), level=level, beam=list(beam), expanded=list(expanded),
                candidates=list(cands), selected=list(selected), pruned=list(dropped),
                explored=frozenset(explored), pruned_all=frozenset(pruned_all), path=path,
                stats={**counters, "beam_size": len(beam)}, message=message))

    snap([], [], [start], [], f"Level 0: the beam holds only the start cell {start}.")
    while beam and not found:
        level += 1
        candidates: List[Cell] = []
        for cell in beam:                                  # expand the whole beam
            for nb in grid.neighbors(cell):
                if nb in closed:                           # never generate a cell twice
                    continue
                closed.add(nb)
                parent[nb] = cell
                candidates.append(nb)
        expanded = list(beam)
        counters["expanded"] += len(expanded)
        counters["generated"] += len(candidates)
        order.extend(expanded)
        explored.update(expanded)
        if not candidates:
            beam = []
            failure = ("The beam died: none of its cells has an unvisited free neighbour. "
                       "Beam search cannot backtrack to pruned cells.")
            snap(expanded, [], [], [], f"Level {level}: no candidates - search fails.")
            break
        selected, dropped = top_k(candidates, k, key=lambda c: grid.manhattan(c, goal))
        pruned_all.update(dropped)
        counters["pruned"] += len(dropped)
        beam = list(selected)
        counters["max_beam"] = max(counters["max_beam"], len(beam))
        found = goal in selected
        snap(expanded, candidates, selected, dropped,
             f"Level {level}: expanded {len(expanded)} cell(s), generated {len(candidates)}, "
             f"kept {len(selected)}, pruned {len(dropped)}.")
    path: List[Cell] = []
    if found:
        cur: Optional[Cell] = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        snap([], [], beam, [], f"Goal reached. Path length: {len(path) - 1} moves.", path=path)
    elif not failure:
        failure = "Search ended without reaching the goal."
    stats: Dict[str, float] = {**counters, "time_us": (perf_counter() - t0) * 1e6, "depth": level}
    return GridBeamResult(grid=grid, k=k, steps=steps, found=found, path=path, expanded_order=order,
                          stats=stats, failure_reason=failure)

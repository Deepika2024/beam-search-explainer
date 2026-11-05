"""Closed-form node counts used by the theory module.

Assumptions (stated explicitly because complexity bounds are meaningless without them):
  * every node has exactly ``b`` children (uniform branching factor),
  * the search runs to depth ``d`` without finding a goal (worst case),
  * the beam keeps at most ``k`` nodes per level,
  * evaluating h(n) costs O(1).
"""
from __future__ import annotations

from typing import Dict, List


def level_table(b: int, d: int, k: int) -> List[Dict[str, int]]:
    """Per-level counts for depth 1..d.

    expanded(i)   nodes of the beam at depth i-1 that get expanded  = min(b^(i-1), k)
    generated(i)  candidates created at depth i                      = b * expanded(i)
    retained(i)   nodes kept in the beam at depth i                  = min(generated(i), k)
    bfs(i)        nodes at depth i that BFS must hold                = b^i
    """
    rows = []
    for i in range(1, d + 1):
        expanded = min(b ** (i - 1), k)
        generated = b * expanded
        rows.append({"depth": i, "bfs_frontier": b ** i, "beam_expanded": expanded,
                     "beam_generated": generated, "beam_retained": min(generated, k)})
    return rows


def totals(b: int, d: int, k: int) -> Dict[str, int]:
    rows = level_table(b, d, k)
    return {
        "bfs_generated": sum(r["bfs_frontier"] for r in rows),
        "beam_generated": sum(r["beam_generated"] for r in rows),
        "beam_expanded": sum(r["beam_expanded"] for r in rows),
        "beam_peak_retained": max(r["beam_retained"] for r in rows) if rows else 0,
        "beam_peak_candidates": max(r["beam_generated"] for r in rows) if rows else 0,
        "bfs_peak_frontier": b ** d,
    }

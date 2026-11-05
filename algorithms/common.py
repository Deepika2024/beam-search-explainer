"""Shared result container and timing helper used by every search algorithm.

Nothing in this file knows about Streamlit: the algorithm layer is pure Python
so it can be unit-tested and reused independently of the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, List, Optional, Set


@dataclass
class SearchSummary:
    """Uniform description of one search run, so different algorithms can be compared."""

    algorithm: str
    found: bool
    path: List[Any] = field(default_factory=list)      # node ids (tree) or cells (grid)
    cost: Optional[float] = None                       # path cost if found
    expanded: int = 0                                  # nodes whose successors were generated
    generated: int = 0                                 # successor nodes created
    pruned: int = 0                                    # nodes discarded without being expanded (beam only)
    max_frontier: int = 0                              # largest number of nodes held in memory at once
    time_us: float = 0.0                               # median wall-clock time of the search, microseconds
    expanded_order: List[Any] = field(default_factory=list)
    generated_ids: Set[Any] = field(default_factory=set)
    pruned_ids: Set[Any] = field(default_factory=set)
    note: str = ""


def benchmark(fn: Callable[[], Any], repeats: int = 25) -> float:
    """Median run time of ``fn`` in microseconds.

    The median of several repeats is used because a single timing of a
    sub-millisecond function is dominated by noise.
    """
    times: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = perf_counter()
        fn()
        times.append((perf_counter() - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


def fmt_time(us: float) -> str:
    """Human-friendly time string."""
    if us < 1000:
        return f"{us:.0f} µs"
    return f"{us / 1000:.2f} ms"

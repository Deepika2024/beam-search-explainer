"""Live search-statistics cards (feature B)."""
from __future__ import annotations

from typing import Dict, Optional

from algorithms.common import fmt_time
from components.ui import stat_cards


def beam_stats_cards(stats: Dict[str, int], time_us: Optional[float] = None, found: Optional[bool] = None) -> None:
    """Cumulative counters of a (partial) beam-search run."""
    items = [
        ("Depth", stats.get("depth", 0), "current level"),
        ("Expanded", stats.get("expanded", 0), "nodes expanded"),
        ("Generated", stats.get("generated", 0), "candidates created"),
        ("Retained", stats.get("beam_size", 0), "nodes in the beam now"),
        ("Pruned", stats.get("pruned", 0), "nodes discarded"),
    ]
    classes = ["", "", "", "accent", "warn"]
    if time_us is not None:
        items.append(("Run time", fmt_time(time_us), "full search"))
        classes.append("")
    stat_cards(items, classes)

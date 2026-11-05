"""Module 6 - BFS vs Greedy Best-First vs Beam Search on the same problems (state keys prefixed ``m6_``)."""
from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from algorithms.beam_search import beam_search, beam_search_grid
from algorithms.bfs import bfs_grid, bfs_tree
from algorithms.common import SearchSummary, benchmark, fmt_time
from algorithms.greedy_best_first import greedy_grid, greedy_tree
from components.explanations import COMPARISON_TABLE
from components.ui import callout, legend, show_df, show_plot
from data.sample_search_trees import TREE_SCENARIOS, default_grid, random_tree
from visualizations.maze_visualizer import maze_figure, maze_legend
from visualizations.search_tree import draw_tree, legend_entries, nodes_from_status, status_from_summary

ss = st.session_state
KS = [1, 2, 3, 5, 10]
RANDOM = "Random tree (seed below)"


def _table(rows: List[SearchSummary], unit: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "Algorithm": r.algorithm, "Goal found": "✅ yes" if r.found else "❌ no", "Expanded": r.expanded,
        "Generated": r.generated, "Max nodes held": r.max_frontier, "Pruned": r.pruned,
        f"Path {unit}": f"{r.cost:g}" if r.found else "-", "Time": fmt_time(r.time_us)} for r in rows])


def _takeaways(rows: List[SearchSummary], unit: str) -> None:
    """Neutral, data-driven observations - the lab never claims one algorithm always wins."""
    found = [r for r in rows if r.found]
    lines = []
    if found:
        fewest = min(r.expanded for r in found)
        lines.append("Fewest expansions among the algorithms that found the goal: <b>"
                     + ", ".join(r.algorithm for r in found if r.expanded == fewest) + f"</b> ({fewest}).")
        cheapest = min(r.cost for r in found)
        lines.append(f"Lowest path {unit}: <b>" + ", ".join(r.algorithm for r in found if r.cost == cheapest)
                     + f"</b> ({cheapest:g}).")
        least_mem = min(r.max_frontier for r in found)
        lines.append("Smallest number of nodes held in memory at once: <b>"
                     + ", ".join(r.algorithm for r in found if r.max_frontier == least_mem) + f"</b> ({least_mem}).")
    failed = [r for r in rows if not r.found]
    if failed:
        lines.append("<b>Did not find the goal:</b> " + ", ".join(r.algorithm for r in failed) + ".")
    callout("info", "<br>".join(lines) + "<br><i>Change the scenario or k: the ranking changes. "
                                          "No algorithm is best on every measure.</i>")


def _tree_tab() -> None:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.selectbox("Problem", list(TREE_SCENARIOS) + [RANDOM], key="m6_scenario")
    with c2:
        st.select_slider("Beam width k", KS, key="m6_k")
    if ss.m6_scenario == RANDOM:
        r1, r2 = st.columns(2)
        r1.number_input("Seed", 0, 9999, key="m6_seed", step=1)
        r2.slider("Heuristic noise", 0.0, 4.0, step=0.5, key="m6_noise")
        tree = random_tree(3, 4, int(ss.m6_seed), ss.m6_noise)
    else:
        tree = TREE_SCENARIOS[ss.m6_scenario]()
    st.caption(tree.description + " Heuristic: lower h is better. Edge costs are shown in the path cost.")

    k = ss.m6_k
    funcs = {"Breadth-First Search": lambda: bfs_tree(tree), "Greedy Best-First": lambda: greedy_tree(tree),
             f"Beam Search (k={k})": lambda: beam_search(tree, k, record=False).summary()}
    rows = []
    for name, fn in funcs.items():
        r = fn()
        r.algorithm = name
        r.time_us = benchmark(fn, 15)
        rows.append(r)
    show_df(_table(rows, "cost"))
    cols = st.columns(3)
    for col, r in zip(cols, rows):
        with col:
            status = status_from_summary(tree, r)
            show_plot(draw_tree(nodes_from_status(tree, status), height=380, title=r.algorithm,
                                show_unseen=True), key=f"m6_tree_{r.algorithm}")
    legend(legend_entries())
    st.caption("Blue = expanded, cyan = generated but never expanded (still waiting), grey = pruned (beam only), "
               "yellow = returned path.")
    _takeaways(rows, "cost")


def _maze_tab() -> None:
    grid = ss.m4_grid if "m4_grid" in ss else default_grid()
    st.caption("This tab uses the maze from Module 4 (or the example maze if you have not opened it). "
               "Edit the maze there, then come back.")
    st.select_slider("Beam width k", KS, key="m6_mk")
    k = ss.m6_mk
    errs = grid.validate()
    if errs:
        st.error(" ".join(errs))
        return
    bres = beam_search_grid(grid, k)
    funcs = {"Breadth-First Search": lambda: bfs_grid(grid), "Greedy Best-First": lambda: greedy_grid(grid),
             f"Beam Search (k={k})": lambda: beam_search_grid(grid, k, record=False).summary()}
    rows: List[SearchSummary] = []
    for name, fn in funcs.items():
        r = fn()
        r.algorithm = name
        r.time_us = benchmark(fn, 15)
        rows.append(r)
    show_df(_table(rows, "moves"))
    cols = st.columns(3)
    for col, r in zip(cols, rows):
        with col:
            st.markdown(f"**{r.algorithm}**")
            pruned = bres.steps[-1].pruned_all if r.algorithm.startswith("Beam") else ()
            show_plot(maze_figure(grid, explored=r.expanded_order, pruned=pruned, path=r.path, height=330,
                                  clickable=False), key=f"m6_maze_{r.algorithm}")
    legend(maze_legend())
    _takeaways(rows, "moves")


def render() -> None:
    for key, val in {"m6_scenario": "Cost trap: wider beam is not better", "m6_k": 2, "m6_seed": 7, "m6_noise": 1.5,
                     "m6_mk": 3}.items():
        ss.setdefault(key, val)
    st.markdown("## Algorithm Comparison Lab")
    t1, t2, t3 = st.tabs(["📋 Properties", "🌳 Tree race", "🧱 Maze race"])
    with t1:
        show_df(COMPARISON_TABLE)
        callout("warn", "<b>Beam Search is a trade-off, not an upgrade.</b> It holds far fewer nodes than BFS, but it can "
                        "fail where BFS and (on finite spaces) Greedy Best-First succeed, and its answer may be longer "
                        "or costlier. BFS is optimal only for uniform step costs, and Greedy Best-First is not optimal "
                        "either.")
        st.markdown("""
**Reading the table**

- *Nodes retained*: BFS keeps the entire frontier, Greedy keeps its entire open list, Beam keeps at most *k* nodes per level.
- *Memory*: bounded for beam search (it does not grow with depth beyond the stored path pointers), exponential for BFS.
- *Completeness*: pruning is what costs beam search its completeness. With *k* at least as large as the widest level it never prunes and behaves like BFS.
""")
    with t2:
        _tree_tab()
    with t3:
        _maze_tab()

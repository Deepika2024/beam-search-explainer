"""Module 4 - interactive maze solver (state keys prefixed ``m4_``).

Editing: click a cell (or drag a box to paint several) with the selected tool; the chart is then reset by
changing its key, which clears the Plotly selection. A coordinate-based editor is provided as a fallback.
"""
from __future__ import annotations

import time

import streamlit as st

from algorithms.beam_search import beam_search_grid
from algorithms.bfs import bfs_grid
from algorithms.common import benchmark
from algorithms.grid import Grid, random_maze
from components.ui import button, callout, legend, show_plot, stat_cards
from data.sample_search_trees import default_grid
from visualizations.maze_visualizer import maze_figure, maze_legend

ss = st.session_state
TOOLS = ["Toggle obstacle", "Set start", "Set goal"]


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def _init() -> None:
    if "m4_grid" not in ss:
        ss.m4_grid = default_grid()
    for key, val in {"m4_rows": ss.m4_grid.rows, "m4_cols": ss.m4_grid.cols, "m4_k": 2, "m4_tool": TOOLS[0],
                     "m4_density": 25, "m4_seed": 3, "m4_trace": None, "m4_step": 0, "m4_playing": False,
                     "m4_ver": 0, "m4_speed": 0.4, "m4_er": 0, "m4_ec": 0}.items():
        ss.setdefault(key, val)


def _invalidate() -> None:
    """The maze changed: drop the old run and reset the chart selection."""
    ss.m4_trace, ss.m4_step, ss.m4_playing = None, 0, False
    ss.m4_ver += 1


def _run(animate: bool) -> None:
    g: Grid = ss.m4_grid
    if g.validate():
        return
    k = ss.m4_k
    res = beam_search_grid(g, k)
    ss.m4_trace = res
    ss.m4_time = benchmark(lambda: beam_search_grid(g, k, record=False), 15)
    ss.m4_bfs = bfs_grid(g)                                    # ground truth: does a path exist at all?
    ss.m4_step = 0 if animate else len(res.steps) - 1
    ss.m4_playing = animate


def _execute() -> None:
    _run(False)


def _animate() -> None:
    if ss.m4_trace is None or ss.m4_trace.k != ss.m4_k:
        _run(True)
    else:
        ss.m4_step, ss.m4_playing = 0, True


def _on_k() -> None:
    if ss.m4_trace is not None:
        _run(False)


def _pause() -> None:
    ss.m4_playing = False


def _back() -> None:
    ss.m4_playing = False
    ss.m4_step = max(0, ss.m4_step - 1)


def _forward() -> None:
    if ss.m4_trace is not None:
        ss.m4_step = min(len(ss.m4_trace.steps) - 1, ss.m4_step + 1)


def _resize() -> None:
    g: Grid = ss.m4_grid
    r, c = ss.m4_rows, ss.m4_cols
    inside = lambda cell: cell[0] < r and cell[1] < c
    start = g.start if inside(g.start) else (0, 0)
    goal = g.goal if inside(g.goal) else (r - 1, c - 1)
    if start == goal:
        goal = (r - 1, c - 1) if start != (r - 1, c - 1) else (0, 0)
    obs = {o for o in g.obstacles if inside(o)} - {start, goal}
    ss.m4_grid = Grid(r, c, start, goal, obs)
    _invalidate()


def _random() -> None:
    g: Grid = ss.m4_grid
    ss.m4_grid = random_maze(g.rows, g.cols, ss.m4_density / 100.0, int(ss.m4_seed), g.start, g.goal)
    _invalidate()


def _clear() -> None:
    ss.m4_grid.obstacles.clear()
    _invalidate()


def _restore() -> None:
    ss.m4_grid = default_grid()
    ss.m4_rows, ss.m4_cols = ss.m4_grid.rows, ss.m4_grid.cols
    _invalidate()


def _edit_xy(tool: str) -> None:
    g: Grid = ss.m4_grid
    cell = (int(ss.m4_er), int(ss.m4_ec))
    ok = {"start": g.set_start, "goal": g.set_goal, "obstacle": g.toggle_obstacle}[tool](cell)
    if ok:
        _invalidate()


def _apply_clicks(points) -> bool:
    """Apply the active tool to the clicked / box-selected cells. Returns True if the maze changed."""
    g: Grid = ss.m4_grid
    cells = []
    for p in points:
        try:
            cells.append((int(round(p["y"])), int(round(p["x"]))))
        except (KeyError, TypeError, ValueError):
            continue
    if not cells:
        return False
    if ss.m4_tool == "Set start":
        return g.set_start(cells[0])
    if ss.m4_tool == "Set goal":
        return g.set_goal(cells[0])
    changed = False
    for c in dict.fromkeys(cells):
        changed |= g.toggle_obstacle(c)
    return changed


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _result_message(res, bfs) -> None:
    if res.found:
        shortest = int(bfs.cost) if bfs.found else None
        extra = ""
        if shortest is not None:
            extra = (" This is the shortest possible path." if res.path_length == shortest else
                     f" The shortest path has {shortest} moves, so beam search returned a path that is "
                     f"{res.path_length - shortest} moves longer - beam search does not guarantee the shortest route.")
        callout("ok", f"<b>Goal reached</b> in {res.path_length} moves with k = {res.k}.{extra}")
    elif bfs.found:
        callout("bad", f"<b>Beam search failed with k = {res.k}</b> - but a path of {int(bfs.cost)} moves exists "
                       "(Breadth-First Search finds it). The beam was too narrow: it pruned the cells that lead around "
                       "the obstacles, and beam search cannot go back to them. Increase k and run again.")
    else:
        callout("warn", "<b>The goal is unreachable.</b> The obstacles completely separate start and goal; Breadth-First "
                        "Search confirms that no path exists, so no search algorithm could succeed here.")


def render() -> None:
    _init()
    g: Grid = ss.m4_grid
    st.markdown("## Interactive Maze Solver")
    st.caption("Four-directional moves, unit cost, heuristic h(n) = Manhattan distance to the goal (**lower is better**). "
               "Click cells in the grid with the chosen tool, or drag a box to paint several obstacles.")

    a, b = st.columns(2)
    with a:
        st.radio("Click action", TOOLS, key="m4_tool", horizontal=True)
        s1, s2 = st.columns(2)
        s1.slider("Rows", 6, 24, key="m4_rows", on_change=_resize)
        s2.slider("Columns", 6, 24, key="m4_cols", on_change=_resize)
    with b:
        st.slider("Beam width k", 1, 20, key="m4_k", on_change=_on_k)
        d1, d2 = st.columns(2)
        d1.slider("Random obstacle density (%)", 5, 45, key="m4_density")
        d2.number_input("Seed", 0, 9999, key="m4_seed", step=1)

    bt = st.columns(4)
    with bt[0]:
        button("🎲 Random maze", key="m4_btn_random", on_click=_random)
    with bt[1]:
        button("🧹 Clear maze", key="m4_btn_clear", on_click=_clear)
    with bt[2]:
        button("↺ Example maze", key="m4_btn_restore", on_click=_restore)
    with bt[3]:
        button("▶ Execute beam search", key="m4_btn_exec", on_click=_execute, type="primary")
    rn = st.columns(4)
    trace = ss.m4_trace
    with rn[0]:
        button("🎬 Execute & animate", key="m4_btn_anim", on_click=_animate, disabled=ss.m4_playing)
    with rn[1]:
        button("⏸ Pause", key="m4_btn_pause", on_click=_pause, disabled=not ss.m4_playing)
    with rn[2]:
        button("◀ Step back", key="m4_btn_back", on_click=_back, disabled=trace is None or ss.m4_step == 0)
    with rn[3]:
        button("Step ▶", key="m4_btn_fwd", on_click=_forward,
               disabled=trace is None or ss.m4_playing or ss.m4_step >= len(trace.steps) - 1)
    st.slider("Animation speed (seconds per level)", 0.1, 1.5, step=0.1, key="m4_speed")

    errs = g.validate()
    if errs:
        st.error(" ".join(errs))
        return

    # ---- chart ---------------------------------------------------------------
    if trace is not None:
        step = trace.steps[ss.m4_step]
        fig = maze_figure(g, step.explored, step.beam, step.pruned_all, step.path, height=500)
    else:
        step = None
        fig = maze_figure(g, height=500)
    key = f"m4_chart_{ss.m4_ver}"
    try:
        event = show_plot(fig, key=key, on_select="rerun", selection_mode=("points", "box"))
    except TypeError:                                  # very old Streamlit: no selection events
        show_plot(fig, key=key)
        event = None
    legend(maze_legend())

    points = []
    if event is not None:
        try:
            points = list(event["selection"]["points"])
        except Exception:                              # no selection yet
            points = []
    if points:
        if _apply_clicks(points):
            _invalidate()
        else:
            ss.m4_ver += 1                              # consume the selection even if nothing changed
        st.rerun()

    # ---- results ---------------------------------------------------------------
    if trace is None:
        callout("info", f"Start {g.start}, goal {g.goal}, {len(g.obstacles)} obstacles. "
                        "Press <b>Execute beam search</b> for the result or <b>Execute & animate</b> to watch it level by level.")
    else:
        stats = step.stats
        at_end = ss.m4_step == len(trace.steps) - 1
        stat_cards([
            ("Level", ss.m4_step, f"of {len(trace.steps) - 1}"),
            ("Expanded", stats["expanded"], "cells expanded"),
            ("Generated", stats["generated"], "candidates"),
            ("Pruned", stats["pruned"], "cells discarded"),
            ("Path length", trace.path_length if (trace.found and at_end) else "-", "moves"),
            ("Run time", f"{ss.m4_time:.0f} µs", "median of 15 runs"),
        ], ["", "", "", "warn", "good" if trace.found else "", ""])
        st.caption(step.message)
        if at_end:
            _result_message(trace, ss.m4_bfs)

    with st.expander("Edit by coordinates (alternative to clicking)"):
        ss.m4_er = min(int(ss.m4_er), g.rows - 1)
        ss.m4_ec = min(int(ss.m4_ec), g.cols - 1)
        e1, e2 = st.columns(2)
        e1.number_input("Row", 0, g.rows - 1, key="m4_er", step=1)
        e2.number_input("Column", 0, g.cols - 1, key="m4_ec", step=1)
        f1, f2, f3 = st.columns(3)
        with f1:
            button("Set start here", key="m4_xy_start", on_click=_edit_xy, args=("start",))
        with f2:
            button("Set goal here", key="m4_xy_goal", on_click=_edit_xy, args=("goal",))
        with f3:
            button("Toggle obstacle here", key="m4_xy_obs", on_click=_edit_xy, args=("obstacle",))

    # ---- animation loop ----------------------------------------------------------
    if ss.m4_playing and trace is not None:
        if ss.m4_step >= len(trace.steps) - 1:
            ss.m4_playing = False
            st.rerun()
        time.sleep(ss.m4_speed)
        ss.m4_step += 1
        st.rerun()

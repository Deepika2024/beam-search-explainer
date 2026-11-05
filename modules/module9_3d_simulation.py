"""Module 9 - 3D four-directional Beam Search simulation.

This is an additive module. The original 2D Maze Solver and all existing modules
remain unchanged. The same tested Grid and beam_search_grid implementation drive
this 3D visualization.
"""
from __future__ import annotations

import time

import streamlit as st

from algorithms.beam_search import beam_search_grid
from algorithms.bfs import bfs_grid
from algorithms.grid import Grid, random_maze
from components.ui import button, callout, legend, show_plot, stat_cards
from visualizations.maze_3d_visualizer import maze_3d_figure

ss = st.session_state


def _default_grid() -> Grid:
    rows, cols = 10, 12
    obstacles = {
        (1, 3), (2, 3), (3, 3), (5, 2), (5, 3), (5, 4),
        (2, 7), (3, 7), (4, 7), (6, 7), (7, 7),
        (7, 2), (7, 3), (7, 4), (2, 10), (3, 10),
    }
    return Grid(rows, cols, (0, 0), (9, 11), obstacles)


def _init() -> None:
    ss.setdefault("m9_grid", _default_grid())
    defaults = {
        "m9_k": 3, "m9_density": 22, "m9_seed": 17, "m9_trace": None,
        "m9_step": 0, "m9_ver": 0, "m9_speed": 0.4,
        "m9_edit_row": 4, "m9_edit_col": 5, "m9_edit_action": "Toggle obstacle",
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)


def _invalidate() -> None:
    ss.m9_trace = None
    ss.m9_step = 0
    ss.m9_ver += 1


def _run() -> None:
    grid: Grid = ss.m9_grid
    errors = grid.validate()
    if errors:
        return
    ss.m9_trace = beam_search_grid(grid, int(ss.m9_k))
    ss.m9_bfs = bfs_grid(grid)
    ss.m9_step = 0


def _randomize() -> None:
    g: Grid = ss.m9_grid
    ss.m9_grid = random_maze(g.rows, g.cols, ss.m9_density / 100.0, int(ss.m9_seed), g.start, g.goal)
    _invalidate()


def _clear() -> None:
    ss.m9_grid.obstacles.clear()
    _invalidate()


def _restore() -> None:
    ss.m9_grid = _default_grid()
    _invalidate()


def _edit() -> None:
    g: Grid = ss.m9_grid
    cell = (int(ss.m9_edit_row), int(ss.m9_edit_col))
    action = ss.m9_edit_action
    if action == "Toggle obstacle":
        g.toggle_obstacle(cell)
    elif action == "Move start":
        g.set_start(cell)
    else:
        g.set_goal(cell)
    _invalidate()


def _step_back() -> None:
    ss.m9_step = max(0, ss.m9_step - 1)


def _step_forward() -> None:
    if ss.m9_trace is not None:
        ss.m9_step = min(len(ss.m9_trace.steps) - 1, ss.m9_step + 1)


def _result_note(result, bfs) -> None:
    if result.found:
        shortest = int(bfs.cost) if bfs.found else None
        if shortest is not None and result.path_length == shortest:
            callout("ok", f"<b>Goal reached.</b> Beam Search found a {result.path_length}-move route. In this run it matches the BFS shortest path.")
        elif shortest is not None:
            callout("warn", f"<b>Goal reached.</b> Beam Search found a {result.path_length}-move route, while BFS found a {shortest}-move shortest route. Beam Search is not guaranteed to be optimal.")
        else:
            callout("ok", f"<b>Goal reached</b> in {result.path_length} moves.")
    elif bfs.found:
        callout("bad", f"<b>Beam Search did not reach the goal with k={result.k}.</b> BFS confirms a route of {int(bfs.cost)} moves exists. A promising cell was likely pruned; try a wider beam.")
    else:
        callout("warn", "<b>No route exists in this maze.</b> BFS also confirms the goal is unreachable because of the obstacle layout.")


def render() -> None:
    _init()
    g: Grid = ss.m9_grid
    st.markdown("## 3D Four-Direction Beam Search Simulator")
    st.caption(
        "Explore the same real Beam Search algorithm in a rotatable 3D maze. "
        "Movement is restricted to four directions on the floor grid: North/Up, South/Down, East/Right and West/Left. "
        "Raised blocks represent obstacles; the 3D height is a visual layer, not an extra movement dimension."
    )

    st.markdown("### Environment controls")
    left, right = st.columns([1.15, 1])
    with left:
        st.slider("Beam width (k)", 1, 15, key="m9_k")
        st.slider("Random obstacle density (%)", 5, 45, key="m9_density")
        st.number_input("Random seed", min_value=0, max_value=9999, key="m9_seed")
    with right:
        st.radio("Edit cell", ["Toggle obstacle", "Move start", "Move goal"], key="m9_edit_action", horizontal=True)
        r1, r2 = st.columns(2)
        r1.number_input("Row", min_value=0, max_value=g.rows - 1, key="m9_edit_row")
        r2.number_input("Column", min_value=0, max_value=g.cols - 1, key="m9_edit_col")
        button("Apply cell edit", key="m9_apply_edit", on_click=_edit)

    controls = st.columns(4)
    with controls[0]:
        button("▶ Run Beam Search", key="m9_run", on_click=_run, type="primary")
    with controls[1]:
        button("🎲 Random maze", key="m9_random", on_click=_randomize)
    with controls[2]:
        button("🧹 Clear obstacles", key="m9_clear", on_click=_clear)
    with controls[3]:
        button("↺ Restore example", key="m9_restore", on_click=_restore)

    errors = g.validate()
    if errors:
        st.error(" ".join(errors))
        return

    trace = ss.m9_trace
    if trace is None:
        explored, beam, pruned, path = set(), {g.start}, set(), None
        step = None
    else:
        step = trace.steps[ss.m9_step]
        explored, beam, pruned, path = step.explored, step.beam, step.pruned_all, step.path

    fig = maze_3d_figure(g, explored=explored, beam=beam, pruned=pruned, path=path)
    show_plot(fig, key=f"m9_3d_{ss.m9_ver}")
    legend([
        ("#22c55e", "Start"), ("#f43f5e", "Goal"), ("#64748b", "Obstacle blocks"),
        ("#38bdf8", "Explored"), ("#a78bfa", "Current beam"),
        ("#f59e0b", "Pruned"), ("#facc15", "Final path"),
    ])

    st.markdown("### Four-direction movement")
    direction_cols = st.columns(4)
    for col, label, detail in zip(direction_cols,
                                  ["⬆ North / Up", "⬇ South / Down", "➡ East / Right", "⬅ West / Left"],
                                  ["row − 1", "row + 1", "column + 1", "column − 1"]):
        with col:
            st.markdown(f"**{label}**")
            st.caption(detail)

    if trace is None:
        callout("info", f"Start: <b>{g.start}</b> · Goal: <b>{g.goal}</b> · Obstacles: <b>{len(g.obstacles)}</b>. Rotate the 3D scene with your mouse, zoom, and pan. Press Run Beam Search to see the real algorithm's decisions.")
        return

    st.markdown("### Step-by-step execution")
    step_cols = st.columns([1, 1, 2])
    with step_cols[0]:
        button("◀ Previous", key="m9_prev", on_click=_step_back, disabled=ss.m9_step == 0)
    with step_cols[1]:
        button("Next ▶", key="m9_next", on_click=_step_forward, disabled=ss.m9_step >= len(trace.steps) - 1)
    with step_cols[2]:
        st.slider("Search level", 0, len(trace.steps) - 1, key="m9_step")

    # Slider changes trigger rerun; redraw the exact recorded snapshot.
    step = trace.steps[ss.m9_step]
    stats = step.stats
    stat_cards([
        ("Level", ss.m9_step, f"of {len(trace.steps) - 1}"),
        ("Expanded", stats["expanded"], "cells expanded"),
        ("Generated", stats["generated"], "candidate cells"),
        ("Pruned", stats["pruned"], "discarded cells"),
        ("Beam size", stats["beam_size"], f"maximum {trace.k}"),
    ], ["", "", "", "warn", "good"])
    st.info(step.message)
    if ss.m9_step == len(trace.steps) - 1:
        _result_note(trace, ss.m9_bfs)
        if trace.found:
            st.success(f"Final route: {trace.path} · Path length: {trace.path_length} moves")
        else:
            st.caption(trace.failure_reason)

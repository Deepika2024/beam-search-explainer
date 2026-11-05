"""Module 2 - step-by-step search-tree explorer.

Session-state keys (all prefixed ``m2_``):
    scenario, k, over (heuristic overrides), tree, result, time_us,
    step, started, playing, speed, predict, unseen, pred, pred_hits, pred_total, editor_v
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from algorithms.beam_search import CHECK, EXPAND, GENERATE, SELECT, beam_search
from algorithms.common import benchmark
from components.explanations import pseudocode_html, step_explanation_html
from components.metrics import beam_stats_cards
from components.ui import button, callout, html_block, legend, show_plot, stat_cards
from data.sample_search_trees import TREE_SCENARIOS
from visualizations.search_tree import draw_tree, legend_entries, nodes_from_status, nodes_from_step

KS = [1, 2, 3, 5, 10]
ss = st.session_state


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------
def _build() -> None:
    """(Re)run the real algorithm for the current scenario, heuristics and k, and rewind the replay."""
    tree = TREE_SCENARIOS[ss.m2_scenario]()
    tree.set_heuristics(ss.m2_over)
    k = ss.m2_k
    ss.m2_tree = tree
    ss.m2_result = beam_search(tree, k)
    ss.m2_time = benchmark(lambda: beam_search(tree, k, record=False), 15)
    ss.m2_step, ss.m2_started, ss.m2_playing, ss.m2_pred = 0, False, False, {}
    for key in [k_ for k_ in ss if k_.startswith("m2_sel_")]:
        del ss[key]


def _on_scenario() -> None:
    ss.m2_over = {}
    ss.m2_editor_v += 1
    ss.m2_pred_hits = ss.m2_pred_total = 0
    _build()


def _reset() -> None:
    ss.m2_pred_hits = ss.m2_pred_total = 0
    _build()


def _init() -> None:
    defaults = {"m2_scenario": next(iter(TREE_SCENARIOS)), "m2_k": 2, "m2_over": {}, "m2_speed": 1.0,
                "m2_predict": False, "m2_unseen": True, "m2_pred_hits": 0, "m2_pred_total": 0, "m2_editor_v": 0}
    for key, val in defaults.items():
        ss.setdefault(key, val)
    if "m2_result" not in ss:
        _build()


def _last() -> int:
    return len(ss.m2_result.steps) - 1


def _level(step) -> int:
    """Depth of the beam that is being expanded in this level (SELECT/CHECK already moved one deeper)."""
    return step.depth if step.phase in (EXPAND, GENERATE) else step.depth - 1


def _needs_prediction() -> bool:
    step = ss.m2_result.steps[ss.m2_step]
    return (ss.m2_predict and ss.m2_started and step.phase == GENERATE
            and _level(step) not in ss.m2_pred and len(step.candidates) > ss.m2_k)


# ---- button callbacks ------------------------------------------------------
def _start() -> None:
    ss.m2_started, ss.m2_step, ss.m2_playing = True, 0, False


def _pause() -> None:
    ss.m2_playing = False


def _resume() -> None:
    ss.m2_playing = True


def _next() -> None:
    if not ss.m2_started:
        ss.m2_started, ss.m2_step = True, 0
    elif ss.m2_step < _last() and not _needs_prediction():
        ss.m2_step += 1


def _prev() -> None:
    ss.m2_playing = False
    if ss.m2_step > 0:
        ss.m2_step -= 1


def _run_auto() -> None:
    if not ss.m2_started or ss.m2_step >= _last():
        ss.m2_step = 0
    ss.m2_started, ss.m2_playing = True, True


def _opt_label(tree, nid: int) -> str:
    return f"{tree.label(nid)} (h={tree.nodes[nid].h:g})"


def _lock_prediction() -> None:
    """Store the user's guess for the current level; it is revealed at the SELECT step."""
    tree, step = ss.m2_tree, ss.m2_result.steps[ss.m2_step]
    lvl = _level(step)
    lookup = {_opt_label(tree, c): c for c in step.candidates}
    guess = [lookup[o] for o in ss.get(f"m2_sel_{lvl}", []) if o in lookup]
    actual = ss.m2_result.steps[ss.m2_step + 1].selected          # the real SELECT result (not shown yet)
    hits = len(set(guess) & set(actual))
    ss.m2_pred[lvl] = {"guess": guess, "actual": list(actual), "hits": hits}
    ss.m2_pred_hits += hits
    ss.m2_pred_total += len(actual)


def _skip_prediction() -> None:
    step = ss.m2_result.steps[ss.m2_step]
    ss.m2_pred[_level(step)] = {"guess": None, "actual": [], "hits": 0}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _controls() -> None:
    started, playing = ss.m2_started, ss.m2_playing
    at_end = ss.m2_step >= _last()
    blocked = _needs_prediction()
    r1 = st.columns(4)
    with r1[0]:
        button("▶ Start", key="m2_btn_start", on_click=_start, disabled=started, type="primary")
    with r1[1]:
        button("⚡ Run automatically", key="m2_btn_auto", on_click=_run_auto, disabled=playing)
    with r1[2]:
        button("⏸ Pause", key="m2_btn_pause", on_click=_pause, disabled=not playing)
    with r1[3]:
        button("⏵ Resume", key="m2_btn_resume", on_click=_resume,
               disabled=playing or not started or at_end)
    r2 = st.columns(3)
    with r2[0]:
        button("◀ Previous step", key="m2_btn_prev", on_click=_prev, disabled=(not started) or ss.m2_step == 0)
    with r2[1]:
        button("Next step ▶", key="m2_btn_next", on_click=_next,
               disabled=playing or (started and (at_end or blocked)))
    with r2[2]:
        button("↺ Reset", key="m2_btn_reset", on_click=_reset)


def _prediction_ui(step) -> None:
    tree = ss.m2_tree
    lvl = _level(step)
    n_keep = min(ss.m2_k, len(step.candidates))
    options = [_opt_label(tree, c) for c in step.candidates]
    html_block('<div class="card"><div class="card-title">Learn by prediction</div>'
               f'<p>The candidates are on screen with their scores. Which <b>{n_keep}</b> will survive '
               f'(lower h wins)? Lock your guess before the beam is revealed.</p></div>')
    st.multiselect(f"Your top-{n_keep} (pick up to {n_keep})", options, key=f"m2_sel_{lvl}", max_selections=n_keep)
    c1, c2 = st.columns(2)
    with c1:
        button("🔒 Lock prediction", key="m2_btn_lock", on_click=_lock_prediction, type="primary")
    with c2:
        button("Skip", key="m2_btn_skip", on_click=_skip_prediction)


def _prediction_feedback(step) -> None:
    pred = ss.m2_pred.get(_level(step))
    if not pred or pred["guess"] is None or step.phase not in (SELECT, CHECK):
        return
    tree = ss.m2_tree
    names = lambda ids: ", ".join(tree.label(i) for i in ids) or "nothing"
    kind = "ok" if pred["hits"] == len(pred["actual"]) and set(pred["guess"]) == set(pred["actual"]) else "warn"
    callout(kind, f"<b>Your prediction:</b> {names(pred['guess'])}<br><b>Actual beam:</b> {names(pred['actual'])}<br>"
                  f"{pred['hits']} of {len(pred['actual'])} correct.")


def _experiment_mode() -> None:
    tree = ss.m2_tree
    with st.expander("🧪 Experiment mode: change heuristic values and watch the search change"):
        st.caption("Edit the h column, press Apply, then step through the run again. Try making the goal's "
                   "ancestors look worse than a dead end, then widen the beam.")
        ids = list(tree.nodes)
        df = pd.DataFrame({"Node": [tree.label(i) for i in ids], "Depth": [tree.nodes[i].depth for i in ids],
                           "Goal": [tree.nodes[i].is_goal for i in ids], "h": [tree.nodes[i].h for i in ids]})
        edited = st.data_editor(df, key=f"m2_editor_{ss.m2_editor_v}", hide_index=True, disabled=["Node", "Depth", "Goal"],
                                column_config={"h": st.column_config.NumberColumn("h", min_value=0.0, max_value=99.0, step=0.5)})
        c1, c2 = st.columns(2)
        with c1:
            apply = button("Apply heuristic changes", key="m2_btn_apply", type="primary")
        with c2:
            restore = button("Restore original values", key="m2_btn_restore")
        if apply:
            if edited["h"].isna().any():
                st.error("Every node needs a numeric h value.")
            else:
                base = TREE_SCENARIOS[ss.m2_scenario]()
                ss.m2_over = {i: float(h) for i, h in zip(ids, edited["h"]) if float(h) != base.nodes[i].h}
                _build()
                st.rerun()
        if restore:
            ss.m2_over = {}
            ss.m2_editor_v += 1
            _build()
            st.rerun()


def render() -> None:
    _init()
    tree, result = ss.m2_tree, ss.m2_result
    st.markdown("## Interactive Search Tree Explorer")
    st.caption(f"{tree.description}  ·  Preferred score: **lower h is better** (h = estimated cost to goal).")

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        st.selectbox("Tree scenario", list(TREE_SCENARIOS), key="m2_scenario", on_change=_on_scenario)
    with c2:
        st.select_slider("Beam width k", KS, key="m2_k", on_change=_build)
    with c3:
        st.slider("Auto-run speed (seconds per step)", 0.3, 3.0, step=0.1, key="m2_speed")
    t1, t2 = st.columns(2)
    with t1:
        st.toggle("Learn by prediction", key="m2_predict",
                  help="Before each selection, guess which candidates survive.")
    with t2:
        st.toggle("Show the not-yet-generated part of the tree", key="m2_unseen")

    _controls()

    step = result.steps[ss.m2_step]
    left, right = st.columns([3, 2])
    with left:
        if ss.m2_started:
            nodes = nodes_from_step(tree, step)
            beam_names = ", ".join(tree.label(n) for n in step.beam) or "empty"
            title = (f"Step {ss.m2_step + 1}/{len(result.steps)}  ·  depth {step.depth}  ·  "
                     f"active beam (k={ss.m2_k}): {beam_names}")
        else:
            nodes = nodes_from_status(tree, {})
            title = "Press ▶ Start, ⚡ Run automatically or Next step to begin."
        show_plot(draw_tree(nodes, show_unseen=ss.m2_unseen or not ss.m2_started, height=500, title=title), key="m2_fig")
        legend(legend_entries())
        st.progress((ss.m2_step + 1) / len(result.steps) if ss.m2_started else 0.0)
    with right:
        if ss.m2_started:
            beam_stats_cards(step.stats, time_us=ss.m2_time)
            if ss.m2_predict and ss.m2_pred_total:
                stat_cards([("Prediction score", f"{ss.m2_pred_hits}/{ss.m2_pred_total}", "correct survivors")], ["accent"])
            if _needs_prediction():
                _prediction_ui(step)
            html_block(step_explanation_html(step, tree, ss.m2_k))
            _prediction_feedback(step)
            with st.expander("Pseudocode (executing lines highlighted)", expanded=True):
                html_block(pseudocode_html(step.lines))
        else:
            callout("info", "Each step explains: <b>which nodes were expanded</b>, <b>which candidates were generated</b>, "
                            "<b>their scores</b>, <b>which top-k were kept</b> and <b>which were discarded and why</b>.")
            with st.expander("Pseudocode", expanded=True):
                html_block(pseudocode_html([]))

    _experiment_mode()

    # ---- automatic playback: advance one step, then rerun the script ----------
    if ss.m2_playing:
        if ss.m2_step >= _last() or _needs_prediction():
            ss.m2_playing = False
            st.rerun()
        time.sleep(ss.m2_speed)
        ss.m2_step += 1
        st.rerun()

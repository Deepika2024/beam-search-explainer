"""Module 3 - beam-width playground with a live comparison dashboard (state keys prefixed ``m3_``)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from algorithms.beam_search import SELECT, beam_search
from algorithms.common import benchmark, fmt_time
from components.explanations import BEAM_WIDTH_LARGE, BEAM_WIDTH_SMALL
from components.metrics import beam_stats_cards
from components.ui import button, callout, html_block, legend, show_df, show_plot
from data.sample_search_trees import TREE_SCENARIOS, random_tree
from visualizations.charts import bars
from visualizations.search_tree import draw_tree, legend_entries, nodes_from_step

KS = [1, 2, 3, 5, 10]
RANDOM = "Random tree (you choose the settings)"
FAIL = "Failure demo: narrow beam loses the goal"
TRAP = "Cost trap: wider beam is not better"
SCENARIOS = list(TREE_SCENARIOS) + [RANDOM]
ss = st.session_state


def _set(scenario: str, k: int) -> None:
    ss.m3_scenario, ss.m3_k = scenario, k


def _tree():
    if ss.m3_scenario == RANDOM:
        return random_tree(ss.m3_b, ss.m3_d, int(ss.m3_seed), ss.m3_noise)
    return TREE_SCENARIOS[ss.m3_scenario]()


def _insights(tree, runs) -> None:
    """Statements are derived from the runs on screen - nothing here is canned."""
    found = [k for k in KS if runs[k].found]
    missed = [k for k in KS if not runs[k].found]
    opt = tree.optimal_cost()
    if not tree.goals():
        callout("bad", "This tree contains no goal node, so no beam width can succeed.")
        return
    if missed and found:
        callout("warn", f"<b>A narrow beam misses a goal that exists.</b> k = {', '.join(map(str, missed))} "
                        f"failed, the first width that reaches the goal is <b>k = {min(found)}</b>.")
    elif missed and not found:
        callout("bad", "<b>No tested width found the goal</b>, although the tree contains one: the heuristic "
                       "ranks the goal's ancestors too low for any of these beams. Try a bigger k or edit h.")
    else:
        callout("ok", "Every tested beam width found the goal on this tree. Switch to the failure demo to see a miss.")
    if opt is None or not found:
        return
    costs = {k: runs[k].cost for k in found}
    worse_pairs = [(a, b) for a in found for b in found if a < b and costs[b] > costs[a]]
    if worse_pairs:
        a, b = worse_pairs[0]
        callout("warn", f"<b>A wider beam is not automatically better.</b> k = {a} found a path of cost {costs[a]:g}, "
                        f"but k = {b} found cost {costs[b]:g}. The optimal cost on this tree is {opt:g}.")
    elif any(costs[k] > opt for k in found):
        bad = [k for k in found if costs[k] > opt]
        callout("warn", f"<b>Finding the goal is not the same as finding the best path.</b> k = {', '.join(map(str, bad))} "
                        f"reached cost {costs[bad[0]]:g} but the optimal cost is {opt:g}.")
    else:
        callout("info", f"All successful widths reached the optimal cost ({opt:g}) here. That is luck of this tree, "
                        "not a guarantee: the cost-trap scenario shows a wider beam returning a worse path.")


def render() -> None:
    for key, val in {"m3_scenario": FAIL, "m3_k": 1, "m3_b": 3, "m3_d": 3, "m3_seed": 7, "m3_noise": 1.5}.items():
        ss.setdefault(key, val)

    st.markdown("## Beam Width Playground")
    st.caption("Change k and the whole search is re-run: tree, selected and pruned nodes, final path and statistics.")

    st.markdown("**Guided experiments**")
    g1, g2 = st.columns(2)
    with g1:
        button("① Narrow beam misses the goal (start at k=1)", key="m3_exp_fail", on_click=_set, args=(FAIL, 1))
    with g2:
        button("② Wider beam is not optimal (start at k=2)", key="m3_exp_trap", on_click=_set, args=(TRAP, 2))

    c1, c2 = st.columns([3, 2])
    with c1:
        st.selectbox("Tree scenario", SCENARIOS, key="m3_scenario")
    with c2:
        st.select_slider("Beam width k", KS, key="m3_k")
    if ss.m3_scenario == RANDOM:
        with st.expander("Random tree settings", expanded=True):
            r1, r2, r3, r4 = st.columns(4)
            r1.slider("Branching b", 2, 4, key="m3_b")
            r2.slider("Depth d", 2, 4, key="m3_d")
            r3.number_input("Seed", 0, 9999, key="m3_seed", step=1)
            r4.slider("Heuristic noise", 0.0, 4.0, step=0.5, key="m3_noise",
                      help="0 = perfect heuristic. Larger = more misleading.")

    try:
        tree = _tree()
    except ValueError as exc:
        st.error(str(exc))
        return
    k = ss.m3_k
    st.caption(tree.description)

    # ---- run the selected k with a full trace, and all k without trace for the dashboard
    result = beam_search(tree, k)
    runs = {kk: beam_search(tree, kk, record=False) for kk in KS}
    times = {kk: benchmark(lambda kk=kk: beam_search(tree, kk, record=False), 15) for kk in KS}

    left, right = st.columns([3, 2])
    with left:
        final = result.steps[-1]
        title = f"Final state for k = {k}" + ("  ·  goal found" if result.found else "  ·  goal NOT found")
        show_plot(draw_tree(nodes_from_step(tree, final), height=470, title=title), key="m3_fig")
        legend(legend_entries())
    with right:
        st.markdown(f"#### Result for k = {k}")
        if result.found:
            callout("ok", "<b>Goal found.</b> Path: " + " → ".join(tree.label(n) for n in result.path)
                          + f"<br>Path cost: <b>{result.cost:g}</b>")
        else:
            callout("bad", "<b>Goal not found.</b> " + result.failure_reason)
        beam_stats_cards(final.stats, time_us=times[k])
        rows = [{"Depth": s.depth, "Kept in beam": ", ".join(tree.label(n) for n in s.selected),
                 "Pruned": ", ".join(tree.label(n) for n in s.pruned) or "-"}
                for s in result.steps if s.phase == SELECT]
        if rows:
            show_df(pd.DataFrame(rows))

    # ---- comparison dashboard --------------------------------------------------
    st.markdown("### Comparison dashboard: same tree, different beam widths")
    opt = tree.optimal_cost()
    table = []
    for kk in KS:
        r = runs[kk]
        verdict = "-"
        if r.found and opt is not None:
            verdict = "optimal" if r.cost == opt else f"+{r.cost - opt:g} above optimal"
        table.append({"k": kk, "Goal found": "✅ yes" if r.found else "❌ no", "Expanded": int(r.stats["expanded"]),
                      "Generated": int(r.stats["generated"]), "Pruned": int(r.stats["pruned"]),
                      "Peak retained": int(r.stats["max_beam"]), "Time": fmt_time(times[kk]),
                      "Path cost": f"{r.cost:g}" if r.found else "-", "Quality": verdict})
    show_df(pd.DataFrame(table))
    if opt is not None:
        st.caption(f"Optimal path cost on this tree (exhaustive check over all goals): {opt:g}.")

    a, b = st.columns(2)
    with a:
        show_plot(bars(KS, [int(runs[x].stats["expanded"]) for x in KS], "Explored (expanded) nodes",
                       highlight_x=k), key="m3_c1")
    with b:
        show_plot(bars(KS, [int(runs[x].stats["max_beam"]) for x in KS], "Retained nodes (peak beam size)",
                       color="#3b82f6", highlight_x=k), key="m3_c2")
    c, d = st.columns(2)
    with c:
        show_plot(bars(KS, [round(times[x], 1) for x in KS], "Search time (µs, median)", color="#f59e0b",
                       highlight_x=k), key="m3_c3")
    with d:
        show_plot(bars(KS, [runs[x].cost if runs[x].found else 0 for x in KS],
                       "Path cost (0 = goal not found)", color="#22c55e", highlight_x=k), key="m3_c4")

    st.markdown("### What the experiment shows")
    _insights(tree, runs)

    s1, s2 = st.columns(2)
    with s1:
        html_block('<div class="card"><div class="card-title">Small beam width</div><ul>'
                   + "".join(f"<li>{t}</li>" for t in BEAM_WIDTH_SMALL) + "</ul></div>")
    with s2:
        html_block('<div class="card"><div class="card-title">Large beam width</div><ul>'
                   + "".join(f"<li>{t.replace('**', '')}</li>" for t in BEAM_WIDTH_LARGE) + "</ul></div>")
    callout("info", "<b>Important:</b> increasing the beam width does <b>not</b> guarantee an optimal result - and it "
                    "does not even guarantee a better one than a smaller beam. Run the cost-trap experiment (②) to see "
                    "k = 1 beat every wider beam.")

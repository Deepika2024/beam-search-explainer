"""Module 8 - quiz and challenge arena (state keys prefixed ``m8_``)."""
from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd
import streamlit as st

from components.ui import button, callout, html_block, show_df, stat_cards
from data.quiz_bank import Question, build_questions

ss = st.session_state


def _new_quiz() -> None:
    seed = random.randint(1, 10_000)
    ss.m8_seed = seed
    ss.m8_questions = build_questions(seed)
    ss.m8_idx = 0
    ss.m8_results = {}          # index -> (chosen options, correct?)
    ss.m8_warn = False


def _widget_key() -> str:
    return f"m8_in_{ss.m8_seed}_{ss.m8_idx}"


def _submit() -> None:
    q: Question = ss.m8_questions[ss.m8_idx]
    chosen = ss.get(_widget_key())
    chosen = [] if chosen is None else (list(chosen) if q.multi else [chosen])
    if not chosen:
        ss.m8_warn = True
        return
    ss.m8_warn = False
    ss.m8_results[ss.m8_idx] = (chosen, set(chosen) == set(q.answer))


def _next() -> None:
    ss.m8_idx += 1
    ss.m8_warn = False


def _summary() -> None:
    qs, res = ss.m8_questions, ss.m8_results
    score = sum(1 for _, ok in res.values() if ok)
    pct = round(100 * score / len(qs))
    st.markdown("### Final performance")
    stat_cards([("Score", f"{score}/{len(qs)}", "correct answers"), ("Percentage", f"{pct}%", ""),
                ("Questions", len(qs), "answered")], ["good" if pct >= 70 else "warn", "accent", ""])
    by_kind = defaultdict(lambda: [0, 0])
    for i, q in enumerate(qs):
        by_kind[q.kind][1] += 1
        by_kind[q.kind][0] += int(res[i][1])
    show_df(pd.DataFrame([{"Question type": k, "Correct": v[0], "Total": v[1],
                           "Accuracy": f"{round(100 * v[0] / v[1])}%"} for k, v in by_kind.items()]))
    weak = [k for k, v in by_kind.items() if v[0] < v[1]]
    if pct == 100:
        callout("ok", "Perfect score: you can predict what beam search keeps, what it prunes, and why.")
    elif weak:
        callout("info", "Review these question types: <b>" + ", ".join(weak) + "</b>. "
                        "Modules 2 and 3 let you test each idea hands-on.")
    with st.expander("Review every question"):
        for i, q in enumerate(qs):
            chosen, ok = res[i]
            st.markdown(f"**{i + 1}. {q.kind}** - {'✅' if ok else '❌'}  \n{q.prompt}")
            st.markdown(f"Your answer: `{', '.join(chosen)}` · Correct: `{', '.join(q.answer)}`")
            st.caption(q.explanation)
    button("↺ New quiz (fresh live questions)", key="m8_retake", on_click=_new_quiz, type="primary")


def render() -> None:
    if "m8_questions" not in ss:
        _new_quiz()
    st.markdown("## Quiz & Challenge Arena")
    st.caption("Questions marked *Predict*, *Identify* and *experiment* are computed by running the real algorithms "
               "on fresh trees, so you cannot memorise the answers.")
    qs, idx = ss.m8_questions, ss.m8_idx
    answered = sum(1 for _, ok in ss.m8_results.values() if ok)
    if idx >= len(qs):
        st.progress(1.0)
        _summary()
        return

    q: Question = qs[idx]
    st.progress(idx / len(qs), text=f"Question {idx + 1} of {len(qs)}  ·  score so far {answered}/{len(ss.m8_results)}")
    html_block(f'<div class="card"><div class="card-title">{q.kind}</div></div>')
    st.markdown(q.prompt)
    if q.table is not None:
        show_df(q.table)

    done = idx in ss.m8_results
    if q.multi:
        st.multiselect("Select all that apply", q.options, key=_widget_key(), disabled=done)
    else:
        st.radio("Choose one", q.options, index=None, key=_widget_key(), disabled=done)

    if not done:
        if ss.m8_warn:
            st.warning("Pick an answer first.")
        button("Submit answer", key="m8_submit", on_click=_submit, type="primary")
    else:
        chosen, ok = ss.m8_results[idx]
        callout("ok" if ok else "bad",
                f"<b>{'Correct.' if ok else 'Not quite.'}</b> Correct answer: <b>{', '.join(q.answer)}</b>.<br>{q.explanation}")
        label = "Next question ▶" if idx + 1 < len(qs) else "See final results ▶"
        button(label, key="m8_next", on_click=_next, type="primary")

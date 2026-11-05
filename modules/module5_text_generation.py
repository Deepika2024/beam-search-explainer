"""Module 5 - beam-search text decoding on a toy bigram model (state keys prefixed ``m5_``)."""
from __future__ import annotations

import math
import time
from typing import List

import pandas as pd
import streamlit as st

from algorithms.beam_search import BEAM, EXPLORED, PATH, PRUNED, UNSEEN
from algorithms.text_generation import (EOS, PROMPTS, TextResult, ToyLanguageModel, beam_search_text, exhaustive_best,
                                        norm_score)
from components.ui import button, callout, html_block, legend, show_df, show_plot, stat_cards
from visualizations.charts import score_bars
from visualizations.search_tree import NodeView, draw_tree, legend_entries

ss = st.session_state
KS = [1, 2, 3, 5, 10]
MODEL = ToyLanguageModel()


def _show(tokens) -> str:
    return " ".join("⟨eos⟩" if t == EOS else t for t in tokens) or "(prompt)"


def _reset_step() -> None:
    ss.m5_step, ss.m5_playing = 0, False


def _next(last: int) -> None:
    ss.m5_step = min(last, ss.m5_step + 1)


def _prev() -> None:
    ss.m5_playing = False
    ss.m5_step = max(0, ss.m5_step - 1)


def _play(last: int) -> None:
    if ss.m5_step >= last:
        ss.m5_step = 0
    ss.m5_playing = True


def _pause() -> None:
    ss.m5_playing = False


def _tree_nodes(res: TextResult, s: int) -> List[NodeView]:
    """Hypothesis tree as NodeViews at step ``s`` (hypotheses born later are 'unseen' and hidden)."""
    status = res.steps[s].status
    on_best = set()
    if s == len(res.steps) - 1:                                   # highlight the winning sentence at the end
        cur = res.best.id
        while cur is not None:
            on_best.add(cur)
            cur = res.hyps[cur].parent
    out = []
    for hid, h in res.hyps.items():
        if h.born > s:
            st_ = UNSEEN
        elif hid in on_best:
            st_ = PATH
        else:
            st_ = {"beam": BEAM, "pruned": PRUNED, "explored": EXPLORED}.get(status.get(hid, "explored"), EXPLORED)
        label = res.prompt.split()[-1] if hid == 0 else h.word.replace(EOS, "⟨eos⟩")
        hover = (f"<b>{_show(h.tokens)}</b><br>P(word|prev) = {h.p:.2f}<br>log P = {h.logp:.3f}"
                 f"<br>P(sequence) = {math.exp(h.logp):.4f}")
        out.append(NodeView(hid, h.parent, label, f"{h.logp:.1f}", st_, len(h.tokens), h.finished, hover,
                            f"{h.p:.2f}" if hid else ""))
    return out


def _candidate_table(res: TextResult, s: int) -> pd.DataFrame:
    stp = res.steps[s]
    rows = []
    for cid in sorted(stp.candidates, key=lambda c: stp.ranks[c]):
        h = res.hyps[cid]
        carried = cid in stp.carried
        decision = "✅ kept" if cid in stp.selected else "✂ pruned"
        if carried:
            decision += " (finished sentence carried over)"
        rows.append({"Rank": stp.ranks[cid], "Sequence": res.prompt + " " + _show(h.tokens) if h.tokens else res.prompt,
                     "P(word | prev)": "-" if carried or cid == 0 else f"{h.p:.2f}", "log P": round(h.logp, 3),
                     "Score": round(stp.scores[cid], 3), "Decision": decision})
    return pd.DataFrame(rows)


def render() -> None:
    for key, val in {"m5_prompt": PROMPTS[0], "m5_k": 3, "m5_len": 6, "m5_alpha": 0.0, "m5_step": 0,
                     "m5_playing": False, "m5_speed": 1.0}.items():
        ss.setdefault(key, val)

    st.markdown("## AI Text Generation Simulator")
    callout("warn", "<b>This is a simplified simulation, not a real LLM.</b> The \"model\" is a small hand-written table "
                    "P(next word | previous word) with about 40 words. Real decoders use the same beam-search "
                    "mechanics, but with neural networks producing the probabilities over tens of thousands of tokens.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.selectbox("Starting prompt", PROMPTS, key="m5_prompt", on_change=_reset_step)
        st.slider("Maximum generation length (new tokens)", 2, 10, key="m5_len", on_change=_reset_step)
    with c2:
        st.slider("Beam width k", 1, 10, key="m5_k", on_change=_reset_step)
        st.slider("Length normalisation α", 0.0, 1.0, step=0.1, key="m5_alpha", on_change=_reset_step,
                  help="Score = log P / (length^α). α = 0 uses the raw cumulative log-probability.")

    try:
        res = beam_search_text(MODEL, ss.m5_prompt, ss.m5_k, ss.m5_len, ss.m5_alpha)
    except ValueError as exc:
        st.error(str(exc))
        return
    last = len(res.steps) - 1
    ss.m5_step = min(ss.m5_step, last)

    # ---- scoring explanation ------------------------------------------------
    with st.expander("How sequences are scored", expanded=False):
        st.latex(r"\log P(y_{1:n}) \;=\; \sum_{i=1}^{n} \log P(y_i \mid y_{i-1})")
        st.latex(r"\text{score}(y_{1:n}) \;=\; \frac{\log P(y_{1:n})}{n^{\,\alpha}}\qquad(\text{higher is better})")
        st.markdown(
            "- Probabilities of the words are **multiplied**; taking logs turns the product into a **sum**, which avoids "
            "numerical underflow and is easy to rank.\n"
            "- log P is never positive, so a score *closer to 0* is better. Longer sequences accumulate more negative "
            "terms, so with α = 0 beam search favours short sentences. Raising α divides by the length and reduces that bias.\n"
            "- A hypothesis ending in ⟨eos⟩ is finished; it stays in the competition with its score unchanged.")
    with st.expander("The toy language model (every probability the simulator uses)"):
        rows = [{"Previous word": w, "Possible next words (probability)":
                 ", ".join(f"{'⟨eos⟩' if n == EOS else n} {p:.2f}" for n, p in MODEL.next_distribution(w))}
                for w in sorted(MODEL.table)]
        show_df(pd.DataFrame(rows))

    # ---- step controls --------------------------------------------------------
    b = st.columns(4)
    with b[0]:
        button("◀ Previous", key="m5_prev", on_click=_prev, disabled=ss.m5_step == 0)
    with b[1]:
        button("Next step ▶", key="m5_next", on_click=_next, args=(last,), disabled=ss.m5_step >= last or ss.m5_playing)
    with b[2]:
        button("▶ Play" if not ss.m5_playing else "⏸ Pause", key="m5_play",
               on_click=_pause if ss.m5_playing else _play, args=() if ss.m5_playing else (last,), type="primary")
    with b[3]:
        button("↺ Replay from start", key="m5_reset", on_click=_reset_step)
    st.slider("Playback speed (seconds per step)", 0.3, 3.0, step=0.1, key="m5_speed")
    st.progress(ss.m5_step / last if last else 1.0)

    s = ss.m5_step
    stp = res.steps[s]
    left, right = st.columns([3, 2])
    with left:
        title = f"Hypothesis tree after step {s} of {last}  ·  node text = cumulative log P, edge text = P(word | prev)"
        show_plot(draw_tree(_tree_nodes(res, s), show_unseen=False, height=480, title=title, show_edge_text=True),
                  key="m5_fig")
        legend([e for e in legend_entries([BEAM, PRUNED, EXPLORED, PATH])])
    with right:
        callout("info", stp.message)
        if s > 0:
            cands = sorted(stp.candidates, key=lambda c: stp.ranks[c])[:12]
            labels = [f"{stp.ranks[c]}. …{_show(res.hyps[c].tokens[-3:])}" for c in cands]
            show_plot(score_bars(labels, [stp.scores[c] for c in cands], [c in stp.selected for c in cands],
                                 f"Candidate scores (top {len(cands)} shown)", height=360), key="m5_bars")
    st.markdown("#### Candidates at this step")
    show_df(_candidate_table(res, s))

    # ---- final result -----------------------------------------------------------
    st.markdown("### Final generated sentence")
    best = res.best
    html_block(f'<div class="card"><div class="card-title">Beam search, k = {ss.m5_k}</div>'
               f'<p style="font-size:clamp(17px,2.4vw,21px);font-family:Poppins,Nunito,sans-serif;line-height:1.5">{res.sentence()}</p></div>')
    stat_cards([("log P", f"{best.logp:.3f}", "cumulative"), ("P(sequence)", f"{math.exp(best.logp):.4f}", "e^(log P)"),
                ("Tokens", len(best.tokens), "generated"), ("Finished", "yes" if best.finished else "no", "ended with ⟨eos⟩")])

    st.markdown("### Greedy decoding (k = 1) vs wider beams")
    rows = []
    results = {kk: beam_search_text(MODEL, ss.m5_prompt, kk, ss.m5_len, ss.m5_alpha) for kk in KS}
    for kk, r in results.items():
        rows.append({"k": str(kk), "Sentence": r.sentence(), "log P": round(r.best.logp, 3),
                     "P(sequence)": round(math.exp(r.best.logp), 4),
                     "Score": round(norm_score(r.best.logp, len(r.best.tokens), ss.m5_alpha), 3)})
    ref_text = ""
    if ss.m5_alpha == 0.0:
        ex = exhaustive_best(MODEL, ss.m5_prompt, ss.m5_len)
        rows.append({"k": "exhaustive (reference)", "Sentence": res.sentence(ex), "log P": round(ex.logp, 3),
                     "P(sequence)": round(math.exp(ex.logp), 4), "Score": round(ex.logp, 3)})
        hit = [kk for kk, r in results.items() if abs(r.best.logp - ex.logp) < 1e-9]
        miss = [kk for kk in KS if kk not in hit]
        ref_text = (f"Widths that found the highest-probability sentence: <b>{', '.join(map(str, hit)) or 'none'}</b>."
                    + (f" Widths that did not: <b>{', '.join(map(str, miss))}</b>." if miss else ""))
    show_df(pd.DataFrame(rows))
    g, w = results[1].best.logp, results[max(KS)].best.logp
    if ss.m5_alpha == 0.0:
        callout("info" if w >= g else "warn",
                f"Greedy (k=1) reached log P = {g:.3f}; k = {max(KS)} reached {w:.3f}. {ref_text} "
                "The \"exhaustive\" row is the true best sequence (uniform-cost search over the whole toy model); "
                "beam search is an approximation of it, which is why a narrow beam can end up below it.")
    else:
        callout("info", "With length normalisation (α > 0) the exhaustive reference row is hidden, because it is only "
                        "defined here for the raw log-probability score (α = 0).")

    if ss.m5_playing:
        if ss.m5_step >= last:
            ss.m5_playing = False
            st.rerun()
        time.sleep(ss.m5_speed)
        ss.m5_step += 1
        st.rerun()

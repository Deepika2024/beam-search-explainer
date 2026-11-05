"""Decision-explanation panel, pseudocode highlighter and the educational text of the lab."""
from __future__ import annotations

import html
from typing import List, Sequence

import pandas as pd

from algorithms.beam_search import (CHECK, DONE, EXPAND, GENERATE, INIT, PHASE_TITLES, PSEUDOCODE, SELECT, BeamStep)
from algorithms.tree import SearchTree


# ---------------------------------------------------------------------------
# pseudocode with the executing line highlighted
# ---------------------------------------------------------------------------
def pseudocode_html(highlight: Sequence[int] = ()) -> str:
    hl = set(highlight)
    lines = []
    for i, text in enumerate(PSEUDOCODE):
        cls = "ln hl" if i in hl else "ln"
        lines.append(f'<span class="{cls}"><b>{i:>2}</b>{html.escape(text)}</span>')
    return '<div class="pseudo">' + "".join(lines) + "</div>"


# ---------------------------------------------------------------------------
# Decision Explanation Panel (feature A)
# ---------------------------------------------------------------------------
def _names(tree: SearchTree, ids: Sequence[int]) -> str:
    return ", ".join(html.escape(tree.label(i)) for i in ids) if ids else "none"


def _scored(tree: SearchTree, ids: Sequence[int]) -> str:
    return ", ".join(f"{html.escape(tree.label(i))}&nbsp;(h={tree.nodes[i].h:g})" for i in ids) if ids else "none"


def step_explanation_html(step: BeamStep, tree: SearchTree, k: int, lower_is_better: bool = True) -> str:
    """The five questions of the brief, filled in progressively as the phases are reached."""
    title = f"Depth {step.depth} - {PHASE_TITLES[step.phase]}"
    head = f'<div class="card-title">{html.escape(title)}</div>'
    if step.phase == INIT:
        return f'<div class="card">{head}<p>{html.escape(step.message)}</p></div>'

    if step.phase == DONE:
        kind = "bad" if step.failed else "ok"
        return (f'<div class="card">{head}<div class="callout {kind}" style="margin:4px 0">'
                f'{html.escape(step.message)}</div></div>')
    reached = {EXPAND: 1, GENERATE: 3, SELECT: 5, CHECK: 5}[step.phase]
    word = "lowest" if lower_is_better else "highest"
    items: List[str] = []

    def item(n: int, text: str) -> None:
        cls = "" if n <= reached else ' class="todo"'
        items.append(f"<li{cls}>{text if n <= reached else '...'}</li>")

    item(1, f"<b>Expanded:</b> {_names(tree, step.expanded)}")
    item(2, f"<b>Candidates generated:</b> {len(step.candidates)} - {_names(tree, step.candidates)}")
    item(3, f"<b>Heuristic scores:</b> {_scored(tree, step.candidates)}")
    if step.phase in (SELECT, CHECK) and step.selected:
        item(4, f"<b>Top-{k} selected</b> ({word} h wins): {_scored(tree, step.selected)}")
        if step.pruned:
            sub = "".join(f"<li><b>{html.escape(tree.label(n))}</b>: {html.escape(step.reasons[n])}</li>" for n in step.pruned)
            items.append(f"<li><b>Discarded and why:</b><ul>{sub}</ul></li>")
        else:
            items.append("<li><b>Discarded:</b> none - every candidate fits in the beam.</li>")
    else:
        item(4, "")
        item(5, "")
    body = f'<ol class="step-list">{"".join(items)}</ol>'
    tail = ""
    if step.phase == CHECK:
        kind = "ok" if step.goal is not None else "info"
        tail = f'<div class="callout {kind}">{html.escape(step.message)}</div>'
    return f'<div class="card">{head}{body}</div>{tail}'


# ---------------------------------------------------------------------------
# Module 1 copy
# ---------------------------------------------------------------------------
CONCEPT_QA: List[tuple] = [
    ("What is a search algorithm?",
     "A procedure that explores a space of **states** - positions in a maze, partial sentences, game positions - "
     "by repeatedly generating successor states, looking for a path from a start state to a goal state."),
    ("Why do we need heuristic search?",
     "Search spaces explode: with branching factor *b* and depth *d* there are about *b^d* nodes. "
     "A **heuristic** h(n) estimates how promising a node is (for example, the estimated distance still to travel), "
     "so the algorithm can spend its effort on promising nodes instead of everything."),
    ("What is Beam Search?",
     "A heuristic search that works **level by level** and keeps only the *k* best-scoring nodes at each level. "
     "Every other candidate is thrown away for good. Memory stays bounded by *k*, at the price of possibly losing the solution."),
    ("Why is it called *Beam* Search?",
     "Think of a flashlight with a narrow beam: it lights up only the most promising region of the tree and leaves "
     "the rest in the dark. The wider the beam, the more of the tree you see - and the more it costs. "
     "The technique became well known through speech-recognition systems in the 1970s."),
    ("How is it different from BFS and Greedy Best-First?",
     "**BFS** keeps *every* node of a level (no pruning, no heuristic). **Greedy Best-First** keeps *every* generated "
     "node in an open list but always expands only the single best one. **Beam Search** expands *k* nodes per level in "
     "parallel and *forgets* everything else. With k = 1 it behaves like a greedy walk with no backtracking; "
     "with k large enough to hold a whole level it behaves like BFS."),
    ("What is the beam width?",
     "The integer **k**: the maximum number of nodes kept at each level. It is the single knob that trades "
     "exploration (and memory/time) against the risk of discarding the right path."),
    ("What are candidate nodes?",
     "All the children of the nodes currently in the beam. They compete for the k places in the next beam. "
     "With b children per node there are up to k x b candidates per level."),
    ("What is pruning?",
     "Permanently discarding candidates that did not make the top k. A pruned node - and the entire subtree below it - "
     "is never visited again. This is exactly why beam search is fast and why it is **not complete**."),
    ("Where is Beam Search used in AI?",
     "Decoding in **machine translation, speech recognition, text summarisation and image captioning** "
     "(choosing the most probable output sequence), handwriting/OCR decoders, and planning or scheduling problems "
     "where the full search tree is too large to keep in memory."),
]

SCORE_DIRECTION = pd.DataFrame([
    {"Demonstration": "Search tree & beam-width playground", "Score": "heuristic h(n) = estimated cost to goal",
     "Preferred": "LOWER is better"},
    {"Demonstration": "Maze solver", "Score": "h(n) = Manhattan distance to goal", "Preferred": "LOWER is better"},
    {"Demonstration": "Text generation", "Score": "cumulative log-probability log P(sequence)",
     "Preferred": "HIGHER is better (closer to 0)"},
])

COMPARISON_TABLE = pd.DataFrame([
    {"": "Strategy", "Breadth-First Search": "Expand all nodes level by level",
     "Greedy Best-First": "Always expand the single best-h node in the open list",
     "Beam Search": "Expand the k best nodes of each level, discard the rest"},
    {"": "Uses a heuristic?", "Breadth-First Search": "No", "Greedy Best-First": "Yes: f(n) = h(n)",
     "Beam Search": "Yes: f(n) = h(n) to rank candidates"},
    {"": "Nodes retained", "Breadth-First Search": "Whole frontier (up to b^d)",
     "Greedy Best-First": "Whole open list (can still be large)", "Beam Search": "At most k per level"},
    {"": "Memory", "Breadth-First Search": "O(b^d)", "Greedy Best-First": "O(b^m) worst case (m = max depth)",
     "Beam Search": "O(k·b) for candidates, O(k) retained"},
    {"": "Complete?", "Breadth-First Search": "Yes (finite branching)",
     "Greedy Best-First": "Yes on finite spaces with repeated-state checking", "Beam Search": "No"},
    {"": "Optimal?", "Breadth-First Search": "Yes for unit step costs", "Greedy Best-First": "No",
     "Beam Search": "No"},
])

BEAM_WIDTH_SMALL = [
    "Less exploration: few paths are followed at once.",
    "Lower memory and time per level.",
    "May discard the one promising path - the heuristic is only an estimate.",
]
BEAM_WIDTH_LARGE = [
    "More exploration: many alternatives stay alive.",
    "Higher memory and time per level (about k·b candidates).",
    "Better chance of keeping a good path - but **no guarantee of optimality**: a wider beam can even reach a worse goal first.",
]

APPLICATIONS = [
    ("Machine translation & summarisation", "Decoders keep the k most probable partial translations at every step."),
    ("Speech recognition", "Keeps the k best partial transcriptions while scanning the audio."),
    ("Image captioning", "Generates captions word by word while keeping several candidate sentences."),
    ("OCR / handwriting recognition", "Decodes character probabilities into the most likely text."),
    ("Planning & scheduling", "Limits the number of partial plans kept when the full tree would not fit in memory."),
]

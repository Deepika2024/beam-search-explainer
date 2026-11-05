"""Quiz questions. Conceptual questions are fixed; the 'live' ones are computed by running the real algorithms,
so their correct answers can never drift out of sync with the code."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from algorithms.beam_search import SELECT, beam_search
from algorithms.text_generation import ToyLanguageModel, beam_search_text
from data.sample_search_trees import cost_trap_tree, failure_tree, random_tree


@dataclass
class Question:
    kind: str                      # "Multiple choice" | "True / False" | "Predict the next step" | "Identify the pruned node" | "Beam-width experiment" | "Calculation"
    prompt: str
    options: List[str]
    answer: List[str]              # correct option(s)
    explanation: str
    multi: bool = False
    table: Optional[pd.DataFrame] = None


def _mc(prompt: str, options: List[str], correct: str, why: str, kind: str = "Multiple choice") -> Question:
    return Question(kind, prompt, options, [correct], why)


def _tf(prompt: str, truth: bool, why: str) -> Question:
    return Question("True / False", prompt, ["True", "False"], ["True" if truth else "False"], why)


def static_questions() -> List[Question]:
    return [
        _mc("With beam width **k = 1**, beam search behaves most like which algorithm?",
            ["Breadth-First Search", "A greedy walk that never backtracks", "Dijkstra's algorithm", "Depth-limited exhaustive search"],
            "A greedy walk that never backtracks",
            "With k = 1 only the single best candidate survives each level, so every other branch is lost for good - "
            "exactly a greedy walk without backtracking."),
        _tf("Increasing the beam width always returns a path with lower or equal cost.", False,
            "False. A wider beam can reach a different goal first, and that goal can be costlier. "
            "The cost-trap experiment in Module 3 shows k = 1 beating every wider beam."),
        _mc("What does *pruning* mean in beam search?",
            ["Re-ordering candidates so the best come first",
             "Permanently discarding every candidate outside the top k",
             "Removing nodes whose heuristic is zero",
             "Stopping the search at a maximum depth"],
            "Permanently discarding every candidate outside the top k",
            "Pruned candidates, and everything below them, are never visited again - the source of both the speed and the incompleteness."),
        _tf("If k is at least as large as the number of nodes at every level, beam search never prunes anything.", True,
            "True. All candidates then fit in the beam, so the search explores level by level like BFS."),
        _tf("Beam search ranks its candidates using the path cost g(n) already paid to reach them.", False,
            "False. In this lab f(n) = h(n): only the heuristic is used. Algorithms such as A* use g(n) + h(n)."),
        _mc("Why do text decoders add log-probabilities instead of multiplying probabilities?",
            ["Because log-probabilities are always larger",
             "A sum avoids numerical underflow of long products and is easy to rank",
             "Because probabilities cannot be compared directly",
             "Because it makes the model deterministic"],
            "A sum avoids numerical underflow of long products and is easy to rank",
            "Multiplying many numbers below 1 quickly underflows. log(ab) = log a + log b turns the product into a sum with identical ordering."),
        _tf("With raw cumulative log-probability (no length normalisation), beam search tends to prefer shorter sequences.", True,
            "True. Every extra word adds a negative log-probability term, so longer sequences score lower unless normalised by length."),
        _mc("Which statement about memory is correct?",
            ["Beam search needs O(b^d) memory like BFS",
             "Beam search keeps at most k nodes per level, plus about k·b candidates while selecting",
             "Beam search memory grows with the size of the whole tree",
             "Beam search needs no memory"],
            "Beam search keeps at most k nodes per level, plus about k·b candidates while selecting",
            "That bound is independent of how many nodes the full tree contains, which is the point of the algorithm."),
    ]


def _predict_tree_question(seed: int) -> Question:
    """Pick a random tree where the cut-off between rank k and k+1 is not a tie, then ask for the survivors."""
    for s in range(seed, seed + 400):
        tree = random_tree(3, 3, s, noise=2.0)
        res = beam_search(tree, 2)
        sels = [st for st in res.steps if st.phase == SELECT]
        if len(sels) < 2:
            continue
        stp = sels[1]
        hs = sorted(tree.nodes[c].h for c in stp.candidates)
        if len(hs) < 4 or hs[1] == hs[2]:
            continue
        prev = sels[0].selected
        rows = [{"Candidate": tree.label(c), "Parent": tree.label(tree.nodes[c].parent), "h": tree.nodes[c].h}
                for c in stp.candidates]
        ranked = ", ".join(f"{tree.label(c)} (h={tree.nodes[c].h:g})" for c in stp.selected + stp.pruned)
        return Question(
            "Predict the next step",
            f"Beam width **k = 2**; lower h is better. The beam is now **{{{', '.join(tree.label(n) for n in prev)}}}**. "
            "Expanding it produced the candidates below. Which **two** survive into the next beam?",
            [tree.label(c) for c in stp.candidates], [tree.label(c) for c in stp.selected],
            f"Ranked best to worst: {ranked}. The two lowest h-values are kept; the rest are pruned.",
            multi=True, table=pd.DataFrame(rows))
    raise RuntimeError("no suitable tree found")


def _pruned_question(rng: random.Random) -> Question:
    labels = ["P", "Q", "R", "S", "T"]
    vals = rng.sample(range(10, 99), 5)
    higher = rng.random() < 0.5
    scores = [v / 100 if higher else v / 10 for v in vals]
    pruned_i = vals.index(min(vals) if higher else max(vals))
    meaning = ("**probabilities** (higher is better)" if higher
               else "**estimated distances to the goal** (lower is better)")
    return Question(
        "Identify the pruned node",
        f"A beam with **k = 4** receives these 5 candidates. Their scores are {meaning}. Which candidate is pruned?",
        labels, [labels[pruned_i]],
        f"Four of the five fit in the beam. The worst-scoring candidate is {labels[pruned_i]} "
        f"(score {scores[pruned_i]:g}), so it is the one discarded.",
        table=pd.DataFrame({"Candidate": labels, "Score": scores}))


def _experiment_question() -> Question:
    tree = failure_tree()
    ks = [1, 2, 3, 5, 10]
    ok = [str(k) for k in ks if beam_search(tree, k, record=False).found]
    return Question(
        "Beam-width experiment",
        "Experiment (Module 3, *Narrow beam misses the goal*): the tree contains one goal. "
        "Which of these beam widths **find** it? Select all that apply.",
        [str(k) for k in ks], ok,
        f"Widths that find the goal: {', '.join(ok)}. The goal's ancestor ranks 5th at depth 2, so any beam narrower "
        "than 5 prunes it.", multi=True)


def _trap_question() -> Question:
    tree = cost_trap_tree()
    r1, r3 = beam_search(tree, 1, record=False), beam_search(tree, 3, record=False)
    if r1.cost < r3.cost:
        ans = "k = 1"
    elif r3.cost < r1.cost:
        ans = "k = 3"
    else:
        ans = "They return the same cost"
    return Question(
        "Beam-width experiment",
        "Experiment (Module 3, *Wider beam is not optimal*): which beam width returns the **cheaper** path?",
        ["k = 1", "k = 3", "They return the same cost"], [ans],
        f"k = 1 returned cost {r1.cost:g} and k = 3 returned cost {r3.cost:g}. A wider beam reached a goal sooner, "
        "but through an expensive edge.")


def _count_question(rng: random.Random) -> Question:
    b, k, i = rng.choice([3, 4, 5]), rng.choice([2, 3, 4]), rng.choice([1, 2, 3])
    answer = b * min(b ** (i - 1), k)
    distract = {k * b, b ** i, k, b, k * b * i, b + k, answer + 1, answer * 2, answer + 3} - {answer, 0}
    options = [answer] + rng.sample(sorted(distract), 3)
    rng.shuffle(options)
    return Question(
        "Calculation",
        f"Branching factor **b = {b}**, beam width **k = {k}**. How many candidates are generated when the beam at "
        f"depth **{i - 1}** is expanded?",
        [str(o) for o in options], [str(answer)],
        f"The beam at depth {i - 1} holds min(b^{i - 1}, k) = {min(b ** (i - 1), k)} node(s); each has {b} children, "
        f"giving {answer} candidates. (k·b is only the upper bound, reached once the beam is full.)")


def _text_question() -> Question:
    model = ToyLanguageModel()
    prompt = "Artificial Intelligence is"
    r1, r2 = beam_search_text(model, prompt, 1, 6), beam_search_text(model, prompt, 2, 6)
    if r2.best.logp > r1.best.logp + 1e-12:
        ans = "k = 2"
    elif r1.best.logp > r2.best.logp + 1e-12:
        ans = "k = 1"
    else:
        ans = "The same"
    return Question(
        "Beam-width experiment",
        "Experiment (Module 5, prompt *Artificial Intelligence is*, max length 6, α = 0): which beam width yields the "
        "sentence with the **higher cumulative log-probability**?",
        ["k = 1", "k = 2", "The same"], [ans],
        f"k = 1 gives \"{r1.sentence()}\" (log P = {r1.best.logp:.3f}); k = 2 gives \"{r2.sentence()}\" "
        f"(log P = {r2.best.logp:.3f}). Greedy chose the locally best word 'a' (0.40), but 'transforming' (0.35) leads "
        "to a much more probable sentence overall.")


def _text_step_question() -> Question:
    model = ToyLanguageModel()
    dist = model.next_distribution("is")
    res = beam_search_text(model, "Artificial Intelligence is", 2, 1)
    kept = [res.hyps[h].word for h in res.steps[1].selected]
    return Question(
        "Predict the next step",
        "Text generation, prompt *Artificial Intelligence is*, **k = 2**. The model proposes the next words below. "
        "Which two hypotheses survive step 1? (higher log-probability is better)",
        [w for w, _ in dist], kept,
        "Log-probability is monotonic in probability, so the two most probable words, "
        + " and ".join(f"'{w}'" for w in kept) + ", are kept.", multi=True,
        table=pd.DataFrame({"Next word": [w for w, _ in dist], "P(word | is)": [p for _, p in dist],
                            "log P": [round(math.log(p), 3) for _, p in dist]}))


def build_questions(seed: int) -> List[Question]:
    """Fixed + live questions, in a shuffled order that depends on ``seed`` (a new seed = a new quiz)."""
    rng = random.Random(seed)
    qs = static_questions() + [_predict_tree_question(seed), _pruned_question(rng), _experiment_question(),
                               _trap_question(), _count_question(rng), _text_question(), _text_step_question()]
    for q in qs:                                    # only plain multiple-choice options are shuffled
        if q.kind == "Multiple choice":
            rng.shuffle(q.options)
    rng.shuffle(qs)
    return qs

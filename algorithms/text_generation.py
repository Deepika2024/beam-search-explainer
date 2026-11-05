"""Beam-search text decoding on a tiny, transparent bigram language model.

This is a *simplified simulation* of how real decoders (machine translation,
speech recognition, LLM sampling heads) use beam search. It is NOT a real LLM:
the "model" is a hand-written table P(next word | previous word).

Scoring: a sequence y_1..y_n has cumulative log-probability

    log P(y_1..y_n) = sum_i log P(y_i | y_{i-1})

Log-probabilities are used because products of many small probabilities
underflow, and because sums are easier to rank and to explain. Higher is better
(log P <= 0, so the "best" score is the one closest to 0).
Optional length normalisation ranks hypotheses by  log P / n^alpha.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Optional, Set, Tuple

EOS = "<eos>"

# P(next | previous). Every row sums to 1 (checked in ToyLanguageModel).
TOY_TABLE: Dict[str, Dict[str, float]] = {
    # sentence openers after the prompt
    "is": {"a": 0.40, "transforming": 0.35, "the": 0.15, "becoming": 0.10},
    "can": {"solve": 0.45, "learn": 0.30, "help": 0.25},
    "will": {"change": 0.50, "transform": 0.30, "help": 0.20},
    # branch A: "a ..." (many similar options -> probability is spread thin)
    "a": {"powerful": 0.30, "new": 0.25, "smart": 0.25, "simple": 0.20},
    "powerful": {"tool": 0.50, "technology": 0.30, "system": 0.20},
    "new": {"technology": 0.50, "system": 0.30, "tool": 0.20},
    "smart": {"system": 0.50, "tool": 0.30, "technology": 0.20},
    "simple": {"tool": 0.40, "system": 0.30, "technology": 0.30},
    "tool": {EOS: 0.60, "for": 0.25, "today": 0.15},
    "technology": {EOS: 0.50, "today": 0.30, "for": 0.20},
    "system": {EOS: 0.55, "today": 0.25, "for": 0.20},
    # branch B: "transforming the world" (confident, concentrated probabilities)
    "transforming": {"the": 0.60, "every": 0.25, "modern": 0.15},
    "the": {"world": 0.70, "future": 0.20, "industry": 0.10},
    "every": {"industry": 0.60, "task": 0.40},
    "modern": {"world": 0.50, "industry": 0.30, "work": 0.20},
    "world": {EOS: 0.80, "today": 0.20},
    "future": {EOS: 0.70, "of": 0.30},
    "industry": {EOS: 0.60, "today": 0.40},
    "becoming": {"a": 0.50, "the": 0.50},
    "of": {"work": 0.60, "industry": 0.40},
    "for": {"work": 0.50, "everyone": 0.50},
    "work": {EOS: 0.70, "today": 0.30},
    "everyone": {EOS: 1.0},
    "today": {EOS: 1.0},
    "task": {EOS: 1.0},
    # other openers
    "solve": {"complex": 0.60, "hard": 0.40},
    "complex": {"problems": 0.70, "tasks": 0.30},
    "hard": {"problems": 1.0},
    "problems": {EOS: 0.80, "today": 0.20},
    "tasks": {EOS: 1.0},
    "learn": {"from": 0.70, "quickly": 0.30},
    "from": {"data": 0.70, "experience": 0.30},
    "data": {EOS: 0.60, "today": 0.40},
    "experience": {EOS: 1.0},
    "quickly": {EOS: 1.0},
    "help": {"people": 0.50, "work": 0.30, "everyone": 0.20},
    "people": {EOS: 0.70, "today": 0.30},
    "change": {"the": 0.60, "every": 0.40},
    "transform": {"the": 0.70, "every": 0.30},
}

PROMPTS = ["Artificial Intelligence is", "Artificial Intelligence can", "Artificial Intelligence will"]


class ToyLanguageModel:
    """Validated lookup table P(next | previous)."""

    def __init__(self, table: Dict[str, Dict[str, float]] = None) -> None:
        self.table = table if table is not None else TOY_TABLE
        self._validate()

    def _validate(self) -> None:
        vocab = set(self.table)
        for w, row in self.table.items():
            total = sum(row.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"Probabilities after '{w}' sum to {total}, not 1.")
            for nxt, p in row.items():
                if p <= 0:
                    raise ValueError(f"Non-positive probability for '{w}' -> '{nxt}'.")
                if nxt != EOS and nxt not in vocab:
                    raise ValueError(f"'{nxt}' (after '{w}') has no row in the table.")

    @property
    def vocabulary(self) -> List[str]:
        words = set(self.table)
        for row in self.table.values():
            words.update(row)
        return sorted(words)

    def next_distribution(self, word: str) -> List[Tuple[str, float]]:
        """All possible next words with probabilities, most likely first."""
        row = self.table.get(word)
        if row is None:
            raise KeyError(f"'{word}' is not in the toy vocabulary.")
        return sorted(row.items(), key=lambda kv: -kv[1])


@dataclass
class Hypothesis:
    id: int
    parent: Optional[int]
    tokens: Tuple[str, ...]          # generated words (prompt excluded)
    word: str                        # last generated word ('' for the root)
    p: float                         # P(word | previous word)
    logp: float                      # cumulative log-probability
    finished: bool = False
    born: int = 0                    # generation step in which it was created


@dataclass
class TextStep:
    index: int                       # 0 = prompt only
    candidates: List[int]            # hypothesis ids competing at this step (incl. carried finished ones)
    selected: List[int]
    pruned: List[int]
    ranks: Dict[int, int]
    scores: Dict[int, float]         # ranking score used (normalised log-prob)
    carried: Set[int]                # finished hypotheses that were carried over unchanged
    status: Dict[int, str]           # hypothesis id -> "beam" | "pruned" | "explored"
    message: str


@dataclass
class TextResult:
    prompt: str
    k: int
    max_new_tokens: int
    alpha: float
    hyps: Dict[int, Hypothesis]
    steps: List[TextStep]
    final_beam: List[int]
    best: Hypothesis

    def sentence(self, h: Optional[Hypothesis] = None) -> str:
        h = h or self.best
        words = [w for w in h.tokens if w != EOS]
        text = (self.prompt + " " + " ".join(words)).strip()
        return text + ("." if h.finished else " ...")


def norm_score(logp: float, n_tokens: int, alpha: float) -> float:
    """Length-normalised score  logP / n^alpha  (alpha = 0 -> raw cumulative log-probability)."""
    return logp / (max(n_tokens, 1) ** alpha)


def beam_search_text(model: ToyLanguageModel, prompt: str, k: int, max_new_tokens: int = 6,
                     alpha: float = 0.0) -> TextResult:
    """Beam search decoding. Higher score is better. Finished (<eos>) hypotheses are carried unchanged."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("Beam width k must be a positive integer.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    words = prompt.lower().split()
    if not words or words[-1] not in model.table:
        raise ValueError(f"The last word of the prompt must be in the toy vocabulary; got '{prompt}'.")
    last_prompt_word = words[-1]

    hyps: Dict[int, Hypothesis] = {0: Hypothesis(0, None, (), "", 1.0, 0.0, False, 0)}
    beam: List[int] = [0]
    steps: List[TextStep] = [TextStep(0, [0], [0], [], {0: 1}, {0: 0.0}, set(), {0: "beam"},
                                      f"Start from the prompt \"{prompt}\". Cumulative log-probability = 0.")]
    next_id = 1
    for t in range(1, max_new_tokens + 1):
        if all(hyps[h].finished for h in beam):
            break
        cands: List[int] = []
        carried: Set[int] = set()
        for hid in beam:
            h = hyps[hid]
            if h.finished:                      # a finished sentence competes with its score unchanged
                cands.append(hid)
                carried.add(hid)
                continue
            prev = h.word if h.word else last_prompt_word
            for w, p in model.next_distribution(prev):
                nh = Hypothesis(next_id, hid, h.tokens + (w,), w, p, h.logp + math.log(p), w == EOS, t)
                hyps[next_id] = nh
                cands.append(next_id)
                next_id += 1
        scores = {c: norm_score(hyps[c].logp, len(hyps[c].tokens), alpha) for c in cands}
        ranked = sorted(cands, key=lambda c: -scores[c])         # stable: ties keep generation order
        selected, dropped = ranked[:k], ranked[k:]
        ranks = {c: i + 1 for i, c in enumerate(ranked)}
        status: Dict[int, str] = {}
        for hid, h in hyps.items():
            if hid in selected:
                status[hid] = "beam"
            elif hid in dropped or steps[-1].status.get(hid) == "pruned":
                status[hid] = "pruned"
            else:
                status[hid] = "explored"
        beam = list(selected)
        steps.append(TextStep(t, cands, selected, dropped, ranks, scores, carried, status,
                              f"Step {t}: scored {len(cands)} candidate sequence(s), kept the best {len(selected)}, "
                              f"discarded {len(dropped)}."))
    final = sorted(beam, key=lambda c: -norm_score(hyps[c].logp, len(hyps[c].tokens), alpha))
    return TextResult(prompt, k, max_new_tokens, alpha, hyps, steps, final, hyps[final[0]])


def exhaustive_best(model: ToyLanguageModel, prompt: str, max_new_tokens: int = 6) -> Hypothesis:
    """Highest raw log-probability sequence (finished or at max length) via uniform-cost search.

    Extending a sequence can only lower its probability, so the first *complete*
    sequence popped from a probability-ordered heap is the global optimum
    (valid for alpha = 0). It is used as a reference to show that beam search
    is an approximation.
    """
    last = prompt.lower().split()[-1]
    tie = count()
    heap: List[Tuple[float, int, Tuple[str, ...], bool]] = [(0.0, next(tie), (), False)]
    while heap:
        neg, _, toks, fin = heapq.heappop(heap)
        if fin or len(toks) == max_new_tokens:
            return Hypothesis(-1, None, toks, toks[-1] if toks else "", 1.0, -neg, fin, len(toks))
        prev = toks[-1] if toks else last
        for w, p in model.next_distribution(prev):
            heapq.heappush(heap, (neg - math.log(p), next(tie), toks + (w,), w == EOS))
    raise RuntimeError("unreachable")

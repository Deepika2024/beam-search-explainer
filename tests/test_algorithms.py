"""Algorithm correctness tests (run:  python -m unittest discover -s tests -v)."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.analysis import totals
from algorithms.beam_search import (BEAM, PRUNED, SELECT, beam_search, beam_search_grid, top_k)
from algorithms.bfs import bfs_grid, bfs_tree
from algorithms.greedy_best_first import greedy_grid, greedy_tree
from algorithms.grid import Grid, random_maze
from algorithms.text_generation import (EOS, TOY_TABLE, ToyLanguageModel, beam_search_text, exhaustive_best)
from data.quiz_bank import build_questions
from data.sample_search_trees import (cost_trap_tree, default_grid, demo_tree, failure_tree, random_tree, uniform_tree)


class TestTopK(unittest.TestCase):
    def test_keeps_k_best_and_is_stable(self):
        kept, dropped = top_k(["a", "b", "c", "d"], 2, key={"a": 3, "b": 1, "c": 1, "d": 2}.get)
        self.assertEqual(kept, ["b", "c"])           # tie between b and c keeps generation order
        self.assertEqual(dropped, ["d", "a"])

    def test_fewer_candidates_than_k(self):
        kept, dropped = top_k([1, 2], 5, key=lambda x: x)
        self.assertEqual((kept, dropped), ([1, 2], []))


class TestTreeBeam(unittest.TestCase):
    def test_invalid_k(self):
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                beam_search(demo_tree(), bad)

    def test_demo_tree_k2_by_hand(self):
        t = demo_tree()
        r = beam_search(t, 2)
        sel = [s for s in r.steps if s.phase == SELECT]
        lab = lambda ids: [t.label(i) for i in ids]
        self.assertEqual(lab(sel[0].selected), ["B", "A"])     # h: B=3, A=5, C=6 pruned
        self.assertEqual(lab(sel[0].pruned), ["C"])
        # beam nodes are expanded in rank order (B before A), so B3 (h=4) is generated before A1 (h=4) and wins the tie
        self.assertEqual(lab(sel[1].selected), ["B1", "B3"])
        self.assertEqual(lab(sel[2].selected), ["G", "B3a"])
        self.assertTrue(r.found)
        self.assertEqual(lab(r.path), ["S", "B", "B1", "G"])

    def test_invariants_on_random_trees(self):
        for seed in range(15):
            t = random_tree(3, 3, seed, noise=2.5)
            for k in (1, 2, 3, 5):
                r = beam_search(t, k)
                for s in r.steps:
                    self.assertLessEqual(len(s.beam), k)
                    self.assertFalse(set(s.selected) & set(s.pruned))
                    self.assertEqual(sorted(s.selected + s.pruned), sorted(s.candidates) if s.phase == SELECT else sorted(s.selected + s.pruned))
                # every selected node must be at least as good as every pruned node of the same step
                for s in r.steps:
                    if s.phase == SELECT and s.pruned:
                        self.assertLessEqual(max(t.nodes[n].h for n in s.selected), min(t.nodes[n].h for n in s.pruned))
                self.assertEqual(r.found, r.goal is not None)
                if r.found:
                    self.assertEqual(r.path[0], t.root)
                    self.assertTrue(t.nodes[r.path[-1]].is_goal)

    def test_higher_is_better_flips_order(self):
        t = demo_tree()
        r = beam_search(t, 1, lower_is_better=False)
        first = [s for s in r.steps if s.phase == SELECT][0]
        self.assertEqual(t.label(first.selected[0]), "C")       # C has the highest h among A=5, B=3, C=6

    def test_failure_demo_needs_wide_beam(self):
        t = failure_tree()
        found = {k: beam_search(t, k).found for k in (1, 2, 3, 5, 10)}
        self.assertEqual(found, {1: False, 2: False, 3: False, 5: True, 10: True})
        miss = beam_search(t, 3)
        self.assertTrue(miss.lost_goals)                        # explains which ancestor was pruned
        self.assertIn("pruned", miss.failure_reason)

    def test_wider_beam_not_optimal(self):
        t = cost_trap_tree()
        r1, r3 = beam_search(t, 1), beam_search(t, 3)
        self.assertEqual(r1.cost, t.optimal_cost())
        self.assertGreater(r3.cost, r1.cost)

    def test_huge_beam_matches_bfs_depth(self):
        for seed in range(10):
            t = random_tree(3, 3, seed, noise=3.0)
            big = beam_search(t, 10_000)
            bfs = bfs_tree(t)
            self.assertTrue(big.found)
            self.assertEqual(len(big.path), len(bfs.path))       # shallowest goal, same as BFS
            self.assertEqual(big.stats["pruned"], 0)

    def test_max_depth_and_root_goal(self):
        t = demo_tree()
        r = beam_search(t, 2, max_depth=1)
        self.assertFalse(r.found)
        t2 = demo_tree()
        t2.nodes[t2.root].is_goal = True
        r2 = beam_search(t2, 1)
        self.assertTrue(r2.found)
        self.assertEqual(r2.path, [t2.root])

    def test_statuses_are_exclusive_and_final_path_marked(self):
        t = demo_tree()
        r = beam_search(t, 2)
        last = r.steps[-1]
        self.assertTrue(all(last.status[n] == "path" for n in r.path))
        for s in r.steps:
            self.assertEqual(set(s.status), set(t.nodes))


class TestOtherSearches(unittest.TestCase):
    def test_bfs_and_greedy_find_goal_where_beam_fails(self):
        t = failure_tree()
        self.assertFalse(beam_search(t, 1).found)
        self.assertTrue(bfs_tree(t).found)
        self.assertTrue(greedy_tree(t).found)

    def test_formula_matches_real_run(self):
        for b, d, k in [(2, 4, 3), (3, 3, 4), (3, 4, 1), (4, 3, 10), (2, 5, 2)]:
            t = uniform_tree(b, d)
            r = beam_search(t, k, max_depth=d, record=False)
            tot = totals(b, d, k)
            self.assertEqual(int(r.stats["generated"]), tot["beam_generated"])
            self.assertEqual(int(r.stats["expanded"]), tot["beam_expanded"])


class TestGrid(unittest.TestCase):
    def _valid_path(self, g, path):
        self.assertEqual(path[0], g.start)
        self.assertEqual(path[-1], g.goal)
        for a, b in zip(path, path[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
            self.assertTrue(g.passable(b))

    def test_default_maze_story(self):
        g = default_grid()
        self.assertFalse(beam_search_grid(g, 1).found)           # narrow beam trapped
        self.assertTrue(bfs_grid(g).found)                       # but a path exists
        r2, r3 = beam_search_grid(g, 2), beam_search_grid(g, 3)
        self.assertTrue(r2.found and r3.found)
        self.assertGreater(r2.path_length, r3.path_length)       # k=2 detours, k=3 shortest
        self.assertEqual(r3.path_length, int(bfs_grid(g).cost))
        for r in (r2, r3):
            self._valid_path(g, r.path)

    def test_unreachable_goal_is_reported(self):
        g = Grid(5, 5, (0, 0), (4, 4), {(3, 4), (4, 3), (3, 3)})   # goal walled in
        for k in (1, 3, 50):
            self.assertFalse(beam_search_grid(g, k).found)
        self.assertFalse(bfs_grid(g).found)
        self.assertFalse(greedy_grid(g).found)

    def test_random_mazes_never_worse_than_invalid(self):
        for seed in range(20):
            g = random_maze(10, 10, 0.3, seed)
            b = bfs_grid(g)
            for k in (1, 2, 5, 100):
                r = beam_search_grid(g, k)
                if r.found:
                    self._valid_path(g, r.path)
                    self.assertTrue(b.found)
                    self.assertGreaterEqual(r.path_length, b.cost)   # never shorter than the true shortest
                self.assertLessEqual(r.stats["max_beam"], k)

    def test_validation(self):
        with self.assertRaises(ValueError):
            beam_search_grid(Grid(4, 4, (0, 0), (9, 9)), 2)
        with self.assertRaises(ValueError):
            beam_search_grid(Grid(4, 4, (0, 0), (1, 1), {(0, 0)}), 2)
        g = Grid(4, 4, (0, 0), (0, 0))
        self.assertTrue(beam_search_grid(g, 1).found)              # start == goal

    def test_editing_rules(self):
        g = Grid(5, 5, (0, 0), (4, 4))
        self.assertFalse(g.toggle_obstacle((0, 0)))
        self.assertFalse(g.toggle_obstacle((4, 4)))
        self.assertTrue(g.toggle_obstacle((2, 2)))
        self.assertTrue(g.set_start((2, 2)))                      # moving start onto an obstacle clears it
        self.assertNotIn((2, 2), g.obstacles)
        self.assertFalse(g.set_goal((2, 2)))                      # goal cannot equal start


class TestText(unittest.TestCase):
    def test_model_is_valid(self):
        m = ToyLanguageModel()
        for row in TOY_TABLE.values():
            self.assertAlmostEqual(sum(row.values()), 1.0)
        with self.assertRaises(ValueError):
            ToyLanguageModel({"a": {"b": 0.5}, "b": {EOS: 1.0}})

    def test_scores_are_cumulative_log_probs(self):
        m = ToyLanguageModel()
        r = beam_search_text(m, "Artificial Intelligence is", 3, 6)
        h = r.best
        prev, expected = "is", 0.0
        for w in h.tokens:
            expected += math.log(TOY_TABLE[prev][w])
            prev = w
        self.assertAlmostEqual(h.logp, expected)

    def test_beam_beats_greedy_and_matches_exhaustive(self):
        m = ToyLanguageModel()
        g = beam_search_text(m, "Artificial Intelligence is", 1, 6)
        b = beam_search_text(m, "Artificial Intelligence is", 2, 6)
        ex = exhaustive_best(m, "Artificial Intelligence is", 6)
        self.assertLess(g.best.logp, b.best.logp)
        self.assertAlmostEqual(b.best.logp, ex.logp)

    def test_beam_size_and_errors(self):
        m = ToyLanguageModel()
        r = beam_search_text(m, "Artificial Intelligence can", 4, 5)
        for s in r.steps:
            self.assertLessEqual(len(s.selected), 4)
        with self.assertRaises(ValueError):
            beam_search_text(m, "Artificial Intelligence is", 0, 5)
        with self.assertRaises(ValueError):
            beam_search_text(m, "zzz qqq", 2, 5)
        with self.assertRaises(ValueError):
            beam_search_text(m, "Artificial Intelligence is", 2, 5, alpha=2.0)

    def test_length_normalisation_changes_preference(self):
        m = ToyLanguageModel()
        raw = beam_search_text(m, "Artificial Intelligence is", 3, 8, alpha=0.0)
        norm = beam_search_text(m, "Artificial Intelligence is", 3, 8, alpha=1.0)
        self.assertGreaterEqual(len(norm.best.tokens), len(raw.best.tokens))


class TestQuiz(unittest.TestCase):
    def test_every_question_is_well_formed(self):
        for seed in (1, 2, 3, 77, 4242):
            for q in build_questions(seed):
                self.assertTrue(q.answer)
                self.assertTrue(set(q.answer) <= set(q.options), q.prompt)
                self.assertEqual(len(set(q.options)), len(q.options), q.prompt)
                if not q.multi:
                    self.assertEqual(len(q.answer), 1)


if __name__ == "__main__":
    unittest.main()

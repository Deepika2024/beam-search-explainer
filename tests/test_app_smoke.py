"""UI smoke tests: drive the real Streamlit app headlessly (run:  python -m unittest discover -s tests -v)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest

from components.sidebar import PAGES

APP = os.path.join(ROOT, "app.py")


def new_app(page=0, timeout=90):
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.session_state["nav"] = PAGES[page]
    at.run()
    return at


def ok(test, at):
    test.assertEqual(len(at.exception), 0, [e.value for e in at.exception])


class TestEveryPageRenders(unittest.TestCase):
    def test_all_pages(self):
        for i, name in enumerate(PAGES):
            with self.subTest(page=name):
                ok(self, new_app(i))


class TestModule2(unittest.TestCase):
    def test_controls(self):
        at = new_app(1)
        ok(self, at)
        at.button(key="m2_btn_start").click().run(); ok(self, at)
        self.assertTrue(at.session_state["m2_started"])
        for _ in range(4):
            at.button(key="m2_btn_next").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m2_step"], 4)
        at.button(key="m2_btn_prev").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m2_step"], 3)
        at.button(key="m2_btn_reset").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m2_step"], 0)
        self.assertFalse(at.session_state["m2_started"])

    def test_change_k_and_scenario_rerun_algorithm(self):
        at = new_app(1)
        at.select_slider(key="m2_k").set_value(5).run(); ok(self, at)
        self.assertEqual(at.session_state["m2_result"].k, 5)
        at.selectbox(key="m2_scenario").set_value("Cost trap: wider beam is not better").run(); ok(self, at)
        self.assertEqual(at.session_state["m2_tree"].name, "Cost trap: a wider beam is not better")

    def test_autoplay_reaches_end_then_stops(self):
        at = new_app(1)
        at.session_state["m2_speed"] = 0.3
        at.button(key="m2_btn_auto").click().run(); ok(self, at)
        self.assertFalse(at.session_state["m2_playing"])
        self.assertEqual(at.session_state["m2_step"], len(at.session_state["m2_result"].steps) - 1)

    def test_prediction_mode(self):
        at = new_app(1)
        at.toggle(key="m2_predict").set_value(True).run()
        at.button(key="m2_btn_start").click().run()
        for _ in range(2):                                   # init -> expand -> generate
            at.button(key="m2_btn_next").click().run()
        step = at.session_state["m2_result"].steps[at.session_state["m2_step"]]
        self.assertEqual(step.phase, "generate")
        self.assertTrue(at.button(key="m2_btn_next").disabled)   # blocked until a prediction is locked
        at.button(key="m2_btn_lock").click().run(); ok(self, at)
        self.assertFalse(at.button(key="m2_btn_next").disabled)
        at.button(key="m2_btn_next").click().run(); ok(self, at)
        self.assertIn("m2_pred", at.session_state)


class TestModule3(unittest.TestCase):
    def test_experiments(self):
        at = new_app(2)
        ok(self, at)
        at.button(key="m3_exp_trap").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m3_k"], 2)
        at.button(key="m3_exp_fail").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m3_k"], 1)
        for k in (2, 3, 5, 10):
            at.select_slider(key="m3_k").set_value(k).run(); ok(self, at)
        at.selectbox(key="m3_scenario").set_value("Random tree (you choose the settings)").run(); ok(self, at)


class TestModule4(unittest.TestCase):
    def test_maze_buttons(self):
        at = new_app(3)
        ok(self, at)
        at.button(key="m4_btn_exec").click().run(); ok(self, at)
        self.assertIsNotNone(at.session_state["m4_trace"])
        self.assertFalse(at.session_state["m4_trace"].found if at.session_state["m4_k"] < 2 else False)
        at.slider(key="m4_k").set_value(1).run(); ok(self, at)
        self.assertFalse(at.session_state["m4_trace"].found)         # k=1 trapped in the example maze
        at.slider(key="m4_k").set_value(3).run(); ok(self, at)
        self.assertTrue(at.session_state["m4_trace"].found)
        at.button(key="m4_btn_back").click().run(); ok(self, at)
        at.button(key="m4_btn_random").click().run(); ok(self, at)
        self.assertIsNone(at.session_state["m4_trace"])
        at.button(key="m4_btn_exec").click().run(); ok(self, at)
        at.button(key="m4_btn_clear").click().run(); ok(self, at)
        self.assertEqual(len(at.session_state["m4_grid"].obstacles), 0)
        at.button(key="m4_btn_restore").click().run(); ok(self, at)
        self.assertGreater(len(at.session_state["m4_grid"].obstacles), 0)

    def test_animation_and_coordinates(self):
        at = new_app(3)
        at.session_state["m4_speed"] = 0.1
        at.slider(key="m4_k").set_value(3).run()
        at.button(key="m4_btn_anim").click().run(); ok(self, at)
        tr = at.session_state["m4_trace"]
        self.assertEqual(at.session_state["m4_step"], len(tr.steps) - 1)
        self.assertFalse(at.session_state["m4_playing"])
        at.number_input(key="m4_er").set_value(0).run()
        at.number_input(key="m4_ec").set_value(0).run()
        at.button(key="m4_xy_obs").click().run(); ok(self, at)
        self.assertIn((0, 0), at.session_state["m4_grid"].obstacles)
        at.button(key="m4_xy_start").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m4_grid"].start, (0, 0))

    def test_unreachable_goal_message(self):
        at = new_app(3)
        from algorithms.grid import Grid
        at.session_state["m4_grid"] = Grid(6, 6, (0, 0), (5, 5), {(4, 5), (5, 4), (4, 4)})
        at.session_state["m4_rows"], at.session_state["m4_cols"] = 6, 6
        at.button(key="m4_btn_exec").click().run(); ok(self, at)
        self.assertFalse(at.session_state["m4_trace"].found)
        self.assertFalse(at.session_state["m4_bfs"].found)
        self.assertTrue(any("unreachable" in m.value.lower() for m in at.markdown))


class TestModule5And6And7(unittest.TestCase):
    def test_text_generation(self):
        at = new_app(4)
        ok(self, at)
        for _ in range(3):
            at.button(key="m5_next").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m5_step"], 3)
        at.slider(key="m5_k").set_value(1).run(); ok(self, at)
        self.assertEqual(at.session_state["m5_step"], 0)
        at.session_state["m5_speed"] = 0.3
        at.button(key="m5_play").click().run(); ok(self, at)
        self.assertFalse(at.session_state["m5_playing"])
        at.slider(key="m5_alpha").set_value(0.5).run(); ok(self, at)
        at.selectbox(key="m5_prompt").set_value("Artificial Intelligence can").run(); ok(self, at)

    def test_comparison_and_theory(self):
        at = new_app(5); ok(self, at)
        at.select_slider(key="m6_k").set_value(1).run(); ok(self, at)
        at.selectbox(key="m6_scenario").set_value("Random tree (seed below)").run(); ok(self, at)
        at.select_slider(key="m6_mk").set_value(1).run(); ok(self, at)
        at = new_app(6); ok(self, at)
        at.slider(key="m7_b").set_value(10).run(); ok(self, at)
        at.slider(key="m7_d").set_value(12).run(); ok(self, at)
        at.slider(key="m7_k").set_value(50).run(); ok(self, at)


class TestModule8(unittest.TestCase):
    def test_full_quiz(self):
        at = new_app(7)
        ok(self, at)
        n = len(at.session_state["m8_questions"])
        for i in range(n):
            q = at.session_state["m8_questions"][i]
            key = f"m8_in_{at.session_state['m8_seed']}_{i}"
            if q.multi:
                at.multiselect(key=key).set_value(list(q.answer)).run()
            else:
                at.radio(key=key).set_value(q.answer[0]).run()
            at.button(key="m8_submit").click().run(); ok(self, at)
            self.assertTrue(at.session_state["m8_results"][i][1], q.prompt)
            at.button(key="m8_next").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m8_idx"], n)
        self.assertTrue(any("Final performance" in m.value for m in at.markdown))
        at.button(key="m8_retake").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m8_idx"], 0)

    def test_submit_requires_answer_and_wrong_answer_marked(self):
        at = new_app(7)
        at.button(key="m8_submit").click().run(); ok(self, at)
        self.assertEqual(len(at.session_state["m8_results"]), 0)
        q = at.session_state["m8_questions"][0]
        key = f"m8_in_{at.session_state['m8_seed']}_0"
        wrong = [o for o in q.options if o not in q.answer]
        if q.multi:
            at.multiselect(key=key).set_value(wrong[:1]).run()
        else:
            at.radio(key=key).set_value(wrong[0]).run()
        at.button(key="m8_submit").click().run(); ok(self, at)
        self.assertFalse(at.session_state["m8_results"][0][1])


class TestSidebar(unittest.TestCase):
    def test_reset_all(self):
        at = new_app(1)
        at.button(key="m2_btn_start").click().run()
        at.button(key="m2_btn_next").click().run()
        self.assertEqual(at.session_state["m2_step"], 1)
        at.session_state["m2_k"] = 5
        at.button(key="reset_all").click().run(); ok(self, at)
        self.assertEqual(at.session_state["m2_step"], 0)           # progress forgotten
        self.assertFalse(at.session_state["m2_started"])
        self.assertEqual(at.session_state["m2_k"], 2)               # settings back to defaults


if __name__ == "__main__":
    unittest.main()

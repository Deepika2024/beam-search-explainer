"""Lightweight checks for the additive 3D maze visualization (no Streamlit needed)."""
import unittest

from algorithms.beam_search import beam_search_grid
from algorithms.grid import Grid
from visualizations.maze_3d_visualizer import maze_3d_figure


class Test3DMazeVisualization(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(5, 6, (0, 0), (4, 5), {(1, 2), (2, 2), (3, 2)})

    def test_existing_grid_keeps_four_direction_neighbors(self):
        self.assertEqual(self.grid.neighbors((2, 1)), [(1, 1), (3, 1), (2, 0)])

    def test_3d_figure_is_generated_from_real_beam_trace(self):
        result = beam_search_grid(self.grid, 5)
        snapshot = result.steps[-1]
        figure = maze_3d_figure(
            self.grid, explored=snapshot.explored, beam=snapshot.beam,
            pruned=snapshot.pruned_all, path=snapshot.path,
        )
        self.assertGreater(len(figure.data), 0)
        self.assertIn("3D Four-Direction", figure.layout.title.text)
        self.assertGreater(len(figure.to_json()), 1000)


if __name__ == "__main__":
    unittest.main()

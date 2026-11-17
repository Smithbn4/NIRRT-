"""Headless checks for custom maps, solver comparison, and measured RRT*."""
import unittest
from unittest.mock import patch

import numpy as np

from datasets.grid_map import GridMap
from path_planning_classes.rrt_star_2d import RRTStar2D
from path_planning_utils.Astar_with_clearance import AStar
from path_planning_utils.custom_map import MapSpec, Obstacle, integer


class CustomMapTests(unittest.TestCase):
    def test_rectangular_map_shape_placement_and_endpoint_validation(self):
        spec = MapSpec(35, 20, (1, 1), (33, 18),
                       (Obstacle('Rectangle', 6, 4, 7, 3), Obstacle('Circle', 24, 10, 3)))
        grid = spec.build_grid()
        self.assertEqual(grid.binary_mask.shape, (20, 35))
        self.assertTrue(np.all(grid.binary_mask[4:7, 6:13] == 0))
        self.assertTrue(grid.is_valid((13, 4)))
        self.assertFalse(grid.is_valid((27, 10)))
        self.assertTrue(grid.is_valid((27, 11)))
        with self.assertRaises(ValueError):
            MapSpec(35, 20, (6, 4), (33, 18), spec.obstacles).build_grid()

    def test_limits_dimensions_and_nonfinite_inputs(self):
        for value in ('2.5', '-1', 'nan', 'inf', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                integer(value, 'value')
        with self.assertRaises(ValueError):
            MapSpec(1, 10, (0, 0), (0, 1)).build_grid()
        obstacle = Obstacle('Rectangle', 3, 3, 2, 2)
        MapSpec(10, 10, (0, 0), (9, 9), (obstacle,)*10).build_grid()
        with self.assertRaises(ValueError):
            MapSpec(10, 10, (0, 0), (9, 9), (obstacle,)*11).build_grid()
        with self.assertRaises(ValueError):
            Obstacle('Circle', 2, 2, 3).validate(10, 10)
        with self.assertRaises(ValueError):
            Obstacle('Rectangle', 9, 9, 2, 1).validate(10, 10)
        with self.assertRaises(ValueError):
            MapSpec(10, 10, (0, 0), (8, 8), clearance=1).build_grid()


class ComparisonTests(unittest.TestCase):
    def astar(self, grid, start, goal, **kwargs):
        planner = AStar(start, goal, grid.valid_mask, 0, 'euclidean')
        path, _ = planner.searching(**kwargs)
        return planner, path[::-1]

    def test_astar_straight_diagonal_and_identical_endpoints(self):
        grid = GridMap(np.ones((10, 15)))
        planner, path = self.astar(grid, (1, 1), (8, 8))
        self.assertTrue(planner.check_success(path))
        self.assertAlmostEqual(RRTStar2D.get_path_len(path), 7*np.sqrt(2))
        self.assertGreaterEqual(planner.planning_seconds, 0)
        _, path = self.astar(grid, (2, 2), (2, 2))
        self.assertEqual(path, [(2, 2)])
        self.assertEqual(RRTStar2D.get_path_len(path), 0)

    def test_unreachable_and_diagonal_corner_return_empty_paths(self):
        mask = np.ones((12, 12))
        mask[:, 6] = 0
        planner, path = self.astar(GridMap(mask), (2, 5), (9, 5))
        self.assertEqual(path, [])
        self.assertFalse(planner.check_success(path))
        mask = np.ones((2, 2))
        mask[0, 1] = mask[1, 0] = 0
        _, path = self.astar(GridMap(mask), (0, 0), (1, 1))
        self.assertEqual(path, [])

    def test_both_solvers_respect_clearance_on_custom_map(self):
        np.random.seed(3)
        spec = MapSpec(35, 25, (2, 2), (32, 22),
                       (Obstacle('Rectangle', 14, 6, 3, 11),), clearance=1)
        grid = spec.build_grid()
        _, astar_path = self.astar(grid, spec.start, spec.goal)
        planner = RRTStar2D(spec.start, spec.goal, 7, 300, grid)
        rrt_path = planner.planning()
        for path in (astar_path, rrt_path):
            self.assertGreater(len(path), 0)
            self.assertTrue(np.array_equal(path[0], spec.start))
            self.assertTrue(np.array_equal(path[-1], spec.goal))
            self.assertTrue(all(not grid.is_collision(a, b) for a, b in zip(path[:-1], path[1:])))
        history = planner.convergence_history
        self.assertGreater(len(history), 0)
        self.assertEqual(planner.first_solution_seconds, history[0][1])
        self.assertEqual(planner.best_solution_seconds, history[-1][1])
        self.assertLessEqual(planner.best_solution_seconds, planner.planning_seconds)
        self.assertAlmostEqual(history[-1][2], planner.get_path_len(rrt_path))
        self.assertTrue(all(a[2] > b[2] for a, b in zip(history[:-1], history[1:])))
        self.assertTrue(all(a[1] <= b[1] for a, b in zip(history[:-1], history[1:])))

    def test_clock_values_direct_solution_and_single_use(self):
        planner = RRTStar2D((1, 1), (3, 1), 5, 20, GridMap(np.ones((10, 10))))
        with patch('path_planning_classes.rrt_star_2d.time.perf_counter', side_effect=[100, 100.25, 100.5]):
            planner.planning(stop_on_first=True)
        self.assertEqual(planner.iterations_run, 0)
        self.assertEqual(planner.first_solution_seconds, 0.25)
        self.assertEqual(planner.best_solution_seconds, 0.25)
        self.assertEqual(planner.planning_seconds, 0.5)
        with self.assertRaises(RuntimeError):
            planner.planning()

    def test_rrt_failure_and_cancellation(self):
        mask = np.ones((12, 12))
        mask[:, 6] = 0
        planner = RRTStar2D((2, 5), (9, 5), 5, 10, GridMap(mask))
        self.assertEqual(len(planner.planning()), 0)
        self.assertIsNone(planner.first_solution_seconds)
        self.assertIsNone(planner.best_solution_seconds)
        self.assertEqual(planner.convergence_history, [])
        planner = RRTStar2D((2, 5), (9, 5), 5, 10, GridMap(mask))
        planner.planning(cancel_check=lambda: True)
        self.assertTrue(planner.cancelled)
        self.assertEqual(planner.iterations_run, 0)
        astar, path = self.astar(GridMap(mask), (2, 5), (9, 5), cancel_check=lambda: True)
        self.assertTrue(astar.cancelled)
        self.assertEqual(path, [])


if __name__ == '__main__':
    unittest.main()

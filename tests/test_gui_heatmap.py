"""Probability-scale and worker integration checks without a desktop display."""
import queue
import threading
import unittest

import numpy as np

from gui import PlanningApp
from path_planning_classes.nrrt_star_gng_2d import NRRTStarGNG2D
from path_planning_utils.custom_map import MapSpec, Obstacle
from path_planning_utils.heatmap import probability_rgb
from plan import prepare_planner, run_planner


class HeatmapTests(unittest.TestCase):
    def test_probability_colors_use_absolute_scale(self):
        together = probability_rgb(np.array([0, 0.5, 1]))
        np.testing.assert_array_equal(probability_rgb(np.full((2, 3), 0.5)),
                                      np.broadcast_to(together[1], (2, 3, 3)))
        self.assertEqual(together.dtype, np.uint8)
        self.assertFalse(np.array_equal(together[0], together[2]))
        np.testing.assert_array_equal(probability_rgb([0, 0.5, 1], opacity=0),
                                      np.full((3, 3), 255))

    def test_invalid_probabilities_are_rejected(self):
        for scores in ([float('nan')], [float('inf')], [-0.1], [1.1]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                probability_rgb(scores)

    def worker(self, network, rate=0.5):
        app = PlanningApp.__new__(PlanningApp)
        app.events = queue.Queue()
        app.cancel_event = threading.Event()
        app._network = network
        spec = MapSpec(35, 19, (1, 1), (33, 17))
        app._worker(spec, 40, 8, 100, False, rate)
        events = []
        while not app.events.empty():
            events.append(app.events.get_nowait())
        self.assertEqual(events[-1][0], 'done')
        return spec, events

    def test_gui_uses_nrrt_guidance_and_overlays_the_same_inference(self):
        coordinates = np.array([(x, 1) for x in range(1, 34)]+[(33, y) for y in range(2, 18)])
        scores = np.zeros((19, 35), dtype=np.float32)
        scores[coordinates[:, 1], coordinates[:, 0]] = 0.9

        class Network:
            calls = 0

            def classify_path_points(self, mask, start, goal):
                self.calls += 1
                self.arguments = mask.copy(), tuple(start), tuple(goal)
                return coordinates, scores > 0.5, scores

        network = Network()
        spec, events = self.worker(network, rate=1)
        self.assertFalse([data for event, data in events if event == 'error'])
        self.assertEqual(network.calls, 1)
        np.testing.assert_array_equal(network.arguments[0], spec.build_grid().binary_mask)
        self.assertEqual(network.arguments[1:], (spec.start, spec.goal))
        guidance = next(data for event, data in events if event == 'guidance')
        self.assertIs(guidance['scores'], scores)
        self.assertGreaterEqual(guidance['seconds'], 0)
        self.assertEqual(guidance['points'], len(coordinates))
        results = dict(data for event, data in events if event == 'result')
        self.assertEqual(set(results), {'nrrt', 'astar'})
        self.assertEqual(results['nrrt']['neural_samples'], 40)
        self.assertEqual(results['nrrt']['uniform_samples'], 0)
        self.assertEqual(results['nrrt']['guidance_points'], len(coordinates))
        self.assertAlmostEqual(results['nrrt']['total_seconds'],
                               results['nrrt']['setup_seconds']+results['nrrt']['seconds'])
        self.assertAlmostEqual(results['nrrt']['first_with_setup'],
                               results['nrrt']['setup_seconds']+results['nrrt']['first'])
        # The GUI run is identical to this project's NRRT* with the same query.
        np.random.seed(100)
        baseline = NRRTStarGNG2D(spec.start, spec.goal, 8, 40, spec.build_grid(), network, 1)
        np.testing.assert_array_equal(results['nrrt']['path'], baseline.planning())

    def test_failed_neural_output_does_not_masquerade_as_nrrt(self):
        class BrokenNetwork:
            def classify_path_points(self, *args):
                raise RuntimeError('test inference failure')

        class MisalignedNetwork:
            def classify_path_points(self, *args):
                return None, None, np.zeros((3, 3))

        for network in (BrokenNetwork(), MisalignedNetwork()):
            with self.subTest(network=type(network).__name__):
                _, events = self.worker(network)
                self.assertTrue(any(event == 'error' for event, _ in events))
                self.assertFalse(any(event == 'guidance' for event, _ in events))
                results = dict(data for event, data in events if event == 'result')
                self.assertEqual(set(results), {'astar'})
                self.assertTrue(len(results['astar']['path']))

    def test_shared_pipeline_filters_occupied_and_clearance_guidance(self):
        class Network:
            def classify_path_points(self, mask, start, goal):
                coordinates = np.array([(0, 1), (9, 7), (10, 7), (5, 5), (20, 10)])
                scores = np.zeros_like(mask)
                scores[coordinates[:, 1], coordinates[:, 0]] = 0.9
                return coordinates, scores > 0.5, scores

        spec = MapSpec(35, 19, (2, 2), (32, 16),
                       (Obstacle('Rectangle', 10, 7, 3, 4),), clearance=1)
        setup = prepare_planner(spec.build_grid(), spec.start, spec.goal, wrapper=Network(),
                                neural_sample_rate=1, device='cpu', iterations=20)
        self.assertIsInstance(setup.planner, NRRTStarGNG2D)
        expected = {(5., 5.), (20., 10.)}
        self.assertEqual(set(map(tuple, setup.planner.guidance)), expected)
        for _ in range(20):
            self.assertIn(tuple(setup.planner.generate_random_node()), expected)
        self.assertEqual(setup.planner.neural_samples, 20)
        self.assertEqual(setup.planner.uniform_samples, 0)

    def test_empty_corridor_uses_the_existing_nrrt_fallback(self):
        class Network:
            def classify_path_points(self, mask, start, goal):
                return np.empty((0, 2)), np.zeros_like(mask), np.zeros_like(mask)

        _, events = self.worker(Network(), rate=1)
        self.assertFalse([data for event, data in events if event == 'error'])
        guidance = next(data for event, data in events if event == 'guidance')
        self.assertEqual(guidance['points'], 0)
        nrrt = dict(data for event, data in events if event == 'result')['nrrt']
        self.assertEqual(nrrt['neural_samples'], 0)
        self.assertEqual(nrrt['uniform_samples'], 40)

    def test_shared_runner_cancellation_preserves_metrics(self):
        spec = MapSpec(35, 19, (1, 1), (33, 17))
        setup = prepare_planner(spec.build_grid(), spec.start, spec.goal,
                                neural_sample_rate=0, device='cpu', iterations=20)
        path = run_planner(setup.planner, cancel_check=lambda: True)
        self.assertEqual(len(path), 0)
        self.assertTrue(setup.planner.cancelled)
        self.assertEqual(setup.planner.neural_samples+setup.planner.uniform_samples, 0)


if __name__ == '__main__':
    unittest.main()

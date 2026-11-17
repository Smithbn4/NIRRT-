"""Behavioral checks for custom-grid planning and training integration."""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets.grid_map import GridMap, network_input
from path_planning_classes.nrrt_star_gng_2d import NRRTStarGNG2D
from path_planning_classes.rrt_star_2d import RRTStar2D
from unet.image_dataset import ImageDataset
from unet.training import run_epoch


class GridCollisionTests(unittest.TestCase):
    def test_thin_wall_and_diagonal_corner(self):
        mask = np.ones((12, 12))
        mask[:, 6] = 0
        grid = GridMap(mask)
        self.assertTrue(grid.is_collision((1, 5), (10, 5)))
        self.assertTrue(grid.is_collision((1.2, 2.3), (10.1, 8.2)))
        mask = np.ones((4, 4))
        mask[0, 1] = 0
        self.assertTrue(GridMap(mask).is_collision((0, 0), (1, 1)))

    def test_boundary_aligned_line_touches_both_rows(self):
        mask = np.ones((8, 8))
        mask[2, 4] = 0
        self.assertTrue(GridMap(mask).is_collision((1, 2.5), (6, 2.5)))

    def test_endpoint_touching_obstacle_is_blocked_in_both_directions(self):
        mask = np.ones((8, 8))
        mask[3, 2] = 0
        grid = GridMap(mask)
        self.assertTrue(grid.is_collision((6, 3), (2.5, 3)))
        self.assertTrue(grid.is_collision((2.5, 3), (6, 3)))

    def test_clearance_and_map_boundary(self):
        mask = np.ones((10, 10))
        mask[5, 5] = 0
        grid = GridMap(mask, clearance=1)
        self.assertFalse(grid.is_valid((4, 4)))
        self.assertFalse(grid.is_valid((0, 5)))
        self.assertFalse(grid.is_valid((-0.1, 4)))
        self.assertTrue(grid.is_valid((2, 2)))

    def test_blocked_final_connection_is_not_a_solution(self):
        mask = np.ones((10, 10))
        mask[:, 5] = 0
        planner = RRTStar2D((3, 5), (7, 5), 8, 1, GridMap(mask))
        self.assertIsNone(planner.search_goal_parent())
        self.assertEqual(len(planner.planning()), 0)

    def test_empty_neural_guidance_falls_back(self):
        class EmptyNetwork:
            def classify_path_points(self, mask, start, goal):
                return np.empty((0, 2), dtype=int), np.zeros_like(mask), np.zeros_like(mask)
        np.random.seed(0)
        planner = NRRTStarGNG2D((1, 1), (18, 18), 5, 150, GridMap(np.ones((20, 20))), EmptyNetwork(), 1)
        path = planner.planning()
        self.assertGreater(len(path), 0)
        self.assertTrue(all(not planner.env.is_collision(a, b) for a, b in zip(path[:-1], path[1:])))

    def test_edge_marker_does_not_wrap(self):
        inputs = network_input(np.ones((7, 9)), (0, 0), (8, 6))
        self.assertEqual(inputs[0].sum(), 4)
        self.assertEqual(inputs[1].sum(), 4)
        self.assertEqual(inputs[0, -1, -1], 0)


class DataAndTrainingTests(unittest.TestCase):
    def test_lazy_non_square_dataset_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'train'
            (root/'env_imgs').mkdir(parents=True)
            (root/'astar_paths').mkdir()
            (root/'envs.json').write_text(json.dumps([dict(env_dims=[17, 35], start=[[0, 0]], goal=[[34, 16]])]))
            cv2.imwrite(str(root/'env_imgs/0.png'), np.full((17, 35), 255, np.uint8))
            np.savetxt(str(root/'astar_paths/0_0.txt'), [[0, 0], [34, 16]], delimiter=',')
            dataset = ImageDataset(directory)
            inputs, labels, token = dataset[0]
            self.assertEqual(tuple(inputs.shape), (3, 32, 64))
            self.assertEqual(tuple(labels.shape), (32, 64))
            self.assertEqual(token, '0_0')
            self.assertTrue(torch.all(labels[17:, :] == -100))
            self.assertTrue(torch.all(labels[:, 35:] == -100))
            self.assertEqual(int((labels == 1).sum()), 8)

    def test_partial_batch_metrics_ignore_padding(self):
        class PerfectModel(torch.nn.Module):
            def forward(self, x):
                return torch.cat([-x, x], dim=1)
        inputs = torch.ones((3, 1, 4, 4))
        labels = torch.ones((3, 4, 4), dtype=torch.long)
        labels[:, -1] = -100
        dataset = TensorDataset(inputs, labels, torch.arange(3))
        metrics = run_epoch(PerfectModel(), DataLoader(dataset, batch_size=2),
                            torch.device('cpu'), torch.ones(2))
        self.assertEqual(metrics['samples'], 3)
        self.assertEqual(metrics['path_iou'], 1.0)
        self.assertTrue(np.isfinite(metrics['loss']))


if __name__ == '__main__':
    unittest.main()

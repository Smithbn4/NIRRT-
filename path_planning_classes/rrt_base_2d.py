"""Tree storage and geometric helpers adapted from the NIRRT* RRTBase2D."""
import numpy as np


class RRTBase2D:
    def __init__(self, x_start, x_goal, step_len, iter_max, env):
        if step_len <= 0 or iter_max < 1:
            raise ValueError('step length and iteration budget must be positive')
        self.x_start = np.asarray(x_start, dtype=float)
        self.x_goal = np.asarray(x_goal, dtype=float)
        env.validate_endpoint(self.x_start, 'Start')
        env.validate_endpoint(self.x_goal, 'Goal')
        self.step_len, self.iter_max, self.env = step_len, iter_max, env
        self.vertices = np.zeros((iter_max+1, 2), dtype=float)
        self.vertex_parents = np.zeros(iter_max+1, dtype=int)
        self.vertices[0] = self.x_start
        self.num_vertices = 1
        self.path = np.empty((0, 2))
        self.iterations_run = 0

    def cost(self, index):
        total = 0.0
        while index != 0:
            parent = self.vertex_parents[index]
            total += np.linalg.norm(self.vertices[index]-self.vertices[parent])
            index = parent
        return float(total)

    def extract_path(self, index):
        path = [self.x_goal]
        while index != 0:
            path.append(self.vertices[index].copy())
            index = self.vertex_parents[index]
        path.append(self.x_start)
        return np.asarray(path[::-1])

    @staticmethod
    def get_path_len(path):
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) else float('inf')


"""RRT* expansion, parent selection, and rewiring adapted from NIRRT*."""
import math
import time
import numpy as np
from path_planning_classes.rrt_base_2d import RRTBase2D


class RRTStar2D(RRTBase2D):
    def __init__(self, x_start, x_goal, step_len, iter_max, env):
        super().__init__(x_start, x_goal, step_len, iter_max, env)
        self.search_radius = math.ceil(math.sqrt(3*env.valid_mask.sum()/math.pi))
        self.goal_candidates = set()
        if self.in_goal_region(self.x_start):
            self.goal_candidates.add(0)
        self.first_solution_seconds = None
        self.best_solution_seconds = None
        self.planning_seconds = 0.0
        self.convergence_history = []  # (iteration, elapsed seconds, best length)
        self.cancelled = False
        self._planning_started = False

    def in_goal_region(self, point):
        return np.linalg.norm(point-self.x_goal) <= self.step_len and not self.env.is_collision(point, self.x_goal)

    def generate_random_node(self):
        return self.env.sample_free()

    def expand(self, target):
        distances = np.linalg.norm(self.vertices[:self.num_vertices]-target, axis=1)
        nearest = int(np.argmin(distances))
        offset = target-self.vertices[nearest]
        distance = np.linalg.norm(offset)
        if distance < 1e-8:
            return
        point = self.vertices[nearest]+offset*min(1, self.step_len/distance)
        if self.env.is_collision(self.vertices[nearest], point):
            return
        index = self.num_vertices
        radius = min(self.step_len, self.search_radius*math.sqrt(math.log(index+1)/(index+1)))
        lengths = np.linalg.norm(self.vertices[:index]-point, axis=1)
        neighbors = [int(i) for i in np.flatnonzero(lengths <= radius)
                     if not self.env.is_collision(self.vertices[i], point)]
        parent = nearest
        best_cost = self.cost(nearest)+np.linalg.norm(point-self.vertices[nearest])
        for candidate in neighbors:
            candidate_cost = self.cost(candidate)+lengths[candidate]
            if candidate_cost < best_cost:
                best_cost, parent = candidate_cost, candidate
        self.vertices[index], self.vertex_parents[index] = point, parent
        self.num_vertices += 1
        for neighbor in neighbors:
            if self.cost(neighbor) > best_cost+lengths[neighbor]+1e-10:
                self.vertex_parents[neighbor] = index
        if self.in_goal_region(point):
            self.goal_candidates.add(index)

    def search_goal_parent(self):
        # Only collision-checked candidates can be selected, including the root.
        if not self.goal_candidates:
            return None
        return min(self.goal_candidates, key=lambda i: self.cost(i)+np.linalg.norm(self.vertices[i]-self.x_goal))

    def planning(self, stop_on_first=False, progress_callback=None, cancel_check=None):
        """Measure first solution and every strict best-path improvement.

        Times cover search and final extraction, excluding construction/map setup.
        The last improvement is an observed value, not proof of convergence.
        Callbacks run in the caller's thread; progress is throttled to 10 Hz.
        """
        if self._planning_started:
            raise RuntimeError('Create a fresh planner for each planning run')
        self._planning_started = True
        started = time.perf_counter()
        last_progress = started
        best_length = float('inf')

        def record_solution(iteration):
            nonlocal best_length
            parent = self.search_goal_parent()
            if parent is None:
                return
            length = self.cost(parent)+np.linalg.norm(self.vertices[parent]-self.x_goal)
            if length < best_length-1e-10:
                elapsed = time.perf_counter()-started
                if self.first_solution_seconds is None:
                    self.first_solution_seconds = elapsed
                self.best_solution_seconds = elapsed
                best_length = float(length)
                self.convergence_history.append((iteration, elapsed, best_length))

        record_solution(0)
        for iteration in range(self.iter_max):
            if cancel_check and cancel_check():
                self.cancelled = True
                break
            if stop_on_first and self.goal_candidates:
                break
            self.expand(self.generate_random_node())
            self.iterations_run = iteration+1
            record_solution(self.iterations_run)
            now = time.perf_counter()
            if progress_callback and now-last_progress >= 0.1:
                progress_callback(self)
                last_progress = now
        parent = self.search_goal_parent()
        self.path = np.empty((0, 2)) if parent is None else self.extract_path(parent)
        self.planning_seconds = time.perf_counter()-started
        if progress_callback:
            progress_callback(self)
        return self.path


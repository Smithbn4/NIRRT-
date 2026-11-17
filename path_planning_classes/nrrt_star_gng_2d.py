"""2D ResNet50/U-Net grid-guided NRRT*, adapted from NRRTStarGNG2D."""
import numpy as np
from path_planning_classes.rrt_star_2d import RRTStar2D


class NRRTStarGNG2D(RRTStar2D):
    def __init__(self, x_start, x_goal, step_len, iter_max, env, gng_wrapper,
                 neural_sample_rate=0.5):
        super().__init__(x_start, x_goal, step_len, iter_max, env)
        if not 0 <= neural_sample_rate <= 1:
            raise ValueError('neural sample rate must be between 0 and 1')
        self.neural_sample_rate = neural_sample_rate
        self.guidance = np.empty((0, 2))
        self.path_score = None
        self.neural_samples = 0
        self.uniform_samples = 0
        if neural_sample_rate:
            if gng_wrapper is None:
                raise ValueError('NRRT* requires a neural wrapper when the sample rate is positive')
            coordinates, _, self.path_score = gng_wrapper.classify_path_points(
                env.binary_mask, np.asarray(x_start, int), np.asarray(x_goal, int))
            self.path_score = np.asarray(self.path_score)
            if self.path_score.shape != env.binary_mask.shape:
                raise ValueError('Network heatmap dimensions do not match the map')
            if not np.isfinite(self.path_score).all() or np.any((self.path_score < 0) | (self.path_score > 1)):
                raise ValueError('Network returned invalid probabilities')
            coordinates = np.asarray(coordinates)
            if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or
                    not np.isfinite(coordinates).all() or
                    np.any(coordinates != np.floor(coordinates)) or
                    np.any(coordinates < 0) or
                    np.any(coordinates[:, 0] >= env.width) or
                    np.any(coordinates[:, 1] >= env.height)):
                raise ValueError('Network returned invalid guidance coordinates')
            coordinates = coordinates.astype(int)
            if len(coordinates):
                # Neural positives still must be free at the requested clearance.
                valid = env.valid_mask[coordinates[:, 1], coordinates[:, 0]]
                self.guidance = coordinates[valid].astype(float)

    def generate_random_node(self):
        if len(self.guidance) and np.random.random() < self.neural_sample_rate:
            self.neural_samples += 1
            return self.guidance[np.random.randint(len(self.guidance))]
        self.uniform_samples += 1
        return self.env.sample_free()


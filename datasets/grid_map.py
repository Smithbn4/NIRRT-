"""Occupancy input, clearance, and conservative grid-edge collision checking.

Coordinates are (x, y) pixel centers; arrays use [y, x]. The origin is top-left.
"""
from pathlib import Path
import math
import cv2
import numpy as np


def load_binary_map(filename, threshold=127, invert=False):
    if not 0 <= threshold <= 255:
        raise ValueError('threshold must be between 0 and 255')
    image = cv2.imread(str(Path(filename).expanduser()), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('Cannot read map image: {}'.format(filename))
    free = image > threshold
    return (~free if invert else free).astype(np.float32)


def stamp_square(array, point, radius=1, value=1):
    """Stamp a marker, clipping at image boundaries instead of wrapping slices."""
    x, y = map(int, point)
    height, width = array.shape[-2:]
    array[..., max(0, y-radius):min(height, y+radius+1),
          max(0, x-radius):min(width, x+radius+1)] = value


def network_input(binary_mask, start, goal):
    free = np.asarray(binary_mask, dtype=np.float32)
    raw = np.zeros((3,) + free.shape, dtype=np.float32)
    stamp_square(raw[0], start)
    stamp_square(raw[1], goal)
    raw[2] = free
    return raw


def pad_to_32(array, fill=0):
    height, width = array.shape[-2:]
    padding = [(0, 0)] * (array.ndim - 2)
    padding += [(0, (-height) % 32), (0, (-width) % 32)]
    return np.pad(array, padding, constant_values=fill)


def normalize_input(raw):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (raw - mean) / std


class GridMap:
    def __init__(self, binary_mask, clearance=0):
        free = np.asarray(binary_mask)
        if free.ndim != 2 or min(free.shape) < 2:
            raise ValueError('Map must be a 2D array at least 2 by 2 pixels')
        if not np.isfinite(free).all() or not np.isin(free, [0, 1]).all():
            raise ValueError('Map values must be binary: 0 obstacle, 1 free')
        if int(clearance) != clearance or clearance < 0:
            raise ValueError('clearance must be a nonnegative integer in pixels')
        self.binary_mask = free.astype(np.float32)
        self.height, self.width = free.shape
        self.clearance = int(clearance)
        blocked = (free == 0).astype(np.uint8)
        if clearance:
            kernel = np.ones((2*self.clearance+1, 2*self.clearance+1), np.uint8)
            blocked = cv2.dilate(blocked, kernel, borderType=cv2.BORDER_CONSTANT,
                                 borderValue=1)
        self.valid_mask = blocked == 0
        self.free_pixels = np.flip(np.argwhere(self.valid_mask), axis=1)
        if not len(self.free_pixels):
            raise ValueError('No free pixels remain after applying clearance')
        self.x_range = (0, self.width-1)
        self.y_range = (0, self.height-1)

    def is_valid(self, point):
        x, y = np.asarray(point, dtype=float)
        if not np.isfinite([x, y]).all():
            return False
        if not (0 <= x <= self.width-1 and 0 <= y <= self.height-1):
            return False
        columns = [int(math.floor(x+0.5))]
        rows = [int(math.floor(y+0.5))]
        if abs((x+0.5)-round(x+0.5)) < 1e-12:
            columns.append(columns[0]-1)
        if abs((y+0.5)-round(y+0.5)) < 1e-12:
            rows.append(rows[0]-1)
        return all(self.valid_mask[row, column] for row in rows for column in columns)

    def validate_endpoint(self, point, name):
        if not self.is_valid(point):
            raise ValueError('{} {} is outside the map, blocked, or violates clearance'.format(name, tuple(point)))

    @staticmethod
    def segment_cells(start, end):
        """Supercover traversal: include both side cells when crossing a corner."""
        start, end = np.asarray(start, float)+0.5, np.asarray(end, float)+0.5
        cell, last = np.floor(start).astype(int), np.floor(end).astype(int)
        delta = end-start
        step = np.sign(delta).astype(int)
        t_delta = np.full(2, np.inf)
        t_max = np.full(2, np.inf)
        for axis in range(2):
            if step[axis]:
                t_delta[axis] = 1/abs(delta[axis])
                edge = cell[axis]+(1 if step[axis] > 0 else 0)
                t_max[axis] = (edge-start[axis])/delta[axis]
        yield tuple(cell)
        while not np.array_equal(cell, last):
            if abs(t_max[0]-t_max[1]) < 1e-12:
                yield (cell[0]+step[0], cell[1])
                yield (cell[0], cell[1]+step[1])
                cell += step
                t_max += t_delta
            else:
                axis = int(np.argmin(t_max))
                cell[axis] += step[axis]
                t_max[axis] += t_delta[axis]
            yield tuple(cell)

    def is_collision(self, start, end):
        if not self.is_valid(start) or not self.is_valid(end):
            return True
        start, end = np.asarray(start, float), np.asarray(end, float)
        # A line exactly on a cell boundary touches the cells on both sides.
        sides = [(0, 0)]
        for axis in range(2):
            shifted = start[axis]+0.5
            if abs(end[axis]-start[axis]) < 1e-12 and abs(shifted-round(shifted)) < 1e-12:
                offset = (-1, 0) if axis == 0 else (0, -1)
                sides += [(a+offset[0], b+offset[1]) for a, b in list(sides)]
        for x, y in self.segment_cells(start, end):
            for dx, dy in sides:
                xx, yy = x+dx, y+dy
                if not (0 <= xx < self.width and 0 <= yy < self.height and self.valid_mask[yy, xx]):
                    return True
        return False

    def sample_free(self):
        return self.free_pixels[np.random.randint(len(self.free_pixels))].astype(float)


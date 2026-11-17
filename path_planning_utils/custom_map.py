"""Validated custom-map definitions shared by the GUI and headless checks.

All dimensions and coordinates are integer pixels. Rectangles occupy exactly
length * width cells from their top-left corner; circles include cell centers
whose distance from the supplied center is at most their radius.
"""
from dataclasses import dataclass
import numpy as np

from datasets.grid_map import GridMap


def integer(value, name, minimum=0):
    try:
        number = int(value)
        if float(value) != number or number < minimum:
            raise ValueError()
    except (TypeError, ValueError, OverflowError):
        raise ValueError('{} must be an integer >= {}'.format(name, minimum)) from None
    return number


@dataclass(frozen=True)
class Obstacle:
    kind: str
    x: int
    y: int
    size: int  # rectangle length along x, or circle radius
    width: int = 1  # rectangle width along y

    def validate(self, map_width, map_height):
        if self.kind not in ('Rectangle', 'Circle'):
            raise ValueError('Choose Rectangle or Circle')
        x, y = integer(self.x, 'Obstacle x'), integer(self.y, 'Obstacle y')
        size = integer(self.size, 'Length / radius', 1)
        width = integer(self.width, 'Rectangle width', 1)
        if self.kind == 'Rectangle':
            fits = x+size <= map_width and y+width <= map_height
        else:
            fits = x-size >= 0 and y-size >= 0 and x+size < map_width and y+size < map_height
        if not fits:
            raise ValueError('Every obstacle must fit entirely inside the map')


@dataclass(frozen=True)
class MapSpec:
    width: int
    height: int
    start: tuple
    goal: tuple
    obstacles: tuple = ()
    clearance: int = 0

    def validate(self):
        integer(self.width, 'Map width', 2)
        integer(self.height, 'Map height', 2)
        integer(self.clearance, 'Clearance')
        for name, point in [('Start', self.start), ('Goal', self.goal)]:
            if len(point) != 2:
                raise ValueError('{} needs x and y coordinates'.format(name))
            x, y = integer(point[0], name+' x'), integer(point[1], name+' y')
            if x >= self.width or y >= self.height:
                raise ValueError('{} must be inside the map'.format(name))
        if len(self.obstacles) > 10:
            raise ValueError('A maximum of 10 obstacles is allowed')
        for obstacle in self.obstacles:
            obstacle.validate(self.width, self.height)

    def build_grid(self):
        self.validate()
        mask = np.ones((self.height, self.width), dtype=np.uint8)
        for obstacle in self.obstacles:
            x, y, size = obstacle.x, obstacle.y, obstacle.size
            if obstacle.kind == 'Rectangle':
                mask[y:y+obstacle.width, x:x+size] = 0
            else:
                yy, xx = np.ogrid[-size:size+1, -size:size+1]
                patch = mask[y-size:y+size+1, x-size:x+size+1]
                patch[xx.astype(float)**2+yy.astype(float)**2 <= float(size)**2] = 0
        grid = GridMap(mask, self.clearance)
        grid.validate_endpoint(self.start, 'Start')
        grid.validate_endpoint(self.goal, 'Goal')
        return grid

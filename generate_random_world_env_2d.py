"""Generate 2D rectangle/circle maps with A* path labels (adapted from NIRRT*)."""
import argparse
import json
import random
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
from path_planning_utils.Astar_with_clearance import generate_start_goal_points, AStar
from project_config import ROOT

def generate_env(
    img_height,
    img_width,
    rectangle_width_range,
    circle_radius_range,
    num_rectangles_range,
    num_circles_range,
):
    env_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    env_dims = (img_height, img_width)
    num_rectangles = random.randint(num_rectangles_range[0], num_rectangles_range[1])
    num_circles = random.randint(num_circles_range[0], num_circles_range[1])
    rectangle_obstacles = []
    circle_obstacles = []
    # Draw random black rectangles
    for i in range(num_rectangles):
        x = random.randint(0, img_width)
        y = random.randint(0, img_height)
        w = random.randint(rectangle_width_range[0], rectangle_width_range[1])
        h = random.randint(rectangle_width_range[0], rectangle_width_range[1])
        cv2.rectangle(env_img, (x, y), (x + w, y + h), (0, 0, 0), -1)
        rectangle_obstacles.append([x,y,w,h])
    # Draw random circular disks
    for i in range(num_circles):
        x = random.randint(0, img_width)
        y = random.randint(0, img_height)
        r = random.randint(circle_radius_range[0], circle_radius_range[1])
        cv2.circle(env_img, (x, y), r, (0, 0, 0), -1)
        circle_obstacles.append([x,y,r])
    
    binary_env = np.zeros(env_dims).astype(int)
    binary_env[env_img[:,:,0]!=0]=1

    return env_img, binary_env, env_dims, rectangle_obstacles, circle_obstacles

def generate_astar_path(
    binary_env,
    s_start,
    s_goal,
    clearance=3,
):
    start_time = time.time()
    astar = AStar(s_start, s_goal, binary_env, clearance, "euclidean")
    path, visited = astar.searching()
    path = astar.get_path_from_start_to_goal(path) # list of tuples
    path_success = astar.check_success(path)
    exec_time = time.time() - start_time
    if path_success:
        return path, exec_time
    else:
        return None, exec_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'env_configs/random_2d.yml')
    parser.add_argument('--output', type=Path, default=ROOT/'data/generated_random_2d')
    parser.add_argument('--train-envs', type=int)
    parser.add_argument('--val-envs', type=int)
    parser.add_argument('--test-envs', type=int)
    parser.add_argument('--samples-per-env', type=int)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--max-attempts-per-env', type=int, default=100)
    args = parser.parse_args()
    with args.config.expanduser().open() as stream:
        config = yaml.safe_load(stream)
    for mode in ['train', 'val', 'test']:
        override = getattr(args, mode+'_envs')
        if override is not None:
            config[mode+'_env_size'] = override
        if config[mode+'_env_size'] < 0:
            parser.error('map counts must be nonnegative')
    if args.samples_per_env is not None:
        config['num_samples_per_env'] = args.samples_per_env
    if config['num_samples_per_env'] < 1 or args.max_attempts_per_env < 1:
        parser.error('samples per map and maximum attempts must be positive')
    if config['path_clearance'] < 0 or min(config['env_height'], config['env_width']) <= 2*config['path_clearance']:
        parser.error('map size must exceed twice the nonnegative clearance')
    if min(config['env_height'], config['env_width'])-2*config['path_clearance'] <= config['start_goal_dim_distance_limit']:
        parser.error('endpoint distance limit is too large for this map size and clearance')
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'generation_config.yml').write_text(yaml.safe_dump(dict(config, random_seed=args.seed)))
    for mode in ['train', 'val', 'test']:
        split = output/mode
        (split/'env_imgs').mkdir(parents=True)
        (split/'astar_paths').mkdir()
        environments = []
        attempts = 0
        wanted = config[mode+'_env_size']
        while len(environments) < wanted:
            attempts += 1
            if attempts > max(1, wanted)*args.max_attempts_per_env:
                raise RuntimeError('Too many disconnected map attempts; reduce obstacles/clearance or endpoint distance')
            image, binary, dimensions, rectangles, circles = generate_env(
                config['env_height'], config['env_width'],
                config['rectangle_width_range'], config['circle_radius_range'],
                config['num_rectangles_range'], config['num_circles_range'])
            starts, goals, paths, timings = [], [], [], []
            for _ in range(config['num_samples_per_env']):
                start, goal = generate_start_goal_points(
                    binary, clearance=config['path_clearance'],
                    distance_lower_limit=config['start_goal_dim_distance_limit'],
                    max_attempt_count=config['start_goal_sampling_attempt_count'])
                if start is None:
                    break
                path, elapsed = generate_astar_path(binary, start, goal, config['path_clearance'])
                if path is None:
                    break
                starts.append(start)
                goals.append(goal)
                paths.append(path)
                timings.append(elapsed)
            if len(paths) != config['num_samples_per_env']:
                continue
            index = len(environments)
            environments.append(dict(env_dims=dimensions, rectangle_obstacles=rectangles,
                                     circle_obstacles=circles, start=starts, goal=goals,
                                     astar_time=timings))
            if not cv2.imwrite(str(split/'env_imgs'/'{}.png'.format(index)), image):
                raise OSError('Unable to save map image')
            for query, path in enumerate(paths):
                np.savetxt(str(split/'astar_paths'/'{}_{}.txt'.format(index, query)), path, fmt='%d', delimiter=',')
            (split/'envs.json').write_text(json.dumps(environments))
            print('{}: {}/{} maps'.format(mode, len(environments), wanted), flush=True)
        (split/'envs.json').write_text(json.dumps(environments))
    print('Dataset saved to {}'.format(output))


if __name__ == '__main__':
    main()

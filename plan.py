"""Solve a provided image map with 2D ResNet50/U-Net-guided NRRT*."""
import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
import uuid
import numpy as np
from datasets.grid_map import GridMap, load_binary_map
from path_planning_classes.nrrt_star_gng_2d import NRRTStarGNG2D
from project_config import ROOT, DEFAULT_DATA, DEFAULT_CHECKPOINT, select_device, seed_everything


@dataclass
class PlanningSetup:
    planner: NRRTStarGNG2D
    wrapper: object
    device: object
    seconds: float


def prepare_planner(grid, start, goal, *, step_len=10, iterations=3000,
                    neural_sample_rate=0.5, checkpoint=DEFAULT_CHECKPOINT,
                    device='auto', threads=4, seed=100, wrapper=None):
    """Shared CLI/GUI map-to-network-to-NRRT* pipeline.

    Pass a previously loaded wrapper to reuse its weights, but infer fresh
    guidance for every map and endpoint query. The planner's path_score is the
    very same network output from which its sampling corridor was obtained.
    """
    if not np.isfinite(step_len) or step_len <= 0 or iterations < 1:
        raise ValueError('Step length and iteration budget must be positive')
    if not 0 <= neural_sample_rate <= 1:
        raise ValueError('Neural sample rate must be between 0 and 1')
    grid.validate_endpoint(start, 'Start')
    grid.validate_endpoint(goal, 'Goal')
    started = time.perf_counter()
    selected_device = select_device(device, threads)
    seed_everything(seed)
    if neural_sample_rate and wrapper is None:
        from wrapper.unet.unet_wrapper import GNGWrapper
        wrapper = GNGWrapper(checkpoint, selected_device)
    planner = NRRTStarGNG2D(start, goal, step_len, iterations, grid,
                           wrapper, neural_sample_rate)
    return PlanningSetup(planner, wrapper, selected_device, time.perf_counter()-started)


def run_planner(planner, *, stop_on_first=False, progress_callback=None, cancel_check=None):
    """Run the prepared NRRT* tree and validate its returned path."""
    path = planner.planning(stop_on_first=stop_on_first,
                            progress_callback=progress_callback, cancel_check=cancel_check)
    if len(path) and (not np.array_equal(path[0], planner.x_start) or
                      not np.array_equal(path[-1], planner.x_goal) or
                      any(planner.env.is_collision(a, b) for a, b in zip(path[:-1], path[1:]))):
        raise RuntimeError('Internal error: returned path failed final collision validation')
    return path


def save_plot(filename, grid, planner):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(grid.binary_mask, cmap='gray', vmin=0, vmax=1, origin='upper')
    if len(planner.guidance):
        axis.scatter(planner.guidance[:, 0], planner.guidance[:, 1], s=1,
                     color='orange', alpha=0.35, label='Neural guidance')
    if len(planner.path):
        axis.plot(planner.path[:, 0], planner.path[:, 1], color='red', linewidth=2, label='NRRT* path')
    axis.scatter(*planner.x_start, c='lime', marker='*', s=110, edgecolors='black', label='Start')
    axis.scatter(*planner.x_goal, c='cyan', marker='*', s=110, edgecolors='black', label='Goal')
    axis.set_title('ResNet50/U-Net NRRT*: {}'.format('path found' if len(planner.path) else 'no path found'))
    axis.set_xlabel('x (pixels)')
    axis.set_ylabel('y (pixels, downwards)')
    axis.legend(loc='best', markerscale=1.5)
    figure.tight_layout()
    figure.savefig(str(filename), dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, help='Image with white free space and black obstacles')
    parser.add_argument('--start', type=int, nargs=2, metavar=('X', 'Y'))
    parser.add_argument('--goal', type=int, nargs=2, metavar=('X', 'Y'))
    parser.add_argument('--sample', help='Dataset query ENV_QUERY, e.g. 100_2 (default: 0_0)')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--step-len', type=float, default=10)
    parser.add_argument('--clearance', type=int, default=3, help='Square obstacle/boundary inflation radius in pixels')
    parser.add_argument('--neural-sample-rate', type=float, default=0.5)
    parser.add_argument('--threshold', type=int, default=127)
    parser.add_argument('--invert', action='store_true', help='Treat dark pixels as free instead')
    parser.add_argument('--stop-on-first', action='store_true')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, help='New output directory; defaults to runs/planning/<unique run>')
    args = parser.parse_args()
    if not 0 <= args.neural_sample_rate <= 1 or args.iterations < 1 or args.step_len <= 0 or args.clearance < 0:
        parser.error('Check sample rate [0,1], positive iterations/step length, and nonnegative clearance')
    if args.map:
        if args.sample or args.start is None or args.goal is None:
            parser.error('--map requires --start X Y and --goal X Y, and cannot be combined with --sample')
        filename, start, goal = args.map.expanduser().resolve(), args.start, args.goal
    else:
        if args.start is not None or args.goal is not None:
            parser.error('Explicit endpoints require --map')
        try:
            env_index, query_index = map(int, (args.sample or '0_0').split('_'))
            if min(env_index, query_index) < 0:
                raise ValueError()
            split_root = args.data_root.expanduser().resolve()/args.split
            envs = json.loads((split_root/'envs.json').read_text())
            env = envs[env_index]
            start, goal = env['start'][query_index], env['goal'][query_index]
            filename = split_root/'env_imgs'/'{}.png'.format(env_index)
        except (ValueError, IndexError):
            parser.error('Unknown dataset sample; use ENV_QUERY, e.g. 0_0')
    grid = GridMap(load_binary_map(filename, args.threshold, args.invert), args.clearance)
    grid.validate_endpoint(start, 'Start')
    grid.validate_endpoint(goal, 'Goal')
    setup = prepare_planner(grid, start, goal, step_len=args.step_len, iterations=args.iterations,
                            neural_sample_rate=args.neural_sample_rate, checkpoint=args.checkpoint,
                            device=args.device, threads=args.threads, seed=args.seed)
    planner = setup.planner
    path = run_planner(planner, stop_on_first=args.stop_on_first)
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    output = (args.output or ROOT/'runs/planning'/run_name).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    result = dict(success=bool(len(path)), map=str(filename), start=list(start), goal=list(goal),
                  path_length=planner.get_path_len(path) if len(path) else None,
                  path=path.tolist(), iterations=planner.iterations_run,
                  tree_vertices=planner.num_vertices, guidance_points=len(planner.guidance),
                  model_and_guidance_seconds=setup.seconds, planning_seconds=planner.planning_seconds,
                  first_solution_seconds=planner.first_solution_seconds,
                  best_solution_seconds=planner.best_solution_seconds,
                  total_seconds=setup.seconds+planner.planning_seconds,
                  first_solution_with_setup_seconds=(setup.seconds+planner.first_solution_seconds
                                                     if planner.first_solution_seconds is not None else None),
                  neural_samples=planner.neural_samples, uniform_samples=planner.uniform_samples,
                  device=str(setup.device), config=config)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savetxt(str(output/'path.csv'), path, delimiter=',', header='x,y', comments='')
    save_plot(output/'plan.png', grid, planner)
    print(json.dumps({k: v for k, v in result.items() if k not in ['path', 'config']}, indent=2))
    print('Output: {}'.format(output))
    return 0 if len(path) else 2


if __name__ == '__main__':
    raise SystemExit(main())

# ResNet NRRT*

A standalone 2D grid-based NRRT* project extracted and adapted from [tedhuang96/nirrt_star](https://github.com/tedhuang96/nirrt_star), local revision `285dd24`. It includes the ICRA 2024 ResNet50/U-Net checkpoint, the 2D image/A* training data, map generation, network training/evaluation, and planning on your own image maps.

This is **NRRT***: ResNet50 encodes a map and endpoint markers, a U-Net decoder predicts a path corridor, and RRT* mixes corridor samples with global free-space samples. The PointNet-based NIRRT* informed/Neural Connect branches are outside this project.

## Clone from GitHub

Install [Git LFS](https://git-lfs.com/) before cloning to download the pretrained checkpoint:

```bash
git lfs install
git clone https://github.com/Smithbn4/NIRRT-.git
cd NIRRT-
git lfs pull
```

The repository includes the training dataset. The model checkpoint is stored with Git LFS.

## Tkinter custom-map GUI

From the copied `ResNet_NRRT_GUI` folder, launch:

```bash
cd "$HOME/Desktop/ResNet_NRRT_GUI"
/home/smithbn4/miniconda3/envs/pngenv/bin/python gui.py
```

With that environment activated, `python gui.py` also works. The GUI uses NumPy,
OpenCV, and Tkinter, and loads the supplied ResNet50/U-Net checkpoint through
PyTorch to guide planning. This interface compares **ResNet50/U-Net-guided NRRT***
with **eight-neighbor A***. Both the GUI and CLI call `prepare_planner()` and
`run_planner()` in `plan.py`; they share inference, NRRT* construction, search,
and final path validation. Tkinter needs a desktop display;
if creating a fresh conda environment without Tk support, install `tk` there.

1. Enter map width and height (each at least 2 pixels), start/goal x and y, and an
   optional clearance radius. There is no fixed map-size limit; available memory
   and search time limit practical sizes. Map dimensions and coordinates are integers.
2. Add up to **10 total obstacles**, mixing rectangles and circles. Rectangles use
   their top-left x/y, length along x, and width along y. A rectangle at `(10, 20)`
   with length 5 and width 3 occupies x=10…14, y=20…22. Circles use center x/y and
   a positive integer radius; occupied pixels have centers within that radius.
   Every obstacle must fit entirely inside the map. Use Remove selected or Clear all
   to revise the list.
3. Click **Preview map** to inspect occupancy and endpoints, then **Solve both**.
   The origin is top-left, x increases rightward, and y increases downward. Start
   and goal must be free, including the requested square clearance margin.
4. The left view shows the unsolved map. The right view overlays the NRRT* path in
   red and the A* path in dashed blue. Both planners use the same full-resolution
   occupancy and clearance; resizing the window only scales the display. A* runs
   first, then its result stays visible during neural inference and NRRT* search.
   Cancel stops the search and retains completed results and any NRRT* path found.

The solved map also displays the network's **per-pixel path-class probability**
underneath the paths. The fixed 0–1 color scale runs from purple (low) through teal
to yellow (high); obstacles and clearance margins keep their occupancy colors.
Use **Show neural probability heatmap** to hide/show it without rerunning a solver.
The custom map's occupancy pixels and start/goal markers are fed into the network
at the map's original resolution, with the same padding and normalization as the
CLI. **One inference per query** provides both the heatmap and the NRRT* sampling
corridor. Hiding the overlay only changes the display; neural guidance remains active.

The sampler preserves the NRRT* implementation in this folder: it uniformly
samples network-positive corridor pixels (path probability > 0.5), after filtering
occupied/clearance pixels, and mixes these with global free-space samples. It does
not sample proportionally to each pixel's probability. **Neural sampling rate**
defaults to `0.5`, meaning 50% corridor targets and 50% global targets. The GUI accepts
rates greater than zero through one. The results include actual neural/global
target-draw counts, which sum to the search iterations performed. If no corridor
pixels survive clearance, the existing NRRT* global-sampling fallback is explicitly
reported. High scores predict a path corridor; collision checking still uses the map.

The supplied checkpoint (`checkpoints/icra2024_resnet50_unet.pt`) loads on CPU
the first time you solve a map and is reused for later queries. Loading and
inference run in the background after A* and before NRRT*. Their elapsed time is
shown separately, along with search-only and setup-inclusive solve times. Cancelling during inference
takes effect after the current neural operation completes. If the checkpoint or
neural libraries are unavailable, NRRT* reports an error and the completed A*
result remains visible. It does not substitute plain RRT* for failed neural inference.
The checkpoint was trained on 224-by-224 random maps; prediction
quality on other sizes and layouts may vary.

The comparison table reports final path length in pixels, time to first solution,
total search time, and search iterations/visited cells. NRRT* additionally reports
the **time to last improvement**: elapsed search time when the final best path
length was first reached. Its convergence plot records every strict best-length
improvement. These are observed finite-run metrics, not proof of mathematical
convergence. Search times use `time.perf_counter()` and exclude map construction,
planner construction, neural processing, final collision validation, and GUI drawing.
The model/guidance setup duration covers device/seed initialization, initial model
loading when needed, inference, and tree construction. **First solution incl. setup**
and **Total time incl. setup** add that duration to the corresponding NRRT* search
measurements. They exclude the separate A* comparison run and map creation. A later
query reuses loaded weights and therefore has a lower setup cost. NRRT* search timing
includes convergence tracking; A* timing includes its search bookkeeping. Host
scheduling can affect wall-clock results. The random seed controls NRRT* samples.

By default NRRT* searches for the full iteration budget to improve its path. Select
**Stop at first NRRT* path** to measure only its first feasible solution. An exhausted
NRRT* budget is reported as “no path found,” rather than proving infeasibility. A*
exhaustion establishes that no path exists on this eight-neighbor raster graph.
A* optimizes grid-edge length, while NRRT* can use continuous straight edges, so the
two lengths need not match even after a long run. Unreachable or cancelled searches
display no path length instead of a misleading zero.

## Quick start on this computer

Your existing environment is `/home/smithbn4/miniconda3/envs/pngenv`. It already has the needed libraries. From a terminal:

```bash
cd "$HOME/Desktop/ResNet_NRRT_GUI"
conda activate /home/smithbn4/miniconda3/envs/pngenv
python plan.py --sample 0_0 --device cpu --stop-on-first
```

If your terminal cannot find `conda`, you can directly use the existing interpreter:

```bash
/home/smithbn4/miniconda3/envs/pngenv/bin/python plan.py --sample 0_0 --device cpu --stop-on-first
```

Run these commands from this project's directory.

The planner saves a unique folder under `runs/planning/` with:

- `plan.png`: map, predicted guidance points, endpoints, and resulting path.
- `path.csv`: the path in x,y pixel coordinates.
- `result.json`: success, coordinates, path length, timing, parameters, and checkpoint path.

Without `--stop-on-first`, planning continues for the full iteration budget to improve the path. A no-solution result exits with status 2 and still saves its diagnostic image and JSON. An exhausted budget does not establish that no feasible path exists.

For a fresh environment, use Python 3.9:

```bash
conda env create -f environment.yml
conda activate resnet-nrrt
```

Or install `requirements.txt` into a Python 3.9 environment. The dependency versions preserve the original compatible Torch/Torchvision stack. This extraction does not need Open3D, SciPy, PointNet, ROS, or notebook packages. The supplied environment has no accessible CUDA device; CPU execution has been verified.

## Solve a provided map-------------------------------------------------------------------------

Use a PNG/JPEG or another OpenCV-readable image. By default, white pixels are free and black pixels are obstacles. Values greater than 127 are free; `--threshold` changes that rule. `--invert` reverses it. Input maps should represent occupancy, with obstacle geometry already encoded in their pixels.

```bash
python plan.py --map examples/map.png --start 132 112 --goal 29 12 \
  --clearance 3 --iterations 3000 --device cpu --stop-on-first
```

For your own file:

```bash
python plan.py --map /path/to/your_map.png --start 20 30 --goal 180 190 \
  --clearance 3 --iterations 5000
```

Replace the example endpoints with free pixels in your image. The origin is the **top-left**; x increases rightwards and y increases downwards. The image is indexed `[y, x]`. Endpoints are integer pixel coordinates. `--clearance` is a nonnegative integer radius in pixels, applied as square obstacle inflation and an equal margin at map boundaries. An invalid endpoint produces an error before inference.

Maps of other dimensions are padded with occupied pixels at their bottom/right to multiples of 32, then network outputs are cropped to the original dimensions. There is no resizing or change to endpoint/path coordinates. The supplied checkpoint was trained on 224-by-224 random worlds; guidance quality on very different maps or resolutions needs validation. Collision checking always uses the actual input occupancy map.

Other useful options:

```bash
python plan.py --sample 100_2 --split test --seed 100
python plan.py --sample 0_0 --neural-sample-rate 0.5 --step-len 10
python plan.py --sample 0_0 --checkpoint runs/my_training/best_unet.pt
python plan.py --map examples/map.png --start 132 112 --goal 29 12 --output runs/my_plan
python plan.py --help
```

The default device is `auto`: CUDA if available, otherwise CPU. A neural sample rate of zero skips model loading and runs the underlying RRT* sampler, which can be useful when checking a map. With nonempty guidance and the default rate 0.5, half the target samples are uniformly selected predicted-positive pixels and half are globally selected free pixels. Empty guidance automatically falls back to free sampling.

## Included data and pretrained weights

```text
checkpoints/icra2024_resnet50_unet.pt    Original complete U-Net state dictionary
data/random_2d/train/                   1,000 maps; 4,000 labeled queries
data/random_2d/val/                     125 maps; 500 labeled queries
data/random_2d/test/                    125 maps; 500 labeled queries
```

Each split contains `envs.json`, `env_imgs/<map>.png`, and `astar_paths/<map>_<query>.txt`. The map configurations include endpoints and the original circles/rectangles. Planning uses the raster image for collision checks, so it also supports maps without analytic obstacle lists. Precomputed point-cloud NPZ files are unnecessary and were omitted.

The original three-channel input is preserved:

```text
start marker | goal marker | free-space map
```

Markers are 3-by-3 squares, clipped at image edges. Inputs use the original ImageNet mean/std normalization. The ResNet50 feature channels and decoder architecture are unchanged, including the two-class output. Therefore the original checkpoint loads strictly without renaming its tensor keys or downloading ImageNet weights. Changing input channels or decoder sizes would require new compatible weights.

## Generate new 2D training maps

The generator uses the original rectangle/circle map generator and eight-neighbor A* labeler, with an explicit configuration, seed, output location, and bounded retry count.

```bash
python generate_random_world_env_2d.py --output data/my_maps \
  --train-envs 1000 --val-envs 125 --test-envs 125 --seed 100
```

For a small first run:

```bash
python generate_random_world_env_2d.py --output data/small_maps \
  --train-envs 10 --val-envs 2 --test-envs 2 --samples-per-env 4
```

Edit `env_configs/random_2d.yml` or pass `--config another_config.yml` to change map dimensions, obstacle sizes/counts, clearance, and endpoint separation. Keep one image resolution per generated dataset for batched training. Changing dimensions does not require multiples of 32 because the loader pads them. For rectangular maps, augmentation uses flips and 180-degree rotations so batch dimensions remain consistent.

Output directories must be new, protecting the included dataset. A* labels are raster paths. Their path thickness for network training is three pixels by default; they are reference grid paths, not exact continuous-space optima.

## Train or fine-tune the network

By default, training starts from the supplied ICRA checkpoint, freezes encoder parameters and BatchNorm state, and fine-tunes the decoder. The dataset loads maps/labels lazily rather than duplicating every image input in RAM.

```bash
python train_unet.py --data-root data/random_2d --output runs/my_training \
  --epochs 100 --batch-size 8 --device auto
```

Or train using the dataset you generated:

```bash
python train_unet.py --data-root data/my_maps --output runs/custom_training \
  --epochs 100 --batch-size 8 --learning-rate 0.001
```

Each run writes:

- `best_unet.pt`: model with the highest validation path IoU.
- `metrics.csv`: training/validation loss, path IoU, and epoch duration.
- `config.json`: parameters, class weights, and device.
- `loss_curve.png`: training and validation loss versus epoch, refreshed after every completed epoch.
- `final_accuracy_report.txt` and `final_accuracy_report.json`: final test results for the saved **best** checkpoint, evaluated after training finishes.

The CSV also records training and validation pixel accuracy. The final report includes weighted cross-entropy loss, pixel accuracy, mean per-query path IoU, global path IoU, path precision, recall, F1/Dice, sample/pixel counts and a confusion matrix. Accuracy metrics in JSON are fractions from 0 to 1; the text report displays percentages. Pixel accuracy includes all valid non-path pixels, including obstacles, so inspect path IoU and F1 as well. These are segmentation measurements against the A* corridor, not planning success rates or path-length error. Padding pixels are ignored.

The test split is evaluated only after training, using the checkpoint selected by validation path IoU. Use `--skip-test-evaluation` for datasets containing only train/val splits. `--max-test-samples N` limits this final evaluation for a quick pipeline check; the report always states how many queries were evaluated. A report on a limited subset is not a full test-set evaluation.

The graph and CSV remain available if you interrupt training with Ctrl+C. The automatic final report is produced only on normal completion. Checkpoint replacement is atomic, so an interrupted save leaves the previous complete `best_unet.pt` intact.

To generate reports for an existing or interrupted run, or regenerate its loss graph from `metrics.csv`:

```bash
python eval_unet.py --data-root data/random_2d \
  --checkpoint runs/my_training/best_unet.pt --split test \
  --output runs/my_training --device auto
```

`--output` on the evaluation command may be an existing directory. It writes the two final reports and, if `metrics.csv` is present there, refreshes `loss_curve.png`. Without `--output`, evaluation prints metrics to the terminal as before. A running Python process does not pick up these code changes; start a new training process to obtain the automatic graph and report.

`--output` must name a new directory, so training does not overwrite the included checkpoint or a previous run. `--train-encoder` also trains encoder weights and BatchNorm state. `--init-checkpoint /path/to/best_unet.pt` fine-tunes another compatible model. This initializes model weights rather than restoring an optimizer's previous state.

To initialize a new full network:

```bash
python train_unet.py --from-scratch --output runs/from_scratch --batch-size 8
```

To initialize its encoder from ImageNet, as in the original training setup:

```bash
python train_unet.py --from-scratch --imagenet --output runs/imagenet_training --batch-size 8
```

Only `--imagenet` may need an initial network download. Fully untrained runs train the encoder too; freezing random encoder weights is avoided. CPU training is supported but a full 100-epoch run will be slow. Reduce batch size if needed for memory; `--threads` sets CPU worker threads and `--workers` sets DataLoader processes.

Loss is class-weighted cross entropy for non-path/path pixels. Padding pixels are ignored. Validation selects the highest mean per-query path IoU. Partial final batches are handled using their actual sizes. This is segmentation evaluation, distinct from planner success or path length.

If a fully trained encoder sees a final singleton batch with a 1-by-1 bottleneck (a 32-by-32 input), that batch uses accumulated encoder BatchNorm statistics so no sample needs to be discarded.

Evaluate the supplied or a newly trained checkpoint:

```bash
python eval_unet.py --checkpoint checkpoints/icra2024_resnet50_unet.pt
python eval_unet.py --data-root data/my_maps --checkpoint runs/custom_training/best_unet.pt
python plan.py --sample 0_0 --checkpoint runs/my_training/best_unet.pt
```

For a brief pipeline check on an existing dataset, use `--epochs 1 --max-train-samples 3 --max-val-samples 3 --batch-size 2` with a new output directory. This checks execution, not training quality.

## Source layout

| Location | Responsibility |
| --- | --- |
| `plan.py` | Shared neural inference/NRRT* pipeline, custom-map/dataset CLI, output export |
| `gui.py` | Tkinter custom-map editor, NRRT*/A* comparison, guidance heatmap, timing and convergence |
| `path_planning_utils/custom_map.py` | Map dimensions, obstacle validation, and occupancy rasterization |
| `path_planning_utils/heatmap.py` | Fixed-scale neural probability colors and legend |
| `generate_random_world_env_2d.py` | 2D map and A* label generation |
| `train_unet.py`, `eval_unet.py` | Training and segmentation evaluation |
| `unet/models/` | Original ResNet50 encoder and U-Net decoder |
| `unet/image_dataset.py`, `unet/training.py` | Lazy loading, labels, augmentation, loss and metrics |
| `wrapper/unet/unet_wrapper.py` | Same semantic preprocessing, padding, inference, coordinate conversion |
| `path_planning_classes/rrt_base_2d.py` | Tree storage, path cost and extraction |
| `path_planning_classes/rrt_star_2d.py` | Expansion, parent selection, rewiring, valid goal connection |
| `path_planning_classes/nrrt_star_gng_2d.py` | Mix neural guidance with free-space samples |
| `datasets/grid_map.py` | Occupancy maps, clearance inflation, supercover segment collision checks |
| `path_planning_utils/Astar_with_clearance.py` | Grid A* labeler and GUI solver, timing and cancellation |
| `tests/test_core.py` | Collision, fallback, preprocessing, padding, and metric checks |
| `tests/test_gui_planning.py` | Custom-map validation, solver comparison, timing, and cancellation checks |
| `tests/test_gui_heatmap.py` | Shared NRRT* pipeline, neural sampling, matching heatmap, clearance, and failure checks |

Run the behavioral checks with:

```bash
python -m unittest discover -s tests -v
```

Before copying this project to your Desktop, validation on the existing CPU environment passed nine behavioral checks, loaded the supplied checkpoint strictly, solved a supplied test query and a 67-by-45 custom map, generated a new labeled dataset, completed one fine-tuning epoch and one from-scratch epoch, and reloaded/evaluated the saved fine-tuned checkpoint. Both short training runs used a partial final batch. These runs verify the pipeline, not convergence or performance on new maps.

The RRT* tree is allocated for one iteration budget. A fresh planner is constructed per query. Neural positives violating obstacle clearance are removed, and every accepted edge and the final exported path are collision-checked. Thin obstacles and diagonal corner crossings use a supercover raster traversal. No supplied analytic rectangle/circle geometry is required for custom-map planning.

## Provenance and scope of changes

The original repository's MIT license is retained in `LICENSE` (copyright Zhe Huang, 2024). The pretrained checkpoint and copied image/A* datasets come from your extracted ICRA 2024 downloads. `ASSETS.json` records the copied checkpoint's checksum and dataset counts.

The model architecture, semantic input format, and neural/global sampling mixture are preserved. The RRT* code was reduced to the expansion machinery needed by NRRT*, and map collision checking was adapted to occupancy pixels to support arbitrary provided maps. This intentionally changes collision discretization from the original analytic circle/rectangle checker; this extraction is not a bit-for-bit reproduction of the paper's planning results.

Additional adaptations include command-line configuration, CPU support, offline checkpoint initialization, padding/cropping, lazy loading, bounded generation retries, safe output locations, valid goal-parent selection, and empty-guidance fallback. Unused 3D, PointNet/PointNet++, informed planning, Neural Connect, block/gap evaluation, historical results, archives, and extracted checkpoint internals are excluded. Your original `nirrt_star` folder was not modified by this extraction.

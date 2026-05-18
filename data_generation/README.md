GesVLA Data Generation Pipeline

This repository provides a hand–object interaction data generation pipeline. It renders hand meshes, performs object detection, and generates data for multiple task types (grab, grab-and-move, etc.).

## Project layout

```
data_generation/
├─ data_generator/              # Core pipeline code
├─ handpoint.glb                # Right-hand mesh
├─ left_handpoint.glb           # Left-hand mesh
└─ assets/groundingdino/         # (Create) GroundingDINO config and weights
```

## Dependencies

Create the conda environment from `environment.yml` and install dependencies first:

```
conda env create -f environment.yml
conda activate gesvla_data
```


> Notes:
> - `groundingdino` is not on PyPI by default; install it following the official repo instructions.

## Install GroundingDINO

Clone and install GroundingDINO after the base environment is ready:

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

## GroundingDINO setup

Make sure the following files existed:

```
GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
GroundingDINO/weights/groundingdino_swint_ogc.pth
```

If the filenames or paths differ, update these fields in `data_generator/configs.py`:

- `DetectionConfig.model_config_path`
- `DetectionConfig.model_weights_path`

## Input data layout

The pipeline expects `data_root` to directly contain subfolders starting with `episode_`.

Default (see `data_generator/configs.py`):

```
data_root = data/input
```

Expected layout:

```
data/input/
  episode_000/
    right_rgb_frame_0.png
    depth_frame_0.png
```
`episode_000/` represents a background scene and must include an RGB image and a depth image.

## How to run

Run from `data_generation/`:

```
python -m data_generator.main
```

By default, it runs keypoint detection and dataset collection. The generated data will be organized in a format suitable for GesVLA hand-gesture reasoning training.


## Where to adjust parameters

All configuration is in `data_generator/configs.py`:

- `CameraConfig`: camera intrinsics, `depth_scale`
- `DetectionConfig`: DINO config/weight paths and thresholds
- `HandConfig`: hand mesh paths and scale
- `GenerationConfig`: `data_root`, `output_root`, episode range, task sample counts

Tasks are enabled when their sample count is greater than 0. See `GenerationConfig.get_enabled_tasks()`.

We currently define two task categories. The "grab" and "grab_and_move" series targets block-and-plate tasks, following a sequence of pointing to blocks first and then to plates. Therefore, data generation requires detection results for both blocks and plates. "grab" produces data where the hand points to blocks; "grab_and_move" produces data where the hand points to blocks first and then to plates.

The "grab_one"/"grab_two" series targets general pointing commands and can be applied to any scene by changing the DINO prompt in configs. "grab_one" generates data pointing to one object, "grab_two" points to two objects, and so on. To extend, follow the existing logic and add new tasks in `pipeline.py`.

## Output structure

Generated data is written to:

```
output_root/episode_{index}_{right_hand|left_hand}/
```

Each task creates samples under its own subfolder.



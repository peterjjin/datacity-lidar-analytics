# Neilson Car Experiment

This experiment uses the existing Neilson KITTI-style export from:

`/mnt/d/Dropbox/SmartMobility/2022.11.23 3D-2D Object Integration/Code/RULabeler/data/AlbanyNeilson/20240201/DATA_20240201_213000/kitti_open3dml`

The setup here builds a clean local KITTI root for `Car` baselines and
fine-tuning in:

- `MMDetection3D`
- `OpenPCDet`
- `Open3D-ML`

Why `Car` first:

- It is the dominant class in Neilson.
- It aligns with the strongest standard KITTI pretrained baselines.
- It avoids forcing weak class remaps for `Trailer`, `Tractor`, `Bus`, and the
  long-tail rider classes before the first benchmark is stable.

## Layout

- `configs/mmdet3d_pointpillars_car.py`
  - MMDetection3D config for Neilson `Car`.
- `configs/openpcdet_pointpillar_car.yaml`
  - OpenPCDet PointPillar config for Neilson `Car`.
- `configs/openpcdet_second_car.yaml`
  - OpenPCDet SECOND config for Neilson `Car`.
- `configs/open3dml_pointpillars_car.yml`
  - Open3D-ML PointPillars config for Neilson `Car`.
- `scripts/prepare_neilson_car_kitti.py`
  - Builds a normalized KITTI root under this workspace.
- `scripts/prepare_mmdet_infos.py`
  - Generates `neilson_car_infos_*.pkl` without camera-FOV cropping.
- `scripts/check_mmdet_dataset.py`
  - Smoke-checks that the MMDetection3D dataset loads.
- `scripts/run_mmdet3d_baseline.sh`
  - Runs evaluation with a pretrained checkpoint you provide.
- `scripts/train_mmdet3d.sh`
  - Runs training or fine-tuning.
- `scripts/train_openpcdet.sh`
  - Runs OpenPCDet training or fine-tuning.
- `scripts/train_open3dml.sh`
  - Runs Open3D-ML training or fine-tuning.
- `scripts/train_all_algorithms.py`
  - Launches the supported training jobs with one manifest.
- `scripts/prepare_autoware_t4_intermediate.py`
  - Converts Neilson into an Autoware/T4-oriented intermediate dataset.
- `AUTOWARE_T4_DATASET.md`
  - Notes on how Autoware training expects lidar files and labels.

## Generated Data Root

By default the scripts write to:

`/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics/data/neilson_car_kitti`

## First-time setup

```bash
python3 experiments/neilson_car/scripts/prepare_neilson_car_kitti.py
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lidar-3d-base
python experiments/neilson_car/scripts/prepare_mmdet_infos.py
python experiments/neilson_car/scripts/check_mmdet_dataset.py
```

## Baseline evaluation

Pretrained checkpoints are not bundled in this workspace. Put them in:

`experiments/neilson_car/checkpoints/`

Expected filenames:

- `mmdet3d_pointpillars_kitti.pth`
- `openpcdet_second_kitti.pth`
- `openpcdet_pointpillar_kitti.pth`
- `open3dml_pointpillars_kitti.pth`

Example:

```bash
python3 experiments/neilson_car/scripts/benchmark_all_algorithms.py \
  --output-root experiments/neilson_car/benchmark_runs_pretrained
```

## Open3D-ML dataset smoke test

Open3D-ML's built-in `KITTI` dataset loader does not use `ImageSets/*.txt`.
Instead, it slices train/val with a numeric `val_split`. For Neilson, that
split point is the train-frame count: `4164`.

Prepare the Open3D-ML compatibility config:

```bash
python3 experiments/neilson_car/scripts/prepare_open3dml_dataset.py
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lidar-open3d-ml
python experiments/neilson_car/scripts/check_open3dml_dataset.py
python experiments/neilson_car/scripts/check_open3dml_env.py
```

## Training or fine-tuning

MMDetection3D:

```bash
bash experiments/neilson_car/scripts/train_mmdet3d.sh
```

MMDetection3D fine-tuning:

```bash
bash experiments/neilson_car/scripts/train_mmdet3d.sh /path/to/checkpoint.pth
```

OpenPCDet PointPillar fine-tuning:

```bash
CONFIG_PATH=experiments/neilson_car/configs/openpcdet_pointpillar_car.yaml \
PRETRAINED_MODEL=experiments/neilson_car/checkpoints/openpcdet_pointpillar_kitti.pth \
EXTRA_TAG=neilson_pointpillar_finetune \
bash experiments/neilson_car/scripts/train_openpcdet.sh
```

OpenPCDet SECOND fine-tuning:

```bash
CONFIG_PATH=experiments/neilson_car/configs/openpcdet_second_car.yaml \
PRETRAINED_MODEL=experiments/neilson_car/checkpoints/openpcdet_second_kitti.pth \
EXTRA_TAG=neilson_second_finetune \
bash experiments/neilson_car/scripts/train_openpcdet.sh
```

OpenPCDet SECOND note on RTX 5090:

- The current `SECONDNet` fine-tune path is not working on the local RTX 5090
  stack.
- The latest diagnostic run (`logs/diagnose_openpcdet_second_20260404_201724.log`)
  reproduced a deterministic `SIGFPE` inside `spconv/cumm` in a standalone
  sparse-convolution smoke test and again during the first training step.
- Freezing `vfe,backbone_3d` does not fix this on the current toolchain, so the
  launchers now fail fast for `openpcdet_second_car` on `sm_120` instead of
  starting a run that will crash inside `spconv`. The diagnostic script now
  applies that same guard before attempting the crash-prone smoke test matrix.
- Use `openpcdet_pointpillar_car`, a different GPU, or an updated
  `spconv/cumm` toolchain that supports this stack cleanly.

Open3D-ML fine-tuning:

```bash
DEVICE=cuda \
CHECKPOINT_PATH=experiments/neilson_car/checkpoints/open3dml_pointpillars_kitti.pth \
RUN_ROOT=experiments/neilson_car/runs/open3dml_pointpillars_car_finetune \
bash experiments/neilson_car/scripts/train_open3dml.sh
```

Open3D-ML caveat on RTX 5090:

- The local Open3D path now applies a repo patch that replaces the unstable
  Open3D `voxelize` and `ragged_to_dense` CUDA ops with a guarded compatibility
  fallback.
- The fallback materializes the PointPillars voxelization stage on CPU,
  validates the returned row splits, then copies the compact voxel outputs back
  to the model device.
- This keeps PointPillars training/inference on GPU while avoiding the
  illegal-memory-access crash seen on RTX 5090 during voxelization.
- The Open3D-ML launchers in this workspace are GPU-only and now reject any
  explicit CPU device selection.
- If CUDA still fails, debug the active `lidar-open3d-ml` environment and the
  installed Open3D wheel rather than switching this workflow to CPU.

Unified launch:

```bash
python3 experiments/neilson_car/scripts/train_all_algorithms.py --epochs 24 --run-tag finetune
```

## Autoware / T4 intermediate export

Autoware's current public training path does not train directly from the
Neilson per-frame JSON labels. It expects a T4 dataset layout. The intermediate
export in this repo writes lidar payloads as `.pcd.bin` and frame-level box
records that match the fields used by TIER IV's annotation tooling.

```bash
python3 experiments/neilson_car/scripts/prepare_autoware_t4_intermediate.py
```

See:

- `experiments/neilson_car/AUTOWARE_T4_DATASET.md`

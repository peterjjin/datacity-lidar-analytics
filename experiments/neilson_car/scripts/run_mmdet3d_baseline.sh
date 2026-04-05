#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash experiments/neilson_car/scripts/run_mmdet3d_baseline.sh /path/to/checkpoint.pth [-- extra test args]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift

WORKSPACE_ROOT="/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
CONFIG_PATH="$WORKSPACE_ROOT/experiments/neilson_car/configs/mmdet3d_pointpillars_car.py"
DATA_ROOT="${NEILSON_CAR_KITTI_ROOT:-$WORKSPACE_ROOT/data/neilson_car_kitti}"
MMDET3D_DIR="${MMDET3D_DIR:-$HOME/src/mmdetection3d}"
WORK_DIR="${WORK_DIR:-$WORKSPACE_ROOT/experiments/neilson_car/runs/mmdet3d_baseline}"

export NEILSON_CAR_KITTI_ROOT="$DATA_ROOT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lidar-3d-base

mkdir -p "$WORK_DIR"
cd "$MMDET3D_DIR"
python tools/test.py "$CONFIG_PATH" "$CHECKPOINT" --work-dir "$WORK_DIR" "$@"

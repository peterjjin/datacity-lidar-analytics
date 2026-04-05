#!/usr/bin/env bash
set -euo pipefail

run_python_script() {
  local script_path="$1"
  shift
  python -c 'import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")' \
    "$script_path" "$@"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_PATH="$WORKSPACE_ROOT/experiments/neilson_car/configs/mmdet3d_pointpillars_car.py"
DATA_ROOT="${NEILSON_CAR_KITTI_ROOT:-$WORKSPACE_ROOT/data/neilson_car_kitti}"
MMDET3D_DIR="${MMDET3D_DIR:-$HOME/src/mmdetection3d}"
WORK_DIR="${WORK_DIR:-$WORKSPACE_ROOT/experiments/neilson_car/runs/mmdet3d_train}"
VAL_INTERVAL="${MMDET3D_VAL_INTERVAL:-1000}"

LOAD_FROM=""
if [[ $# -gt 0 && "$1" != "--" ]]; then
  LOAD_FROM="$1"
  shift
fi

EXTRA_ARGS=("$@")

export NEILSON_CAR_KITTI_ROOT="$DATA_ROOT"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
export MMDET3D_LOAD_FROM="${LOAD_FROM}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lidar-3d-base

run_python_script "$WORKSPACE_ROOT/experiments/neilson_car/scripts/require_cuda.py" \
  --label "MMDetection3D (lidar-3d-base)" \
  --require-arch
run_python_script "$WORKSPACE_ROOT/experiments/neilson_car/scripts/prepare_mmdet_infos.py"

mkdir -p "$WORK_DIR"
cd "$MMDET3D_DIR"

run_python_script tools/train.py "$CONFIG_PATH" --work-dir "$WORK_DIR" \
  --cfg-options "train_cfg.val_interval=$VAL_INTERVAL" "${EXTRA_ARGS[@]}"

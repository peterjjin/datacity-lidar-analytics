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
DATA_ROOT="${NEILSON_CAR_KITTI_ROOT:-$WORKSPACE_ROOT/data/neilson_car_kitti}"
CONFIG_PATH="${CONFIG_PATH:-$WORKSPACE_ROOT/experiments/neilson_car/configs/open3dml_pointpillars_car.yml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$WORKSPACE_ROOT/experiments/neilson_car/checkpoints/open3dml_pointpillars_kitti.pth}"
RUN_ROOT="${RUN_ROOT:-$WORKSPACE_ROOT/experiments/neilson_car/runs/open3dml_pointpillars_car_train}"
ENV_NAME="${ENV_NAME:-lidar-open3d-ml}"
DEVICE="${DEVICE:-cuda}"
VAL_SPLIT="${VAL_SPLIT:-}"
MAX_EPOCH="${MAX_EPOCH:-24}"
NUM_WORKERS="${NUM_WORKERS:-12}"

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [[ "$DEVICE" != "cuda" ]]; then
  echo "Open3D-ML training is GPU-only in this workspace; received DEVICE=$DEVICE" >&2
  exit 2
fi

run_python_script "$WORKSPACE_ROOT/experiments/neilson_car/scripts/require_cuda.py" \
  --label "Open3D-ML ($ENV_NAME)" \
  --require-arch

run_python_script "$WORKSPACE_ROOT/experiments/neilson_car/scripts/prepare_open3dml_dataset.py"

if [[ -z "$VAL_SPLIT" ]]; then
  VAL_SPLIT="$(python3 - <<'PY'
from pathlib import Path
import os
path = Path(os.environ["NEILSON_CAR_KITTI_ROOT"]) / "ImageSets" / "train.txt"
print(sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
PY
)"
fi

run_python_script "$WORKSPACE_ROOT/experiments/neilson_car/scripts/train_open3dml_pointpillars.py" \
  --config "$CONFIG_PATH" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT_PATH" \
  --run-root "$RUN_ROOT" \
  --device "$DEVICE" \
  --val-split "$VAL_SPLIT" \
  --max-epoch "$MAX_EPOCH" \
  --num-workers "$NUM_WORKERS"

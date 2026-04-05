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
OPENPCDET_PATCHED_ROOT="${OPENPCDET_PATCHED_ROOT:-/tmp/openpcdet_patched}"
DATA_ROOT="${NEILSON_CAR_KITTI_ROOT:-$WORKSPACE_ROOT/data/neilson_car_kitti}"
CONFIG_PATH="${CONFIG_PATH:-$WORKSPACE_ROOT/experiments/neilson_car/configs/openpcdet_pointpillar_car.yaml}"
ENV_NAME="${ENV_NAME:-lidar-openpcdet}"
EXTRA_TAG="${EXTRA_TAG:-neilson_train}"
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-24}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-}"
OPENPCDET_FREEZE_MODULES="${OPENPCDET_FREEZE_MODULES:-}"

EXTRA_ARGS=("$@")

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
export OPENPCDET_FREEZE_MODULES
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

run_python_script "$WORKSPACE_ROOT/experiments/neilson_car/scripts/bootstrap_openpcdet_runtime.py"

export NEILSON_CAR_KITTI_ROOT="$DATA_ROOT"
export PYTHONPATH="$OPENPCDET_PATCHED_ROOT:/tmp/openpcdet_extra_py${PYTHONPATH:+:$PYTHONPATH}"

run_python_script "$WORKSPACE_ROOT/experiments/neilson_car/scripts/require_cuda.py" \
  --label "OpenPCDet ($ENV_NAME)" \
  --require-arch

run_python_script "$WORKSPACE_ROOT/experiments/neilson_car/scripts/guard_openpcdet_runtime.py" \
  --cfg-file "$CONFIG_PATH" \
  --label "OpenPCDet ($ENV_NAME)"

if [[ ! -d "$OPENPCDET_PATCHED_ROOT/tools" ]]; then
  echo "Missing OpenPCDet patched tree: $OPENPCDET_PATCHED_ROOT/tools" >&2
  exit 1
fi

run_python_script "$WORKSPACE_ROOT/experiments/neilson_car/scripts/prepare_openpcdet_infos.py"

cd "$OPENPCDET_PATCHED_ROOT/tools"

CMD=(
  python -c 'import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")' train.py
  --cfg_file "$CONFIG_PATH"
  --extra_tag "$EXTRA_TAG"
  --workers "$WORKERS"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
)

if [[ -n "$PRETRAINED_MODEL" ]]; then
  CMD+=(--pretrained_model "$PRETRAINED_MODEL")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

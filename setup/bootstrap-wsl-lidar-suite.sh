#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
BASE_ENV_NAME="${BASE_ENV_NAME:-lidar-3d-base}"
OPENPCDET_ENV_NAME="${OPENPCDET_ENV_NAME:-lidar-openpcdet}"
OPEN3D_ML_ENV_NAME="${OPEN3D_ML_ENV_NAME:-lidar-open3d-ml}"
OPEN3D_ML_ENV_FILE="$WORKSPACE_ROOT/setup/lidar-open3d-ml-wsl.yml"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SKIP_APT=0

usage() {
  cat <<EOF
Usage:
  bash setup/bootstrap-wsl-lidar-suite.sh [--skip-apt]

What it does:
  1. Builds the main MMDetection3D-ready WSL environment.
  2. Builds a separate OpenPCDet environment.
  3. Builds a separate Open3D-ML environment.
  4. Runs package checks for all configured environments.

Optional flags:
  --skip-apt
      Reuses existing Ubuntu build tools instead of calling apt.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

step() {
  echo
  echo "==> $*"
}

if [[ ! -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]]; then
  echo "Miniforge conda init script not found: $MINIFORGE_DIR/etc/profile.d/conda.sh" >&2
  echo "Run setup/bootstrap-wsl-lidar.sh first if Miniforge is not installed yet." >&2
  exit 1
fi

APT_ARGS=()
if [[ "$SKIP_APT" -eq 1 ]]; then
  APT_ARGS+=(--skip-apt)
fi

step "Preparing main WSL detection environment"
bash "$WORKSPACE_ROOT/setup/bootstrap-wsl-lidar.sh" "${APT_ARGS[@]}" --with-torch --with-mmdet3d

step "Preparing isolated OpenPCDet environment"
bash "$WORKSPACE_ROOT/setup/bootstrap-wsl-openpcdet.sh"

step "Preparing isolated Open3D-ML environment"
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$OPEN3D_ML_ENV_NAME"; then
  conda env update -n "$OPEN3D_ML_ENV_NAME" -f "$OPEN3D_ML_ENV_FILE" --prune
else
  conda env create -f "$OPEN3D_ML_ENV_FILE"
fi

conda activate "$OPEN3D_ML_ENV_NAME"
python -m pip install --upgrade pip wheel
python -m pip install --upgrade torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"
python - <<'PY'
import open3d
import torch
print("open3d", open3d.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

python "$WORKSPACE_ROOT/experiments/neilson_car/scripts/require_cuda.py" \
  --label "Open3D-ML bootstrap ($OPEN3D_ML_ENV_NAME)" \
  --require-arch

step "Running suite checks"
bash "$WORKSPACE_ROOT/setup/wsl-doctor.sh" \
  --env "$BASE_ENV_NAME" \
  --openpcdet-env "$OPENPCDET_ENV_NAME" \
  --open3d-ml-env "$OPEN3D_ML_ENV_NAME"

step "Suite setup complete"
echo "Recommended next steps:"
echo "  1. Download model checkpoints."
echo "  2. Run bash \"$WORKSPACE_ROOT/setup/run-mmdet3d-model-test.sh\" --list-presets"
echo "  3. Validate OpenPCDet-specific CUDA deps such as spconv inside $OPENPCDET_ENV_NAME."

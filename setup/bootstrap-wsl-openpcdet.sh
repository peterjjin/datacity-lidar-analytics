#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$WORKSPACE_ROOT/setup/lidar-openpcdet-wsl.yml"
ENV_NAME="${ENV_NAME:-lidar-openpcdet}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
OPENPCDET_DIR="${OPENPCDET_DIR:-$HOME/src/OpenPCDet}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SETUPTOOLS_PIN="${SETUPTOOLS_PIN:-60.2.0}"

step() {
  echo
  echo "==> $*"
}

source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

step "Creating or updating conda environment from $ENV_FILE"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  conda env create -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"

step "Installing PyTorch from $PYTORCH_INDEX_URL"
python -m pip install --upgrade pip wheel "setuptools==${SETUPTOOLS_PIN}"
python -m pip install --upgrade torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"

step "Cloning OpenPCDet"
mkdir -p "$(dirname "$OPENPCDET_DIR")"
if [[ ! -d "$OPENPCDET_DIR/.git" ]]; then
  git clone https://github.com/open-mmlab/OpenPCDet.git "$OPENPCDET_DIR"
fi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

python "$WORKSPACE_ROOT/experiments/neilson_car/scripts/require_cuda.py" \
  --label "OpenPCDet bootstrap ($ENV_NAME)" \
  --require-arch

step "Next action"
echo "OpenPCDet is intentionally kept separate from MMDetection3D."
echo "This script prepares the env and repo clone only."
echo "Install spconv and the project-specific build deps after validating the current upstream compatibility for your GPU/toolchain."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$WORKSPACE_ROOT/setup/lidar-3d-base-wsl.yml"
ENV_NAME="${ENV_NAME:-lidar-3d-base}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
MMDET3D_DIR="${MMDET3D_DIR:-$HOME/src/mmdetection3d}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SETUPTOOLS_PIN="${SETUPTOOLS_PIN:-60.2.0}"
INSTALL_OPEN3D_ML=0

INSTALL_TORCH=0
INSTALL_MMDET3D=0
SKIP_APT=0

usage() {
  cat <<'EOF'
Usage:
  bash setup/bootstrap-wsl-lidar.sh [--skip-apt] [--with-torch] [--with-mmdet3d] [--with-open3d-ml]

What it does:
  - installs Ubuntu system packages needed for LiDAR Python/C++ stacks
  - installs Miniforge into ~/miniforge3 if missing
  - creates or updates the lidar-3d-base conda environment

Optional flags:
  --skip-apt
      Skips sudo apt install. Use this only if the Ubuntu build tools are already present.
  --with-torch
      Installs torch/torchvision/torchaudio using PYTORCH_INDEX_URL.
  --with-mmdet3d
      Installs MMDetection3D into ~/src/mmdetection3d after torch is available.
  --with-open3d-ml
      Installs Open3D and the Open3D-ML extras into the active base environment.

Environment variables:
  PYTORCH_INDEX_URL
      Defaults to https://download.pytorch.org/whl/cu128
  SETUPTOOLS_PIN
      Defaults to 60.2.0 to keep pkg_resources available for mmcv/openxlab.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-torch)
      INSTALL_TORCH=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --with-mmdet3d)
      INSTALL_MMDET3D=1
      INSTALL_TORCH=1
      shift
      ;;
    --with-open3d-ml)
      INSTALL_OPEN3D_ML=1
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

configure_cuda_env() {
  local detected_cuda_home=""

  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    detected_cuda_home="$CUDA_HOME"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    detected_cuda_home="$CONDA_PREFIX"
  elif command -v nvcc >/dev/null 2>&1; then
    detected_cuda_home="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else
    for candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
      if [[ -x "$candidate/bin/nvcc" ]]; then
        detected_cuda_home="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$detected_cuda_home" ]]; then
    return 1
  fi

  export CUDA_HOME="$detected_cuda_home"
  case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
  esac

  if [[ -d "$CUDA_HOME/lib64" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$CUDA_HOME/lib64:"*) ;;
      *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" ;;
    esac
  elif [[ -d "$CUDA_HOME/lib" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$CUDA_HOME/lib:"*) ;;
      *) export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}" ;;
    esac
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

if [[ "$SKIP_APT" -eq 1 ]]; then
  step "Skipping Ubuntu package install"
  echo "Assuming the required build tools are already present."
else
  step "Installing Ubuntu packages"
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required for the apt step. Re-run with --skip-apt if the packages are already installed." >&2
    exit 1
  fi
  if ! sudo -n true >/dev/null 2>&1; then
    echo "sudo could not run non-interactively in this shell." >&2
    echo "Run this script in your normal WSL terminal, or re-run with --skip-apt if the packages are already installed." >&2
    exit 1
  fi
  sudo apt update
  sudo apt install -y \
    build-essential \
    git \
    git-lfs \
    curl \
    wget \
    zip \
    unzip \
    cmake \
    ninja-build \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-dev
fi

if [[ ! -x "$MINIFORGE_DIR/bin/conda" ]]; then
  step "Installing Miniforge into $MINIFORGE_DIR"
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  wget -O "$TMP_DIR/Miniforge3.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$TMP_DIR/Miniforge3.sh" -b -p "$MINIFORGE_DIR"
else
  step "Miniforge already present at $MINIFORGE_DIR"
fi

require_cmd bash
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

if ! grep -q 'miniforge3/etc/profile.d/conda.sh' "$HOME/.bashrc" 2>/dev/null; then
  step "Adding conda initialization to ~/.bashrc"
  {
    echo ""
    echo "# Added by setup/bootstrap-wsl-lidar.sh"
    echo "source \"$MINIFORGE_DIR/etc/profile.d/conda.sh\""
  } >> "$HOME/.bashrc"
fi

step "Creating or updating conda environment from $ENV_FILE"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  conda env create -f "$ENV_FILE"
fi

set +u
conda activate "$ENV_NAME"
set -u

step "Preparing Python packaging tools"
python -m pip install --upgrade pip wheel "setuptools==${SETUPTOOLS_PIN}"

if [[ "$INSTALL_TORCH" -eq 1 ]]; then
  step "Installing PyTorch from $PYTORCH_INDEX_URL"
  python -m pip install --upgrade torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
else
  step "Skipping PyTorch install"
  echo "Re-run with --with-torch after checking the current official PyTorch selector if needed."
fi

if [[ "$INSTALL_OPEN3D_ML" -eq 1 ]]; then
  step "Installing Open3D and Open3D-ML support"
  python -m pip install --upgrade open3d
  python -m pip install --upgrade scikit-learn pandas
  python - <<'PY'
import open3d
print("open3d", open3d.__version__)
try:
    import open3d.ml as _ml
    print("open3d_ml", "available")
except Exception as exc:
    print("open3d_ml", f"import_failed: {exc}")
    raise
PY
else
  step "Skipping Open3D-ML install"
  echo "Re-run with --with-open3d-ml if you want Open3D/Open3D-ML in the base WSL environment."
fi

if [[ "$INSTALL_MMDET3D" -eq 1 ]]; then
  step "Installing MMDetection3D"
  python - <<'PY'
import importlib.util
import sys
missing = [name for name in ("torch",) if importlib.util.find_spec(name) is None]
sys.exit(1 if missing else 0)
PY
  python -m pip install -U openmim
  mim install "mmengine>=0.8.0"
  if ! mim install "mmcv>=2.0.0rc4,<2.2.0"; then
    step "Retrying MMCV from source with build isolation disabled"
    python -m pip uninstall -y mmcv mmcv-lite >/dev/null 2>&1 || true
    python -m pip install --upgrade wheel "setuptools==${SETUPTOOLS_PIN}"
    if ! configure_cuda_env; then
      echo "MMCV needs a CUDA toolkit in WSL because no prebuilt wheel is available for this torch/CUDA pair." >&2
      echo "CUDA_HOME is not set and no CUDA toolkit with nvcc was found in the active conda env, PATH, or /usr/local/cuda*." >&2
      echo "Install a matching CUDA toolkit in WSL, then re-run this script." >&2
      exit 1
    fi
    echo "Using CUDA toolkit at $CUDA_HOME"
    MMCV_WITH_OPS=1 python -m pip install -v --no-build-isolation "mmcv>=2.0.0rc4,<2.2.0"
  fi
  mim install "mmdet>=3.0.0,<3.4.0"
  mkdir -p "$(dirname "$MMDET3D_DIR")"
  if [[ ! -d "$MMDET3D_DIR/.git" ]]; then
    git clone https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_DIR"
  fi
  python -m pip install --upgrade "numpy<2" "opencv-python<4.11" "opencv-python-headless<4.11" "shapely~=2.0.3"
  (
    cd "$MMDET3D_DIR"
    python setup.py develop
  )
  python - <<'PY'
import mmdet3d
print("mmdet3d", mmdet3d.__version__)
PY
else
  step "Skipping MMDetection3D install"
  echo "Re-run with --with-mmdet3d once torch is working in WSL."
fi

step "Done"
echo "Next checks:"
echo "  bash \"$WORKSPACE_ROOT/setup/wsl-doctor.sh\""
echo "  python -c \"import torch; print(torch.cuda.is_available())\""

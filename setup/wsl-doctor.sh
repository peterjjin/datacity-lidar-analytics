#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-lidar-3d-base}"
OPENPCDET_ENV_NAME="${OPENPCDET_ENV_NAME:-lidar-openpcdet}"
OPEN3D_ML_ENV_NAME="${OPEN3D_ML_ENV_NAME:-lidar-open3d-ml}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"

PASS_COUNT=0
WARN_COUNT=0

usage() {
  cat <<'EOF'
Usage:
  bash setup/wsl-doctor.sh [--env NAME] [--openpcdet-env NAME] [--open3d-ml-env NAME]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --openpcdet-env)
      OPENPCDET_ENV_NAME="$2"
      shift 2
      ;;
    --open3d-ml-env)
      OPEN3D_ML_ENV_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "[pass] $*"
}

warn() {
  WARN_COUNT=$((WARN_COUNT + 1))
  echo "[warn] $*"
}

check_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "command available: $cmd"
  else
    warn "command missing: $cmd"
  fi
}

check_env_python_import() {
  local env_name="$1"
  local module_name="$2"
  local label="$3"

  if ! command -v conda >/dev/null 2>&1; then
    warn "conda is not available for $label check"
    return
  fi

  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    warn "$label env missing: $env_name"
    return
  fi

  if conda run -n "$env_name" python -c "import $module_name" >/dev/null 2>&1; then
    local version
    version="$(conda run -n "$env_name" python - <<PY 2>/dev/null
import ${module_name} as mod
print(getattr(mod, "__version__", "unknown"))
PY
)"
    pass "$label import works in $env_name: $version"
  else
    warn "$label import failed in $env_name"
  fi
}

check_env_torch_cuda() {
  local env_name="$1"
  local label="$2"

  if ! command -v conda >/dev/null 2>&1; then
    warn "conda is not available for $label CUDA check"
    return
  fi

  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    warn "$label env missing for CUDA check: $env_name"
    return
  fi

  local cuda_report
  set +e
  cuda_report="$(bash -lc "source \"$MINIFORGE_DIR/etc/profile.d/conda.sh\" && \
    conda activate \"$env_name\" && \
    python \"$WORKSPACE_ROOT/experiments/neilson_car/scripts/require_cuda.py\" \
      --label \"$label\" \
      --require-arch" 2>&1)"
  local cuda_status=$?
  set -e

  if [[ "$cuda_status" -eq 0 ]]; then
    pass "$label CUDA works in $env_name"
  else
    warn "$label CUDA check failed in $env_name"
    while IFS= read -r line; do
      [[ -n "$line" ]] && echo "  $line"
    done <<<"$cuda_report"
  fi
}

echo "WSL doctor for LiDAR detection"
echo "workspace: $WORKSPACE_ROOT"
echo

check_cmd uname
check_cmd python3
check_cmd git
check_cmd nvidia-smi

if mount | grep -q ' on /mnt/c '; then
  pass "Windows drives are mounted under /mnt"
else
  warn "Windows drives do not appear to be mounted under /mnt"
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
  pass "running inside WSL"
else
  warn "this does not appear to be a WSL kernel"
fi

if [[ -e /dev/dxg ]]; then
  pass "WSL GPU device is present: /dev/dxg"
else
  warn "WSL GPU device is missing: /dev/dxg"
fi

if [[ -f /proc/sys/fs/binfmt_misc/WSLInterop ]]; then
  pass "Windows interop binfmt entry is present"
else
  warn "Windows interop binfmt entry is missing"
fi

if command -v cmd.exe >/dev/null 2>&1; then
  if cmd.exe /c ver >/dev/null 2>&1; then
    pass "Windows interop execution works"
  else
    warn "cmd.exe is on PATH but Windows interop execution failed"
  fi
else
  warn "cmd.exe is not available on PATH"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_SMI_L="$(nvidia-smi -L 2>/dev/null || true)"
  if grep -q '^GPU ' <<<"$NVIDIA_SMI_L"; then
    pass "GPU visible in WSL"
  else
    warn "GPU not usable from WSL via nvidia-smi"
  fi
fi

if [[ -x "$MINIFORGE_DIR/bin/conda" ]]; then
  pass "Miniforge present at $MINIFORGE_DIR"
  # shellcheck disable=SC1091
  source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
else
  warn "Miniforge not found at $MINIFORGE_DIR"
fi

if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    pass "conda env exists: $ENV_NAME"
    set +u
    conda activate "$ENV_NAME"
    set -u
    if python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
      pass "python runs inside $ENV_NAME"
    else
      warn "python failed inside $ENV_NAME"
    fi
  else
    warn "conda env missing: $ENV_NAME"
  fi
fi

if python -c "import torch" >/dev/null 2>&1; then
  TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
  pass "torch import works: $TORCH_VERSION"
  TORCH_ARCH="$(python - <<'PY' 2>/dev/null
import sys
import torch

if not torch.cuda.is_available():
    sys.exit(1)

major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
print(arch)
sys.exit(0 if arch in torch.cuda.get_arch_list() else 2)
PY
)"
  TORCH_CUDA_STATUS=$?
  if [[ "$TORCH_CUDA_STATUS" -eq 0 ]]; then
    DEVICE_NAME="$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    pass "torch CUDA works: $DEVICE_NAME ($TORCH_ARCH)"
  elif [[ "$TORCH_CUDA_STATUS" -eq 2 ]]; then
    warn "torch sees the GPU but this wheel does not include the active architecture: $TORCH_ARCH"
  else
    warn "torch installed but CUDA is not available"
  fi
else
  warn "torch is not installed in the active environment"
fi

if python -c "import mmdet3d" >/dev/null 2>&1; then
  MMDET3D_VERSION="$(python -c 'import mmdet3d; print(mmdet3d.__version__)')"
  pass "mmdet3d import works: $MMDET3D_VERSION"
else
  warn "mmdet3d is not installed in the active environment"
fi

check_env_python_import "$OPENPCDET_ENV_NAME" torch "OpenPCDet torch"
check_env_torch_cuda "$OPENPCDET_ENV_NAME" "OpenPCDet torch"
if conda env list | awk '{print $1}' | grep -qx "$OPENPCDET_ENV_NAME"; then
  if conda run -n "$OPENPCDET_ENV_NAME" python -c "import pcdet" >/dev/null 2>&1; then
    pass "OpenPCDet import works in $OPENPCDET_ENV_NAME"
  else
    warn "OpenPCDet package is not installed yet in $OPENPCDET_ENV_NAME"
  fi
fi

check_env_python_import "$OPEN3D_ML_ENV_NAME" torch "Open3D-ML torch"
check_env_torch_cuda "$OPEN3D_ML_ENV_NAME" "Open3D-ML torch"
check_env_python_import "$OPEN3D_ML_ENV_NAME" open3d "Open3D"
if conda env list | awk '{print $1}' | grep -qx "$OPEN3D_ML_ENV_NAME"; then
  if conda run -n "$OPEN3D_ML_ENV_NAME" python -c "import open3d.ml" >/dev/null 2>&1; then
    pass "open3d.ml import works in $OPEN3D_ML_ENV_NAME"
  else
    warn "open3d.ml import failed in $OPEN3D_ML_ENV_NAME"
  fi
fi

echo
echo "Summary: $PASS_COUNT pass, $WARN_COUNT warn"
if [[ "$WARN_COUNT" -gt 0 ]]; then
  echo
  echo "Suggested next checks:"
  if [[ ! -e /dev/dxg ]]; then
    echo "  - In Windows PowerShell: run .\\setup\\wsl-host-check.ps1"
    echo "  - In an elevated PowerShell window: wsl --shutdown"
    echo "  - Reboot Windows if WSL GPU support was just enabled or updated"
  fi
  if [[ ! -f /proc/sys/fs/binfmt_misc/WSLInterop ]]; then
    echo "  - Check /etc/wsl.conf and your Windows WSL settings for interop changes"
  fi
  if command -v cmd.exe >/dev/null 2>&1 && ! cmd.exe /c ver >/dev/null 2>&1; then
    echo "  - Run from Windows PowerShell: wsl --shutdown"
    echo "  - Reopen Ubuntu and retest cmd.exe /c ver"
  fi
  exit 1
fi

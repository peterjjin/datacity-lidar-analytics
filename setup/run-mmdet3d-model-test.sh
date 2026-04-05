#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-lidar-3d-base}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
MMDET3D_DIR="${MMDET3D_DIR:-$HOME/src/mmdetection3d}"
CONFIG=""
CHECKPOINT=""
WORK_DIR=""
DEVICE="${DEVICE:-cuda:0}"
USE_TTA=0
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage:
  bash setup/run-mmdet3d-model-test.sh --config RELPATH --checkpoint PATH [options] [-- extra test args]

Required:
  --config RELPATH
      Path relative to \$MMDET3D_DIR, for example:
      configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
  --checkpoint PATH
      Path to a downloaded checkpoint file.

Options:
  --repo PATH
      Override MMDetection3D repo path. Default: \$HOME/src/mmdetection3d
  --work-dir PATH
      Output directory for logs and predictions. Default:
      $WORKSPACE_ROOT/runs/<config-basename>
  --device NAME
      Torch device. Default: cuda:0
  --list-presets
      Print example vehicle/pedestrian model configs and exit.
  --tta
      Enable test-time augmentation if the selected config supports it.

Examples:
  bash setup/run-mmdet3d-model-test.sh \\
    --config configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \\
    --checkpoint /data/checkpoints/centerpoint_nuscenes.pth

  bash setup/run-mmdet3d-model-test.sh \\
    --config configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \\
    --checkpoint /data/checkpoints/pointpillars_kitti_3class.pth \\
    --device cpu -- --show-dir "$WORKSPACE_ROOT/runs/vis"
EOF
}

list_presets() {
  cat <<'EOF'
Suggested MMDetection3D starting points for vehicle/pedestrian testing:

  nuScenes:
    configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
    configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py

  KITTI 3-class:
    configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py
    configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py

Notes:
  - KITTI 3-class configs cover car, pedestrian, and cyclist.
  - nuScenes configs evaluate the standard nuScenes class set; vehicle and pedestrian metrics are a useful first pass.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --repo)
      MMDET3D_DIR="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --list-presets)
      list_presets
      exit 0
      ;;
    --tta)
      USE_TTA=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
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

if [[ -z "$CONFIG" || -z "$CHECKPOINT" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda init script: $MINIFORGE_DIR/etc/profile.d/conda.sh" >&2
  exit 1
fi

source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if ! python -c "import mmdet3d" >/dev/null 2>&1; then
  echo "mmdet3d is not importable in $ENV_NAME." >&2
  echo "Run: bash \"$WORKSPACE_ROOT/setup/bootstrap-wsl-lidar.sh\" --with-mmdet3d" >&2
  exit 1
fi

if [[ ! -d "$MMDET3D_DIR" ]]; then
  echo "MMDetection3D repo not found: $MMDET3D_DIR" >&2
  exit 1
fi

if [[ ! -f "$MMDET3D_DIR/$CONFIG" ]]; then
  echo "Config not found: $MMDET3D_DIR/$CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ -z "$WORK_DIR" ]]; then
  CONFIG_BASENAME="$(basename "$CONFIG" .py)"
  WORK_DIR="$WORKSPACE_ROOT/runs/$CONFIG_BASENAME"
fi

mkdir -p "$WORK_DIR"

python - <<PY
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_request", "${DEVICE}")
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo "Running MMDetection3D test"
echo "repo: $MMDET3D_DIR"
echo "config: $MMDET3D_DIR/$CONFIG"
echo "checkpoint: $CHECKPOINT"
echo "work_dir: $WORK_DIR"
echo "device: $DEVICE"

TEST_ARGS=(
  "$CONFIG"
  "$CHECKPOINT"
  --work-dir "$WORK_DIR"
  --cfg-options default_hooks.logger.interval=10
)

if [[ "$USE_TTA" -eq 1 ]]; then
  TEST_ARGS+=(--tta)
fi

TEST_ARGS+=("${EXTRA_ARGS[@]}")

cd "$MMDET3D_DIR"
python tools/test.py "${TEST_ARGS[@]}"

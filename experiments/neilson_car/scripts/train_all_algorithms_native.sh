#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
CONFIG_ROOT="$WORKSPACE_ROOT/experiments/neilson_car/configs"
CHECKPOINT_ROOT="$WORKSPACE_ROOT/experiments/neilson_car/checkpoints"
RUN_ROOT="$WORKSPACE_ROOT/experiments/neilson_car/runs"
LOG_ROOT="$WORKSPACE_ROOT/logs"
MMDET3D_ROOT="${MMDET3D_ROOT:-$HOME/src/mmdetection3d}"
OPENPCDET_PATCHED_ROOT="${OPENPCDET_PATCHED_ROOT:-/tmp/openpcdet_patched}"

EPOCHS=24
WORKERS=12
RUN_TAG="finetune"
OPEN3D_DEVICE="cuda"
DATA_ROOT=""
DRY_RUN=0
ALGORITHMS=(
  "mmdet3d_pointpillars_car"
  "openpcdet_pointpillar_car"
  "openpcdet_second_car"
  "open3dml_pointpillars_kitti"
)

usage() {
  cat <<'EOF'
Usage:
  bash experiments/neilson_car/scripts/train_all_algorithms_native.sh [options]

Options:
  --epochs N
  --workers N
  --run-tag TAG
  --open3d-device DEVICE
  --data-root PATH
  --algorithm NAME
  --dry-run
  -h, --help

Defaults:
  --epochs 24
  --workers 12
  --run-tag finetune
  --open3d-device cuda
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --open3d-device)
      OPEN3D_DEVICE="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --algorithm)
      if [[ ${#ALGORITHMS[@]} -eq 4 ]]; then
        ALGORITHMS=()
      fi
      ALGORITHMS+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ "$OPEN3D_DEVICE" != "cuda" ]]; then
  echo "Open3D-ML training is GPU-only in this workspace; received --open3d-device $OPEN3D_DEVICE" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="$LOG_ROOT/training_log_${TIMESTAMP}.log"
MANIFEST_PATH="$WORKSPACE_ROOT/experiments/neilson_car/training_manifest_native.json"

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="$WORKSPACE_ROOT/data/neilson_car_kitti"
fi
DATA_ROOT="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$DATA_ROOT")"

RECIPES_JSON=""

log() {
  local message="$1"
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$message" | tee -a "$LOG_PATH"
}

append_recipe() {
  local name="$1"
  local env_name="$2"
  local command="$3"
  local escaped
  escaped="$(printf '%s' "$command" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  if [[ -n "$RECIPES_JSON" ]]; then
    RECIPES_JSON+=","
  fi
  RECIPES_JSON+=$'\n'"    {\"name\": \"${name}\", \"env\": \"${env_name}\", \"command\": ${escaped}}"
  export RECIPES_JSON
}

write_manifest() {
  python3 - "$MANIFEST_PATH" "$LOG_PATH" "$EPOCHS" "$WORKERS" "$RUN_TAG" "$OPEN3D_DEVICE" "$DATA_ROOT" <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

manifest_path, log_path, epochs, workers, run_tag, open3d_device, data_root = sys.argv[1:]
recipes_json = os.environ.get("RECIPES_JSON", "").strip()
recipes = json.loads(f"[{recipes_json}\n]") if recipes_json else []

payload = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "log_path": log_path,
    "epochs": int(epochs),
    "workers": int(workers),
    "run_tag": run_tag,
    "open3d_device": open3d_device,
    "data_root": data_root,
    "recipes": recipes,
}

with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
}

run_recipe() {
  local name="$1"
  local env_name="$2"
  local command="$3"

  append_recipe "$name" "$env_name" "$command"
  log "=== [$name] starting in $env_name ==="

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\n' "$command" | tee -a "$LOG_PATH"
    log "=== [$name] dry-run complete ==="
    return
  fi

  set +e
  bash -lc "$command" 2>&1 | tee -a "$LOG_PATH"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    log "=== [$name] failed with exit $status ==="
    write_manifest
    exit "$status"
  fi

  log "=== [$name] completed ==="
}

log "logging to $LOG_PATH"
log "workspace root: $WORKSPACE_ROOT"
log "data root: $DATA_ROOT"

for algorithm in "${ALGORITHMS[@]}"; do
  case "$algorithm" in
    mmdet3d_pointpillars_car)
      MMDET_WORK_DIR="$RUN_ROOT/mmdet3d_pointpillars_car_${RUN_TAG}"
      MMDET_CMD="
source \"$CONDA_SH\" &&
conda activate lidar-3d-base &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  python -c \"import torch; ok=torch.cuda.is_available(); print(ok, torch.cuda.get_device_name(0) if ok else 'no-gpu'); raise SystemExit(0 if ok else 1)\" &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  python \"$WORKSPACE_ROOT/experiments/neilson_car/scripts/prepare_mmdet_infos.py\" &&
mkdir -p \"$MMDET_WORK_DIR\" &&
cd \"$MMDET3D_ROOT\" &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  NVCC_PREPEND_FLAGS=\"\${NVCC_PREPEND_FLAGS-}\" \
  MPLCONFIGDIR=\"\${MPLCONFIGDIR:-/tmp/matplotlib}\" \
  XDG_CACHE_HOME=\"\${XDG_CACHE_HOME:-/tmp/.cache}\" \
  MMDET3D_LOAD_FROM=\"$CHECKPOINT_ROOT/mmdet3d_pointpillars_kitti.pth\" \
  MMDET3D_TRAIN_WORKERS=\"$WORKERS\" \
  MMDET3D_VAL_WORKERS=\"$WORKERS\" \
  MMDET3D_VAL_INTERVAL=\"$((EPOCHS + 1))\" \
  PYTHONPATH=\"$MMDET3D_ROOT\" \
  python \"$MMDET3D_ROOT/tools/train.py\" \
  \"$CONFIG_ROOT/mmdet3d_pointpillars_car.py\" \
  --work-dir \"$MMDET_WORK_DIR\" \
  --cfg-options train_cfg.val_interval=$((EPOCHS + 1)) train_cfg.max_epochs=$EPOCHS
"
      run_recipe "$algorithm" "lidar-3d-base" "$MMDET_CMD"
      ;;
    openpcdet_pointpillar_car)
      OPENPCDET_CMD="
source \"$CONDA_SH\" &&
conda activate lidar-openpcdet &&
env NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" python \"$WORKSPACE_ROOT/experiments/neilson_car/scripts/bootstrap_openpcdet_runtime.py\" &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  PYTHONPATH=\"$OPENPCDET_PATCHED_ROOT:/tmp/openpcdet_extra_py\${PYTHONPATH:+:\$PYTHONPATH}\" \
  python \"$WORKSPACE_ROOT/experiments/neilson_car/scripts/require_cuda.py\" --label openpcdet &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  PYTHONPATH=\"$OPENPCDET_PATCHED_ROOT:/tmp/openpcdet_extra_py\${PYTHONPATH:+:\$PYTHONPATH}\" \
  python \"$WORKSPACE_ROOT/experiments/neilson_car/scripts/guard_openpcdet_runtime.py\" --cfg-file \"$CONFIG_ROOT/openpcdet_pointpillar_car.yaml\" --label openpcdet &&
test -d \"$OPENPCDET_PATCHED_ROOT/tools\" &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  PYTHONPATH=\"$OPENPCDET_PATCHED_ROOT:/tmp/openpcdet_extra_py\${PYTHONPATH:+:\$PYTHONPATH}\" \
  python -c \"import runpy; runpy.run_path('$WORKSPACE_ROOT/experiments/neilson_car/scripts/prepare_openpcdet_infos.py', run_name='__main__')\" &&
python3 - <<'PY'
from pathlib import Path
import yaml
src = Path(\"$CONFIG_ROOT/openpcdet_pointpillar_car.yaml\")
dst = Path(\"/tmp/openpcdet_pointpillar_car.subset.yaml\")
payload = yaml.safe_load(src.read_text(encoding=\"utf-8\"))
payload[\"DATA_CONFIG\"][\"DATA_PATH\"] = \"$DATA_ROOT\"
dst.write_text(yaml.safe_dump(payload, sort_keys=False), encoding=\"utf-8\")
print(dst)
PY
cd \"$OPENPCDET_PATCHED_ROOT/tools\" &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  NVCC_PREPEND_FLAGS=\"\${NVCC_PREPEND_FLAGS-}\" \
  PYTHONPATH=\"$OPENPCDET_PATCHED_ROOT:/tmp/openpcdet_extra_py\${PYTHONPATH:+:\$PYTHONPATH}\" \
  python -c \"import runpy, sys; sys.argv=['train.py', '--cfg_file', '/tmp/openpcdet_pointpillar_car.subset.yaml', '--extra_tag', 'neilson_pointpillar_${RUN_TAG}', '--workers', '$WORKERS', '--batch_size', '4', '--epochs', '$EPOCHS', '--pretrained_model', '$CHECKPOINT_ROOT/openpcdet_pointpillar_kitti.pth']; runpy.run_path('$OPENPCDET_PATCHED_ROOT/tools/train.py', run_name='__main__')\"
"
      run_recipe "$algorithm" "lidar-openpcdet" "$OPENPCDET_CMD"
      ;;
    openpcdet_second_car)
      OPENPCDET_CMD="
source \"$CONDA_SH\" &&
conda activate lidar-openpcdet &&
env NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" python \"$WORKSPACE_ROOT/experiments/neilson_car/scripts/bootstrap_openpcdet_runtime.py\" &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  PYTHONPATH=\"$OPENPCDET_PATCHED_ROOT:/tmp/openpcdet_extra_py\${PYTHONPATH:+:\$PYTHONPATH}\" \
  python \"$WORKSPACE_ROOT/experiments/neilson_car/scripts/require_cuda.py\" --label openpcdet &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  PYTHONPATH=\"$OPENPCDET_PATCHED_ROOT:/tmp/openpcdet_extra_py\${PYTHONPATH:+:\$PYTHONPATH}\" \
  python \"$WORKSPACE_ROOT/experiments/neilson_car/scripts/guard_openpcdet_runtime.py\" --cfg-file \"$CONFIG_ROOT/openpcdet_second_car.yaml\" --label openpcdet &&
test -d \"$OPENPCDET_PATCHED_ROOT/tools\" &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  PYTHONPATH=\"$OPENPCDET_PATCHED_ROOT:/tmp/openpcdet_extra_py\${PYTHONPATH:+:\$PYTHONPATH}\" \
  python -c \"import runpy; runpy.run_path('$WORKSPACE_ROOT/experiments/neilson_car/scripts/prepare_openpcdet_infos.py', run_name='__main__')\" &&
python3 - <<'PY'
from pathlib import Path
import yaml
src = Path(\"$CONFIG_ROOT/openpcdet_second_car.yaml\")
dst = Path(\"/tmp/openpcdet_second_car.subset.yaml\")
payload = yaml.safe_load(src.read_text(encoding=\"utf-8\"))
payload[\"DATA_CONFIG\"][\"DATA_PATH\"] = \"$DATA_ROOT\"
dst.write_text(yaml.safe_dump(payload, sort_keys=False), encoding=\"utf-8\")
print(dst)
PY
cd \"$OPENPCDET_PATCHED_ROOT/tools\" &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  NVCC_PREPEND_FLAGS=\"\${NVCC_PREPEND_FLAGS-}\" \
  PYTHONPATH=\"$OPENPCDET_PATCHED_ROOT:/tmp/openpcdet_extra_py\${PYTHONPATH:+:\$PYTHONPATH}\" \
  python -c \"import runpy, sys; sys.argv=['train.py', '--cfg_file', '/tmp/openpcdet_second_car.subset.yaml', '--extra_tag', 'neilson_second_${RUN_TAG}', '--workers', '$WORKERS', '--batch_size', '4', '--epochs', '$EPOCHS', '--pretrained_model', '$CHECKPOINT_ROOT/openpcdet_second_kitti.pth']; runpy.run_path('$OPENPCDET_PATCHED_ROOT/tools/train.py', run_name='__main__')\"
"
      run_recipe "$algorithm" "lidar-openpcdet" "$OPENPCDET_CMD"
      ;;
    open3dml_pointpillars_kitti)
      OPEN3D_RUN_ROOT="$RUN_ROOT/open3dml_pointpillars_car_${RUN_TAG}"
      OPEN3D_VAL_SPLIT="$(python3 - "$DATA_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "ImageSets" / "train.txt"
print(sum(1 for line in root.read_text(encoding="utf-8").splitlines() if line.strip()))
PY
)"
      OPEN3D_CMD="
source \"$CONDA_SH\" &&
conda activate lidar-open3d-ml &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  python \"$WORKSPACE_ROOT/experiments/neilson_car/scripts/require_cuda.py\" --label open3dml --require-arch &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  python -c \"import runpy; runpy.run_path('$WORKSPACE_ROOT/experiments/neilson_car/scripts/prepare_open3dml_dataset.py', run_name='__main__')\" &&
env \
  NEILSON_CAR_KITTI_ROOT=\"$DATA_ROOT\" \
  NVCC_PREPEND_FLAGS=\"\${NVCC_PREPEND_FLAGS-}\" \
  python -c \"import runpy, sys; sys.argv=['train_open3dml_pointpillars.py', '--config', '$CONFIG_ROOT/open3dml_pointpillars_car.yml', '--data-root', '$DATA_ROOT', '--checkpoint', '$CHECKPOINT_ROOT/open3dml_pointpillars_kitti.pth', '--run-root', '$OPEN3D_RUN_ROOT', '--device', '$OPEN3D_DEVICE', '--val-split', '$OPEN3D_VAL_SPLIT', '--max-epoch', '$EPOCHS', '--num-workers', '$WORKERS']; runpy.run_path('$WORKSPACE_ROOT/experiments/neilson_car/scripts/train_open3dml_pointpillars.py', run_name='__main__')\"
"
      run_recipe "$algorithm" "lidar-open3d-ml" "$OPEN3D_CMD"
      ;;
    *)
      echo "Unsupported training target: $algorithm" >&2
      exit 2
      ;;
  esac
done

export RECIPES_JSON
write_manifest
log "wrote $MANIFEST_PATH"
log "all requested algorithms completed"

# Codex CLI Handoff For Native WSL Training Runs

## Purpose

This handoff is for a native WSL Codex session that should:

1. run the full Neilson training launcher from a real GPU-capable WSL shell
2. monitor progress from the unified log file
3. fix training bugs as they appear
4. rerun until `python3 experiments/neilson_car/scripts/train_all_algorithms.py`
   completes cleanly

Use this guide from native WSL, not from a restricted sandbox session.

## Current assumptions

- WSL2 Ubuntu is working.
- In a normal WSL shell, CUDA works in:
  - `lidar-3d-base`
  - `lidar-openpcdet`
  - `lidar-open3d-ml`
- The workspace root is:

```bash
/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics
```

- The organized workspace view also exists at:

```bash
/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics/lidar-analytics
```

- `lidar-analytics/autoware` points to the existing Autoware core workspace.

## First commands to run

Open a normal WSL shell, then:

```bash
cd "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
codex
```

Then prompt Codex with:

```text
Read setup/CODEX-CLI-HANDOFF-TRAINING.md and take over the Neilson full training run. Use the native WSL shell, launch python3 experiments/neilson_car/scripts/train_all_algorithms.py, monitor ./logs/training_log_[datetime].log, fix failures in order, and rerun until the full multi-framework training completes.
```

## What the training launcher already does

- writes the unified run log to:
  - `./logs/training_log_[datetime].log`
- launches, in order:
  1. `MMDetection3D`
  2. `OpenPCDet PointPillar`
  3. `OpenPCDet SECOND`
  4. `Open3D-ML PointPillars`
- stops immediately on the first failing framework
- writes a manifest to:
  - `experiments/neilson_car/training_manifest.json`

## Required operating rules

- Always run the real launcher from native WSL:

```bash
python3 experiments/neilson_car/scripts/train_all_algorithms.py
```

- Do not treat a restricted Codex sandbox CUDA check as authoritative.
- Trust the native WSL shell for CUDA validation.
- Fix failures one at a time in launcher order.
- After each fix:
  1. rerun the failing single-framework script first if that is faster
  2. then rerun the unified launcher
- Keep all changes inside this workspace unless a true external dependency must
  be changed.
- Do not move Autoware into detector conda envs.
- Do not build Autoware inside the detector conda envs.

## Preflight checks

Before launching training, verify all three conda envs in the native shell:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
for env in lidar-3d-base lidar-openpcdet lidar-open3d-ml; do
  echo "== $env =="
  conda activate "$env"
  python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY
done
```

If any env reports `False`, stop and fix CUDA before training.

## Expected launcher and log paths

Primary command:

```bash
python3 experiments/neilson_car/scripts/train_all_algorithms.py
```

Tail the current log in another shell:

```bash
latest_log="$(ls -1t logs/training_log_*.log | head -n 1)"
tail -f "$latest_log"
```

Search failures quickly:

```bash
rg -n "Traceback|Error|RuntimeError|CalledProcessError|IndexError|FileNotFoundError|UnpicklingError" "$latest_log"
```

## Framework-specific debugging order

### 1. MMDetection3D

Primary script:

```bash
bash experiments/neilson_car/scripts/train_mmdet3d.sh
```

Known historical issue:

- KITTI metric evaluation crashed on empty-instance validation samples

Relevant files:

- `experiments/neilson_car/scripts/prepare_mmdet_infos.py`
- `experiments/neilson_car/configs/mmdet3d_pointpillars_car.py`
- `experiments/neilson_car/scripts/train_mmdet3d.sh`

What to verify:

- info files are regenerated correctly
- validation does not run on malformed empty-instance annotations
- the run can finish all epochs and evaluation

### 2. OpenPCDet

Primary script:

```bash
bash experiments/neilson_car/scripts/train_openpcdet.sh
```

Known historical issues:

- `/tmp/openpcdet_patched` missing at runtime
- checkpoint reload after training failed because `torch.load` defaulted to
  `weights_only=True`
- noisy `gpustat` command spam in logs

Relevant files:

- `experiments/neilson_car/scripts/bootstrap_openpcdet_runtime.py`
- `experiments/neilson_car/scripts/prepare_openpcdet_infos.py`
- `experiments/neilson_car/scripts/train_openpcdet.sh`

What to verify:

- runtime patch tree is created before training
- training finishes
- post-train evaluation can reload the epoch checkpoint

### 3. Open3D-ML

Primary script:

```bash
bash experiments/neilson_car/scripts/train_open3dml.sh
```

Relevant files:

- `experiments/neilson_car/scripts/prepare_open3dml_dataset.py`
- `experiments/neilson_car/scripts/train_open3dml_pointpillars.py`
- `experiments/neilson_car/configs/open3dml_pointpillars_car.yml`

What to verify:

- `val_split` matches the real Neilson train split count
- run paths are correct after any workspace reorganization
- training output lands under `experiments/neilson_car/runs/`

## Dataset-size expectations

Current Neilson split counts:

- train: `2687`
- val: `847`
- test: `893`

If a regenerated dataset changes these counts unexpectedly, stop and inspect the
data-prep scripts before continuing.

## Fast recovery workflow

When the unified launcher fails:

1. Read the last 100 to 200 lines of the latest log.
2. Identify the failing framework.
3. Reproduce with the individual training script.
4. Fix the bug in the smallest relevant file set.
5. Re-run the single framework until it passes the previous failure point.
6. Re-run the unified launcher.

## Commands Codex should prefer

Inspect latest log:

```bash
latest_log="$(ls -1t logs/training_log_*.log | head -n 1)"
sed -n '1,200p' "$latest_log"
tail -n 200 "$latest_log"
```

MMDetection3D only:

```bash
bash experiments/neilson_car/scripts/train_mmdet3d.sh \
  experiments/neilson_car/checkpoints/mmdet3d_pointpillars_kitti.pth \
  --cfg-options train_cfg.max_epochs=24
```

OpenPCDet PointPillar only:

```bash
CONFIG_PATH=experiments/neilson_car/configs/openpcdet_pointpillar_car.yaml \
PRETRAINED_MODEL=experiments/neilson_car/checkpoints/openpcdet_pointpillar_kitti.pth \
EXTRA_TAG=neilson_pointpillar_finetune \
bash experiments/neilson_car/scripts/train_openpcdet.sh
```

OpenPCDet SECOND only:

```bash
CONFIG_PATH=experiments/neilson_car/configs/openpcdet_second_car.yaml \
PRETRAINED_MODEL=experiments/neilson_car/checkpoints/openpcdet_second_kitti.pth \
EXTRA_TAG=neilson_second_finetune \
bash experiments/neilson_car/scripts/train_openpcdet.sh
```

Open3D-ML only:

```bash
DEVICE=cuda \
CHECKPOINT_PATH=experiments/neilson_car/checkpoints/open3dml_pointpillars_kitti.pth \
RUN_ROOT=experiments/neilson_car/runs/open3dml_pointpillars_car_finetune \
bash experiments/neilson_car/scripts/train_open3dml.sh
```

## Definition of done

The takeover is complete only when:

1. `python3 experiments/neilson_car/scripts/train_all_algorithms.py` exits with code `0`
2. the latest `logs/training_log_*.log` shows all four algorithms completed
3. `experiments/neilson_car/training_manifest.json` reflects the successful run
4. the main run directories contain the expected outputs for all frameworks

## If a non-workspace external fix is required

Only step outside the repo when truly necessary, for example:

- native WSL CUDA driver/runtime problem
- broken system package required by one framework
- missing upstream source tree under `/home/datacity/src/...`

If that happens, make the smallest external change, document it in the final
response, then return to the workspace and rerun the failing stage.

# Codex CLI Handoff For New Native WSL Machine

## Purpose

This handoff is for a native WSL Codex session running on a new Windows machine with:

- RTX A5000
- Dropbox synced to `J:\Dropbox`
- WSL path `/mnt/j/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics`

Use this handoff only from a real native WSL shell with GPU access, not from a restricted sandbox.

## Target result

Leave the machine with:

1. the Git repo cloned at `~/lidar-analytics`
2. required datasets copied from `/mnt/j/...`
3. conda environments installed and working:
   - `lidar-3d-base`
   - `lidar-openpcdet`
   - `lidar-open3d-ml`
4. CUDA visible inside all required envs
5. Autoware workspace bootstrapped at `~/autoware_core_workspace`
6. local tracked Autoware patch applied
7. `codex` installed and usable from the WSL shell
8. doctor scripts passing or any residual issue documented clearly

## Read first

Read these files before taking action:

- `setup/NEW-MACHINE-WSL-J-DRIVE.md`
- `setup/README.md`
- `setup/CODEX-CLI-HANDOFF.md`
- `setup/CODEX-CLI-HANDOFF-TRAINING.md`
- `autoware_custom/README.md`

## Expected paths

Repo root:

```bash
/home/$USER/lidar-analytics
```

Dropbox root:

```bash
/mnt/j/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics
```

Autoware workspace:

```bash
/home/$USER/autoware_core_workspace
```

## Required local assets from Dropbox

Verify these exist in the repo after copy:

```bash
data/neilson_autoware_t4_intermediate
data/neilson_car_kitti
experiments/neilson_car/checkpoints
```

Optional but preferred:

```bash
data/neilson_car_kitti_subset
experiments/neilson_car/runs
logs
```

## Operating rules

- Prefer the repo clone at `~/lidar-analytics` as the working tree.
- Use Dropbox under `/mnt/j/...` as the source for large copied assets.
- Do not convert the Dropbox folder itself into the active code workspace.
- Keep Autoware in its own ROS 2 workspace.
- Do not move Autoware into the detector conda environments.
- Do not vendor the full Autoware upstream source into this repo.
- Use `autoware_custom/` for tracked local Autoware changes.

## First commands to run

```bash
cd ~/lidar-analytics
git pull
```

Verify WSL GPU visibility:

```bash
ls /dev/dxg
nvidia-smi
```

If either fails, stop and repair the host/WSL GPU setup before touching Python packages.

## Dataset and checkpoint verification

Run:

```bash
cd ~/lidar-analytics
du -sh data/neilson_autoware_t4_intermediate data/neilson_car_kitti experiments/neilson_car/checkpoints 2>/dev/null
```

Expected order of magnitude:

- `data/neilson_autoware_t4_intermediate` about 15 GB
- `data/neilson_car_kitti` about 283 MB
- `experiments/neilson_car/checkpoints` about 168 MB

If missing, copy them from:

```bash
/mnt/j/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics
```

## Environment bring-up order

1. `lidar-3d-base`
2. `lidar-openpcdet`
3. `lidar-open3d-ml`
4. `autoware_core_workspace`

### Main environment

```bash
cd ~/lidar-analytics
bash setup/bootstrap-wsl-lidar.sh --skip-apt --with-torch --with-mmdet3d
```

### OpenPCDet

```bash
cd ~/lidar-analytics
bash setup/bootstrap-wsl-openpcdet.sh
```

### Open3D-ML

```bash
cd ~/lidar-analytics
bash setup/bootstrap-wsl-lidar-suite.sh --skip-apt
```

If the suite script is redundant after prior successful steps, it is acceptable to verify the envs manually instead of rerunning everything.

## CUDA verification in all envs

Use:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
for env in lidar-3d-base lidar-openpcdet lidar-open3d-ml; do
  echo "== $env =="
  conda activate "$env"
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY
done
```

If any env reports `cuda_available False`, fix that before proceeding.

## Doctor scripts

Run:

```bash
cd ~/lidar-analytics
bash setup/wsl-doctor.sh
```

Autoware:

```bash
bash setup/bootstrap-wsl-autoware.sh --flavor core
bash autoware_custom/apply_autoware_customizations.sh
bash setup/wsl-autoware-doctor.sh --flavor core
```

## Codex install verification

Codex should be installed through npm as `@openai/codex`.

Verify:

```bash
which codex
codex --version
npm -g list --depth=0 | rg '@openai/codex'
```

If missing:

```bash
mkdir -p "$HOME/.local/npm-global"
npm config set prefix "$HOME/.local/npm-global"
grep -q 'npm-global/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/npm-global/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
npm install -g @openai/codex
```

## Final acceptance checks

The machine is ready only if all of the following are true:

1. `git status` in `~/lidar-analytics` is clean or intentionally documented
2. required datasets and checkpoints are present locally
3. `python3 train_all_algorithms.py --dry-run` succeeds from repo root
4. `wsl-doctor.sh` passes or residual warnings are understood
5. all required conda envs report CUDA available
6. `codex --version` works
7. Autoware workspace exists and the tracked patch applies cleanly

## If something fails

Capture:

- exact command
- full error output
- relevant package version
- current working directory
- whether the failing path is under `~/lidar-analytics`, `/mnt/j/...`, or `~/autoware_core_workspace`

Then fix in the smallest possible scope and continue.

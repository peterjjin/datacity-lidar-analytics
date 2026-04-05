# New Machine WSL Bring-Up Guide For `J:` Dropbox

This guide is for a new Windows machine with:

- NVIDIA RTX A5000
- Dropbox synced to `J:\Dropbox`
- a clean WSL2 Ubuntu 22.04 install target

It takes the machine from a clean Windows/WSL state through:

1. WSL and Ubuntu install
2. native WSL package install
3. Miniforge and conda setup
4. Git and Codex CLI setup
5. cloning the GitHub repo
6. copying required datasets, checkpoints, and optional run outputs from Dropbox on `J:`
7. bootstrapping the LiDAR environments
8. preparing a native WSL Codex session to finish the remaining install/debug work

This machine has an RTX A5000, so you should not hit the same bleeding-edge compatibility problems that appeared on the RTX 5090 machine. Still use the exact environment definitions and verification commands below.

## 0. Expected paths on the new machine

Windows Dropbox root:

```powershell
J:\Dropbox\_Rutgers\2026.03.14 Open3D-ML LiDAR Analytics
```

The same path inside WSL will be:

```bash
/mnt/j/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics
```

Git repo clone target inside WSL:

```bash
/home/<your-linux-user>/lidar-analytics
```

Autoware workspace target:

```bash
/home/<your-linux-user>/autoware_core_workspace
```

## 1. Windows-side prerequisites

Before opening WSL:

1. Install the latest stable NVIDIA driver for the RTX A5000.
2. Confirm Dropbox has fully synced:
   - `J:\Dropbox\_Rutgers\2026.03.14 Open3D-ML LiDAR Analytics`
3. Open an elevated PowerShell and install WSL:

```powershell
wsl --install -d Ubuntu-22.04
```

If needed, reboot Windows.

Then verify:

```powershell
wsl -l -v
```

You want to see `Ubuntu-22.04` running on WSL version `2`.

## 2. First Ubuntu launch

Launch Ubuntu 22.04 and create your Linux user.

Then run:

```bash
sudo apt update
sudo apt install -y git curl wget unzip zip build-essential
```

Confirm the `J:` drive is visible:

```bash
ls "/mnt/j/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
```

If that path does not exist, stop and fix Dropbox sync or the drive mapping first.

## 3. Clone the Git repo

Clone the synchronized code repo from GitHub into your Linux home:

```bash
cd ~
git clone https://github.com/peterjjin/datacity-lidar-analytics.git lidar-analytics
cd ~/lidar-analytics
```

Set Git identity:

```bash
git config --global user.name "peterjjin"
git config --global user.email "peter.j.jin@rutgers.edu"
git config --global credential.helper cache
git config --global init.defaultBranch main
git config --global fetch.prune true
git config --global pull.rebase false
git config --global core.autocrlf input
git config --global core.filemode false
```

## 4. Copy large local assets from Dropbox `J:`

The Git repo intentionally does not version large datasets, checkpoints, or run outputs.

Use the Dropbox folder as the source of truth for those local assets.

Set these helper variables:

```bash
export REPO_ROOT="$HOME/lidar-analytics"
export DROPBOX_ROOT="/mnt/j/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
```

### 4.1 Required datasets

These are the core folders needed for training/evaluation:

- `data/neilson_autoware_t4_intermediate` about 15 GB
- `data/neilson_car_kitti` about 283 MB

Recommended copy command:

```bash
mkdir -p "$REPO_ROOT/data"
rsync -avh --info=progress2 \
  "$DROPBOX_ROOT/data/neilson_autoware_t4_intermediate" \
  "$DROPBOX_ROOT/data/neilson_car_kitti" \
  "$REPO_ROOT/data/"
```

### 4.2 Optional subset dataset

Useful for quick smoke tests:

- `data/neilson_car_kitti_subset`

```bash
rsync -avh --info=progress2 \
  "$DROPBOX_ROOT/data/neilson_car_kitti_subset" \
  "$REPO_ROOT/data/"
```

### 4.3 Required pretrained checkpoints

These are needed if you want to start from the existing weights instead of rebuilding from scratch:

- `experiments/neilson_car/checkpoints/`

Approximate size: 168 MB

```bash
mkdir -p "$REPO_ROOT/experiments/neilson_car"
rsync -avh --info=progress2 \
  "$DROPBOX_ROOT/experiments/neilson_car/checkpoints" \
  "$REPO_ROOT/experiments/neilson_car/"
```

### 4.4 Optional run outputs

Copy these only if you want existing training artifacts and logs available locally:

- `experiments/neilson_car/runs/` about 2.5 GB
- `experiments/neilson_car/benchmark_runs*/`
- `logs/`

```bash
rsync -avh --info=progress2 \
  "$DROPBOX_ROOT/experiments/neilson_car/runs" \
  "$REPO_ROOT/experiments/neilson_car/"

rsync -avh --info=progress2 \
  "$DROPBOX_ROOT/logs" \
  "$REPO_ROOT/"
```

If you want the benchmark result folders too:

```bash
rsync -avh --info=progress2 \
  "$DROPBOX_ROOT/experiments/neilson_car/" \
  "$REPO_ROOT/experiments/neilson_car/" \
  --include='benchmark_runs*/' \
  --include='benchmark_runs*/**' \
  --exclude='*'
```

### 4.5 What not to copy by default

Usually skip these unless you explicitly need them:

- `20250410/`
- `DataCity_SMTG/`

## 5. Install Miniforge in WSL

Use the project bootstrap so the new machine matches the existing setup.

From inside the cloned repo:

```bash
cd "$REPO_ROOT"
bash setup/bootstrap-wsl-lidar.sh --with-torch
```

This will:

- install Ubuntu build tools if needed
- install Miniforge into `~/miniforge3`
- create or update the `lidar-3d-base` conda env
- install PyTorch from the configured CUDA 12.8 wheel index

After the script completes, start a fresh shell or source conda:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lidar-3d-base
```

## 6. Verify CUDA and PyTorch in WSL

Run:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY
```

You want `cuda_available True` and the device to show the RTX A5000.

Then run:

```bash
bash setup/wsl-doctor.sh
```

If `/dev/dxg` is missing or Torch reports no GPU, from Windows PowerShell run:

```powershell
wsl --shutdown
```

Then relaunch Ubuntu and repeat the checks.

## 7. Build the project environments

### 7.1 Main detector environment

Main environment:

- name: `lidar-3d-base`
- file: [`setup/lidar-3d-base-wsl.yml`](./lidar-3d-base-wsl.yml)

To install MMDetection3D too:

```bash
cd "$REPO_ROOT"
bash setup/bootstrap-wsl-lidar.sh --skip-apt --with-torch --with-mmdet3d
```

### 7.2 OpenPCDet environment

Separate environment:

- name: `lidar-openpcdet`
- file: [`setup/lidar-openpcdet-wsl.yml`](./lidar-openpcdet-wsl.yml)

Bootstrap:

```bash
cd "$REPO_ROOT"
bash setup/bootstrap-wsl-openpcdet.sh
```

Important: this prepares the env and clones OpenPCDet, but intentionally does not force every CUDA extension dependency up front.

### 7.3 Open3D-ML environment

Separate environment:

- name: `lidar-open3d-ml`
- file: [`setup/lidar-open3d-ml-wsl.yml`](./lidar-open3d-ml-wsl.yml)

Fast path:

```bash
cd "$REPO_ROOT"
bash setup/bootstrap-wsl-lidar-suite.sh --skip-apt
```

That script builds:

1. `lidar-3d-base`
2. `lidar-openpcdet`
3. `lidar-open3d-ml`

and then runs the suite checks.

## 8. Autoware workspace

Autoware should remain separate from the Python detector conda envs.

Bootstrap the separate ROS 2 workspace:

```bash
cd "$REPO_ROOT"
bash setup/bootstrap-wsl-autoware.sh --flavor core
```

Expected location:

```bash
~/autoware_core_workspace/src/autoware_core
```

Then verify:

```bash
bash setup/wsl-autoware-doctor.sh --flavor core
```

### 8.1 Apply tracked local Autoware customizations

The repo tracks only your local Autoware modifications as patches, not the full upstream Autoware source.

Apply them after the Autoware clone exists:

```bash
cd "$REPO_ROOT"
bash autoware_custom/apply_autoware_customizations.sh
```

## 9. Install Node.js, npm, and Codex CLI

This machine currently uses:

- `node v24.14.0`
- `npm 11.12.1`
- global npm prefix: `~/.local/npm-global`
- Codex package: `@openai/codex`

### 9.1 Install Node.js in WSL

If `node` is not already available in Ubuntu, use the NodeSource or distro path you prefer.

A direct apt path that is usually enough:

```bash
sudo apt update
sudo apt install -y nodejs npm
```

If that gives you an older Node version and you want the same major line as this machine, install a newer Node release first, then continue.

### 9.2 Configure npm global prefix

```bash
mkdir -p "$HOME/.local/npm-global"
npm config set prefix "$HOME/.local/npm-global"
```

Add this to `~/.bashrc` if it is not already present:

```bash
export PATH="$HOME/.local/npm-global/bin:$PATH"
```

Apply it:

```bash
source ~/.bashrc
```

### 9.3 Install Codex CLI

```bash
npm install -g @openai/codex
codex --version
```

## 10. Configure Codex

Create `~/.codex/config.toml` modeled on the existing machine:

```toml
# =======================
# Codex global settings
# =======================
approval_policy = "never"
sandbox_mode = "workspace-write"
personality = "pragmatic"
model = "gpt-5.4"
model_reasoning_effort = "medium"
use_legacy_landlock = true
service_tier = "fast"

[sandbox_workspace_write]
network_access = true
writable_roots = ["/home/<your-linux-user>", "/mnt/j/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics", "/tmp"]
exclude_slash_tmp = true

[projects."/mnt/j/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"]
trust_level = "trusted"

[projects."/home/<your-linux-user>"]
trust_level = "trusted"

[projects."/home/<your-linux-user>/lidar-analytics"]
trust_level = "trusted"

[projects."/home/<your-linux-user>/autoware_core_workspace"]
trust_level = "trusted"

[plugins."canva@openai-curated"]
enabled = true

[plugins."github@openai-curated"]
enabled = true
```

Replace `<your-linux-user>` with your actual WSL username.

## 11. Quick post-install verification

Run all of these from WSL:

```bash
cd "$REPO_ROOT"
git status
python3 train_all_algorithms.py --dry-run
bash setup/wsl-doctor.sh
codex --version
```

Check the conda envs too:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
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

## 12. Hand off to native WSL Codex

After the manual steps above, start Codex in the repo:

```bash
cd "$REPO_ROOT"
codex
```

Then prompt it with:

```text
Read setup/CODEX-CLI-HANDOFF-NEW-MACHINE.md and finish the new-machine native WSL environment setup using the J: Dropbox mirror. Verify the three conda envs, verify CUDA in each env, verify the copied datasets and checkpoints, bootstrap the separate Autoware workspace, apply the tracked Autoware patch, and run the available doctor scripts. Fix any package or path issue you find and leave the machine ready for training.
```

## 13. Summary of what to copy from Dropbox

Must copy:

- `data/neilson_autoware_t4_intermediate`
- `data/neilson_car_kitti`
- `experiments/neilson_car/checkpoints`

Recommended optional copy:

- `data/neilson_car_kitti_subset`
- `experiments/neilson_car/runs`
- `logs`
- `experiments/neilson_car/benchmark_runs*`

Normally skip:

- `20250410`
- `DataCity_SMTG`


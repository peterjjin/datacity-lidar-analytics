# Codex CLI Handoff For Native WSL Setup

## Current verified state

- WSL2 Ubuntu is running correctly.
- Native WSL `codex` is installed and active:
  - `which codex` -> `/usr/bin/codex`
  - `codex --version` -> `codex-cli 0.115.0`
- Miniforge exists at `/home/datacity/miniforge3`.
- Conda env `lidar-3d-base` exists.
- In a normal WSL shell, PyTorch sees the GPU:
  - `torch 2.10.0+cu128`
  - `torch.cuda.is_available()` -> `True`
  - GPU -> `NVIDIA GeForce RTX 5090`
- `mmdet3d` is not installed yet.

## What blocked the install from the restricted Codex session

The existing `MMDetection3D` bootstrap progressed until `mmcv`.

Current issue:

- No matching prebuilt `mmcv` wheel was found for the current stack:
  - `torch 2.10.0`
  - `cu128`
- `mmcv` fell back to a source build.
- The source build failed because WSL does not currently have a CUDA toolkit installed:
  - `CUDA_HOME` was unset
  - no `/usr/local/cuda*` path existed
  - `nvcc` was not present

This means the next step is to install a CUDA toolkit inside WSL, then rerun the `MMDetection3D` bootstrap.

## Files already prepared in this workspace

- `setup/bootstrap-wsl-lidar.sh`
  - now emits a clearer error when `mmcv` needs CUDA toolkit build support
- `setup/PACKAGE-LANDSCAPE.md`
  - explains the install order and package roles
- `setup/bootstrap-wsl-openpcdet.sh`
  - separate scaffold for an isolated `OpenPCDet` environment
- `setup/lidar-openpcdet-wsl.yml`
  - base env file for `OpenPCDet`
- `setup/lidar-open3d-ml-wsl.yml`
  - separate env file for `Open3D-ML`
- `setup/bootstrap-wsl-lidar-suite.sh`
  - one entry point to build the MMDetection3D, OpenPCDet, and Open3D-ML test environments

## Recommended package order

1. Finish `MMDetection3D`
2. Create separate `OpenPCDet` env
3. Add `Open3D-ML` only if needed for auxiliary experiments
4. Keep `Autoware` in a separate ROS 2 workspace

## Separate Autoware workspace support

This repo also includes a separate Autoware bootstrap path:

```bash
bash setup/bootstrap-wsl-autoware.sh --flavor core
```

or:

```bash
bash setup/bootstrap-wsl-autoware.sh --flavor main
```

Verification:

```bash
bash setup/wsl-autoware-doctor.sh --flavor core
```

## What Codex CLI should do next inside native WSL

Run Codex CLI from this workspace:

```bash
cd "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
codex
```

Then instruct it to continue from this exact state:

```text
Read setup/CODEX-CLI-HANDOFF.md, then continue the native WSL deployment for LiDAR 3D detection packages. Start with MMDetection3D. Install whatever is still missing inside WSL, including a CUDA toolkit if mmcv needs to build from source. Verify the install with imports and keep OpenPCDet in a separate environment.
```

## Exact commands Codex CLI can use

### 1. Verify the current Python/GPU state

```bash
source /home/datacity/miniforge3/etc/profile.d/conda.sh
conda activate lidar-3d-base
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### 2. Check for CUDA toolkit in WSL

```bash
which nvcc
ls -d /usr/local/cuda*
```

### 3. Preferred route: install CUDA toolkit in WSL if missing

If `sudo` is available in the native shell, Codex CLI can use either:

```bash
sudo apt-get update
```

plus the appropriate CUDA toolkit install path for Ubuntu 22.04,

or a Conda-based toolkit install if it decides to keep the toolkit isolated to the env.

The key requirement is:

- `nvcc` must be present
- `CUDA_HOME` must resolve to a valid toolkit root

### 4. Retry MMDetection3D install

```bash
cd "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
bash setup/bootstrap-wsl-lidar.sh --skip-apt --with-mmdet3d
```

### 5. Verify MMDetection3D

```bash
source /home/datacity/miniforge3/etc/profile.d/conda.sh
conda activate lidar-3d-base
python -c "import mmdet3d; print(mmdet3d.__version__)"
bash setup/wsl-doctor.sh
```

## If MMDetection3D still fails at mmcv

Have Codex CLI capture:

- `torch.__version__`
- `torch.version.cuda`
- `python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"`
- `which nvcc`
- full `mmcv` build error

Then adjust the toolkit path or package pins from there.

## After MMDetection3D succeeds

Move to the isolated `OpenPCDet` env:

```bash
cd "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
bash setup/bootstrap-wsl-openpcdet.sh
```

Do not merge `OpenPCDet` into `lidar-3d-base`.

# Codex CLI Handoff For Native WSL Autoware Setup

## Current verified state

The non-ROS LiDAR environments are working:

- `lidar-3d-base`
  - `torch 2.10.0+cu128`
  - CUDA works on `NVIDIA GeForce RTX 5090`
  - `mmdet3d 1.4.0` imports
- `lidar-openpcdet`
  - `torch 2.10.0+cu128`
  - CUDA works
  - `pcdet 0.6.0+233f849` imports
- `lidar-open3d-ml`
  - `open3d 0.19.0`
  - `open3d.ml` imports
  - `torch 2.10.0+cu128`
  - CUDA works

Autoware bootstrap is also mostly complete:

- ROS 2 Humble is installed at `/opt/ros/humble`
- `rosdep`, `colcon`, and `vcs` are installed
- `autoware_core` repo is cloned at:
  - `/home/datacity/autoware_core_workspace/src/autoware_core`
- `rosdep install` for the workspace succeeded

## Current blocker

Autoware build fails because `colcon` is using the Conda Python from `lidar-3d-base` instead of the system ROS Python.

Observed failing interpreter in the log:

```text
/home/datacity/miniforge3/envs/lidar-3d-base/bin/python3
```

Observed failure:

```text
ModuleNotFoundError: No module named 'catkin_pkg'
```

This is not an Autoware package bug. It is a build environment contamination issue.

## Root cause

Autoware was built from a shell that still had Conda active, or the workspace cached the Conda interpreter in `build/`.

Autoware must be built from a clean WSL shell, not from `(lidar-3d-base)` or any Conda environment.

## What has already been patched locally

- `setup/bootstrap-wsl-autoware.sh`
  - fixes ROS apt repo and key handling
  - uses `python3-rosdep2`
  - safely sources ROS setup under `set -u`
  - now warns in next steps to deactivate Conda
- `setup/README.md`
  - explicitly says to build Autoware from a non-Conda shell

## What native Codex CLI should do next

Run from a clean WSL shell:

```bash
cd "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
codex
```

Then give it this prompt:

```text
Read setup/CODEX-CLI-HANDOFF-AUTOWARE.md and continue the native WSL Autoware setup. Fix the current build failure by ensuring colcon uses the system ROS Python instead of the Conda Python, clean the cached build artifacts, rebuild autoware_core, and verify the workspace with setup/wsl-autoware-doctor.sh.
```

## Exact recovery steps

### 1. Confirm shell is clean

```bash
conda deactivate
conda deactivate
echo "${CONDA_PREFIX:-}"
which python3
```

Expected:

- `CONDA_PREFIX` empty
- `which python3` -> `/usr/bin/python3`

### 2. Remove cached build state

```bash
cd /home/datacity/autoware_core_workspace
rm -rf build install log
```

### 3. Source ROS and verify Python again

```bash
source /opt/ros/humble/setup.bash
echo "${CONDA_PREFIX:-}"
which python3
python3 -c "import catkin_pkg; print(catkin_pkg.__file__)"
```

Expected:

- `CONDA_PREFIX` empty
- `which python3` -> `/usr/bin/python3`
- `catkin_pkg` import succeeds

### 4. Rebuild Autoware core

```bash
cd /home/datacity/autoware_core_workspace
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### 5. Verify workspace

```bash
bash "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics/setup/wsl-autoware-doctor.sh" --flavor core
```

## If it still picks the Conda Python

Have Codex CLI capture:

```bash
echo "${CONDA_PREFIX:-}"
env | grep -E 'CONDA|PYTHON|AMENT|CMAKE_PREFIX_PATH'
which python3
```

Then adjust the shell/session startup so ROS build commands are run from a clean shell.

## Do not do this

- Do not install `catkin_pkg` into `lidar-3d-base` as a workaround.
- Do not build Autoware inside a Conda environment.
- Do not merge Autoware into the detector Conda envs.

# VS Code Codex Handoff For Native WSL Autoware Setup

## Date

- 2026-03-20

## Current verified state

- ROS 2 Humble is installed at `/opt/ros/humble`.
- `rosdep`, `colcon`, `vcs`, and `git` are available.
- `autoware_core` is present at:
  - `/home/datacity/autoware_core_workspace/src/autoware_core`
- `rosdep` is initialized.
- The original Conda contamination issue is understood and reproducible from the old cached workspace state.

## What was verified in this session

In a clean shell with Conda variables removed:

- `CONDA_PREFIX` was empty
- `which python3` returned `/usr/bin/python3`
- `python3 -c "import catkin_pkg; print(catkin_pkg.__file__)"`
  succeeded and resolved to:
  - `/usr/lib/python3/dist-packages/catkin_pkg/__init__.py`

This confirms the original failure was environment contamination, not an Autoware package bug.

## What blocked in this restricted Codex session

This session could read `/home/datacity/autoware_core_workspace` but could not modify it.

Observed sandbox limitation:

- renaming or deleting `/home/datacity/autoware_core_workspace/build`
- renaming or deleting `/home/datacity/autoware_core_workspace/install`
- renaming or deleting `/home/datacity/autoware_core_workspace/log`
- running `colcon build` in that workspace, because it needs to create new log/build artifacts there

So the in-place recovery could not be completed from this restricted session.

## Important forensic result

The old cached workspace still contains Conda-tainted build metadata.

Examples found under `/home/datacity/autoware_core_workspace/build/*/colcon_command_prefix_build.sh.env`:

- `PATH` still included `/home/datacity/miniforge3/condabin`
- `XML_CATALOG_FILES` pointed at `.../miniforge3/envs/lidar-3d-base/...`
- `_CONDA_EXE=/home/datacity/miniforge3/bin/conda`
- `_CONDA_ROOT=/home/datacity/miniforge3`

That cached state must be removed before the real in-place rebuild.

## Clean out-of-tree validation that was completed

Because the original workspace was read-only here, a clean mirror workspace was created at:

- `/tmp/autoware_core_workspace_clean`

Layout:

- `/tmp/autoware_core_workspace_clean/src/autoware_core`
  - symlink to `/home/datacity/autoware_core_workspace/src/autoware_core`

The rebuild there was started with:

- Conda variables unset
- `PYTHONHOME`, `PYTHONPATH`, and `XML_CATALOG_FILES` unset
- `PATH` reduced to system and ROS paths
- ROS sourced from `/opt/ros/humble/setup.bash`

Observed result:

- `colcon` used `/usr/bin/python3`
- `catkin_pkg` imported from system ROS Python
- no `ModuleNotFoundError`
- no `No module named 'catkin_pkg'`
- no Python/CMake errors were found in the live build logs

Build progress reached at least:

- `56/70` packages completed successfully during this session

Packages visibly installed in the clean workspace included:

- `autoware_core_control`
- `autoware_core_perception`
- `autoware_core_sensing`
- `autoware_core_vehicle`

## Doctor verification already completed

This command passed:

```bash
bash "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics/setup/wsl-autoware-doctor.sh" \
  --workspace /tmp/autoware_core_workspace_clean \
  --flavor core
```

Result:

- `Summary: 9 pass, 0 warn`

This verifies the clean test workspace layout and toolchain, but it does not replace the needed in-place cleanup of `/home/datacity/autoware_core_workspace`.

## What the external VS Code Codex agent should do next

Run from a normal unrestricted WSL shell or VS Code terminal, not from a Conda-activated prompt.

### 1. Confirm the shell is clean

```bash
conda deactivate || true
conda deactivate || true
echo "${CONDA_PREFIX:-}"
which python3
python3 -c "import sys; print(sys.executable)"
```

Expected:

- empty `CONDA_PREFIX`
- `/usr/bin/python3`

### 2. Remove the contaminated cached workspace state in place

```bash
cd /home/datacity/autoware_core_workspace
rm -rf build install log
```

### 3. Source ROS and verify system Python again

```bash
source /opt/ros/humble/setup.bash
echo "${CONDA_PREFIX:-}"
which python3
python3 -c "import sys, catkin_pkg; print(sys.executable); print(catkin_pkg.__file__)"
```

Expected:

- empty `CONDA_PREFIX`
- `/usr/bin/python3`
- successful `catkin_pkg` import from `/usr/lib/python3/dist-packages`

### 4. Rebuild the real workspace in place

```bash
cd /home/datacity/autoware_core_workspace
env -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    -u CONDA_EXE \
    -u _CONDA_ROOT \
    -u _CONDA_EXE \
    -u _CE_M \
    -u _CE_CONDA \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u XML_CATALOG_FILES \
    PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    bash -lc 'source /opt/ros/humble/setup.bash && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'
```

### 5. Verify the real workspace

```bash
bash "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics/setup/wsl-autoware-doctor.sh" --flavor core
```

## If the in-place build still picks Conda Python

Capture:

```bash
echo "${CONDA_PREFIX:-}"
env | grep -E 'CONDA|PYTHON|AMENT|CMAKE_PREFIX_PATH|XML_CATALOG_FILES'
which python3
python3 -c "import sys; print(sys.executable)"
```

Also inspect:

```bash
rg -n "/home/datacity/miniforge3|lidar-3d-base|_CONDA_|XML_CATALOG_FILES|PYTHON_EXECUTABLE" \
  -S /home/datacity/autoware_core_workspace/build /home/datacity/autoware_core_workspace/log
```

## Do not do this

- Do not install `catkin_pkg` into the Conda env as a workaround.
- Do not build Autoware from `(lidar-3d-base)` or any active Conda env.
- Do not merge Autoware into the detector Conda environments.

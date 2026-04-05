# LiDAR 3D Package Landscape

## Current local history

The prior setup work in this workspace already converged on this order:

1. `MMDetection3D` first in WSL for detector testing.
2. `OpenPCDet` second in a separate WSL environment.
3. `Autoware` in its own Ubuntu 22.04 + ROS 2 Humble workspace.
4. `Open3D-ML` as a lighter auxiliary environment for experiments, not the main detector stack.

This is reflected in:

- [`README.md`](./README.md)
- [`NEXT-STEPS-WSL.md`](./NEXT-STEPS-WSL.md)
- [`bootstrap-wsl-lidar.sh`](./bootstrap-wsl-lidar.sh)

## Package roles

### MMDetection3D

- General-purpose OpenMMLab 3D detection framework.
- Good first target for vehicle and pedestrian detector benchmarking.
- Strong model/config ecosystem, including `CenterPoint`, `PointPillars`, and `SECOND`.
- Current blocker on this machine: no matching prebuilt `mmcv` wheel for `torch 2.10.0 + cu128`, so `mmcv` falls back to source build.

### OpenPCDet

- Focused LiDAR 3D detection toolbox with widely used reference implementations.
- Valuable for cross-checking detector behavior against another ecosystem.
- More fragile on very new GPU/toolchain combinations because of `spconv` and CUDA extension dependencies.
- Keep this in a separate WSL environment from `MMDetection3D`.

### Autoware

- Full autonomy stack, not just a Python detector library.
- Relevant when the goal is integration into a robotics/AV pipeline rather than standalone detector benchmarking.
- Requires its own Ubuntu 22.04 + ROS 2 Humble workspace.
- Do not merge it into the detector Conda environments.
- This workspace now includes a separate WSL bootstrap for `autoware_core` or the full `autoware` repository.

### Open3D-ML

- 3D ML toolkit oriented toward point clouds and broader 3D tasks.
- Useful for preprocessing, experimentation, and some detection workflows.
- Not the strongest primary target for vehicle/pedestrian detector bake-offs compared with `MMDetection3D` and `OpenPCDet`.

## Recommended setup order on this machine

1. Finish `MMDetection3D`.
2. Create a separate `OpenPCDet` environment.
3. Add an `Open3D-ML` auxiliary environment if needed.
4. Keep `Autoware` separate and install only when you need ROS integration.

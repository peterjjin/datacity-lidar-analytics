# Remaining WSL Steps for LiDAR 3D Detector Testing

## Current state observed on 2026-03-16

- WSL2 Ubuntu is running.
- `miniforge3` and the `lidar-3d-base` conda env already exist.
- `torch 2.10.0+cu128` imports in WSL.
- `/dev/dxg` is missing, so WSL GPU passthrough is not active in this session.
- `cmd.exe` and `powershell.exe` are on `PATH`, but Windows interop execution fails.
- `mmdet3d` is not installed yet in the active env.

## What must be fixed outside this workspace

Run these from Windows, not from the sandboxed shell:

```powershell
cd "d:\Dropbox\_Rutgers\2026.03.14 Open3D-ML LiDAR Analytics"
powershell -ExecutionPolicy Bypass -File .\setup\wsl-host-check.ps1
wsl --shutdown
```

Then reboot Windows if `/dev/dxg` is still missing after the shutdown/restart cycle.

After Ubuntu is back up, verify:

```bash
cmd.exe /c ver
ls -l /dev/dxg
bash "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics/setup/wsl-doctor.sh"
```

You want these outcomes before model testing:

- `cmd.exe /c ver` succeeds
- `/dev/dxg` exists
- `python -c "import torch; print(torch.cuda.is_available())"` prints `True`

## Finish the detector stack

From a normal Ubuntu WSL terminal:

```bash
cd "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
bash setup/bootstrap-wsl-lidar.sh --with-mmdet3d
bash setup/wsl-doctor.sh
```

If the apt packages are already present and you only need the Python side:

```bash
bash setup/bootstrap-wsl-lidar.sh --skip-apt --with-mmdet3d
```

## Run vehicle and pedestrian model tests

List the starter model presets:

```bash
bash setup/run-mmdet3d-model-test.sh --list-presets
```

Typical first tests:

```bash
bash setup/run-mmdet3d-model-test.sh \
  --config configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --checkpoint /path/to/centerpoint_nuscenes.pth
```

```bash
bash setup/run-mmdet3d-model-test.sh \
  --config configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
  --checkpoint /path/to/pointpillars_kitti_3class.pth
```

## Recommended order

1. Restore Windows interop and `/dev/dxg`.
2. Confirm PyTorch CUDA works inside WSL.
3. Install `mmdet3d`.
4. Start with `CenterPoint` on nuScenes and `PointPillars` or `SECOND` on KITTI 3-class.
5. Keep `OpenPCDet` in a separate env only after the MMDetection3D path is stable.

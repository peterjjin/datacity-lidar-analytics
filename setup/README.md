# LiDAR 3D Detection Setup on This Windows Machine

## Bottom line

Yes: for **OpenPCDet**, **MMDetection3D**, and **Autoware**, you should set up **WSL2** on this machine.

No: you do **not** need WSL2 for **VeloView** or for a lightweight native Windows environment used for PCAP inspection, point-cloud export, and basic Open3D/Open3D-ML experiments.

## What I verified locally on 2026-03-14

- `conda` is already installed: `conda 24.9.2`
- `WSL` is **not** installed yet
- GPU: `NVIDIA GeForce RTX 5090`
- Driver: `576.88`

## Recommended split

- **Native Windows**
  - Use for: `VeloView`, PCAP inspection, quick point-cloud export, notebooks, and light preprocessing
  - Environment file: [lidar-win-prep.yml](./lidar-win-prep.yml)
- **WSL2 Ubuntu 22.04**
  - Use for: `MMDetection3D`, `OpenPCDet`, `Open3D-ML`, most serious detector testing, and any custom CUDA/C++ extension builds
  - Environment file: [lidar-3d-base-wsl.yml](./lidar-3d-base-wsl.yml)
  - Suite bootstrap: [bootstrap-wsl-lidar-suite.sh](./bootstrap-wsl-lidar-suite.sh)
- **Autoware**
  - Run in its own Ubuntu/ROS 2 workspace
  - Do not try to fold it into the same conda env as the Python detector frameworks

## Why WSL2 is the right call

- The official **OpenPCDet** requirements page says it is tested on **Linux** and still lists an older stack: Python 3.6+, PyTorch 1.1 to 1.10, CUDA 10.0 to 11.3, and `spconv-cu113`.
- The official **MMCV** install docs warn that Windows can have compiler compatibility issues.
- The official **Autoware** installation docs target **Ubuntu 22.04**.
- Your GPU is a very new generation. Trying to force legacy LiDAR detection stacks into native Windows is the slow path.

## Why Ubuntu 22.04

`Ubuntu 22.04` is the version to standardize on for this project.

- `Autoware` explicitly targets Ubuntu 22.04.
- `MMDetection3D` and `OpenPCDet` need Linux, but do not force a stricter Ubuntu pin than that for the path we are using here.
- Using one common Ubuntu version avoids splitting the toolchain across distros.

## Fast path

From an elevated PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup\install-wsl-ubuntu.ps1 -Install -Distro Ubuntu-22.04
```

After Ubuntu launches and you create your Linux user:

```bash
bash /mnt/d/Dropbox/_Rutgers/2026.03.14\ Open3D-ML\ LiDAR\ Analytics/setup/bootstrap-wsl-lidar.sh
```

If you also want PyTorch and MMDetection3D installed in one pass:

```bash
bash /mnt/d/Dropbox/_Rutgers/2026.03.14\ Open3D-ML\ LiDAR\ Analytics/setup/bootstrap-wsl-lidar.sh --with-mmdet3d
```

If you want all test environments prepared in sequence:

```bash
bash /mnt/d/Dropbox/_Rutgers/2026.03.14\ Open3D-ML\ LiDAR\ Analytics/setup/bootstrap-wsl-lidar-suite.sh
```

Then verify:

```bash
bash /mnt/d/Dropbox/_Rutgers/2026.03.14\ Open3D-ML\ LiDAR\ Analytics/setup/wsl-doctor.sh
```

If PyTorch later reports `cuda_available False` or the doctor reports `/dev/dxg` missing, run this from Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup\wsl-host-check.ps1
```

## Step 1: Install WSL2

Run this in an elevated PowerShell window, or use [install-wsl-ubuntu.ps1](./install-wsl-ubuntu.ps1):

```powershell
wsl --install -d Ubuntu-22.04
```

Then reboot if Windows asks you to.

After reboot, confirm the distro is present:

```powershell
wsl -l -v
```

If `Ubuntu-22.04` is not listed yet, run:

```powershell
wsl --install -d Ubuntu-22.04
```

Then open Ubuntu 22.04 and create your Linux user.

## Step 2: Install Linux build tools inside WSL

You can do this manually, or just run [bootstrap-wsl-lidar.sh](./bootstrap-wsl-lidar.sh).

Inside Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential git git-lfs curl wget cmake ninja-build pkg-config \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 python3-dev
```

## Step 3: Install Miniforge inside WSL

Inside Ubuntu:

```bash
mkdir -p ~/tmp
cd ~/tmp
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda init bash
exec bash
```

## Step 4: Create the WSL base environment

You can do this manually, or let [bootstrap-wsl-lidar.sh](./bootstrap-wsl-lidar.sh) do it for you.

From Ubuntu, using your shared Windows workspace mounted under `/mnt/d`:

```bash
cd "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics"
conda env create -f setup/lidar-3d-base-wsl.yml
conda activate lidar-3d-base
```

## Step 5: Install PyTorch in WSL

Use the **current official PyTorch selector** when you run this, not an old blog post or an old README command. On a new GPU generation, that matters.

Official selector:

- https://pytorch.org/get-started/locally/

After installing PyTorch, verify CUDA is visible:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

If that prints `False` in WSL, the Linux env is present but GPU passthrough is not active yet. From Windows PowerShell:

```powershell
wsl --shutdown
```

Then reboot Windows, launch Ubuntu again, and rerun [wsl-doctor.sh](./wsl-doctor.sh). If `/dev/dxg` is still missing, run [wsl-host-check.ps1](./wsl-host-check.ps1) and use that output to diagnose the Windows side.

## Step 6: Install MMDetection3D in WSL

You can do this manually, or run:

```bash
bash /mnt/d/Dropbox/_Rutgers/2026.03.14\ Open3D-ML\ LiDAR\ Analytics/setup/bootstrap-wsl-lidar.sh --with-mmdet3d
```

This is the first detector stack I would test on your machine.

```bash
conda activate lidar-3d-base
pip install --upgrade pip wheel "setuptools<82"
pip install -U openmim
mim install "mmengine>=0.8.0"
MMCV_WITH_OPS=1 pip install -v --no-build-isolation "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.0.0,<3.4.0"
git clone https://github.com/open-mmlab/mmdetection3d.git ~/src/mmdetection3d
cd ~/src/mmdetection3d
pip install -v -e .
python -c "import mmdet3d; print(mmdet3d.__version__)"
```

If `mim install mmcv` falls back to a source tarball for a newer Torch or CUDA combination, pin `setuptools<82` and disable build isolation for that step. `mmcv 2.1.0` still imports `pkg_resources` in `setup.py`, so a too-new isolated build env can fail before compilation starts.

## Step 7: Treat OpenPCDet as secondary on this hardware

I would **not** start with OpenPCDet on this RTX 5090.

Reason:

- its official requirements are still anchored to an older CUDA/PyTorch stack
- its install docs still reference `spconv-cu113`
- that makes it a poorer first target on a very new GPU generation

If you want OpenPCDet after MMDetection3D is working, do it in a **separate WSL env**, not the same one.

Bootstrap command:

```bash
bash /mnt/d/Dropbox/_Rutgers/2026.03.14\ Open3D-ML\ LiDAR\ Analytics/setup/bootstrap-wsl-openpcdet.sh
```

This script creates the env, installs PyTorch, and clones `OpenPCDet`, but intentionally stops short of forcing `spconv` or other CUDA-extension pins until you confirm the current compatible upstream path for your GPU/toolchain.

## Step 8: Add Open3D-ML as an auxiliary WSL environment

If you want a clean environment for Open3D visualization and Open3D-ML experiments instead of mixing those packages into the detector stacks:

```bash
conda env create -f setup/lidar-open3d-ml-wsl.yml
conda activate lidar-open3d-ml
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import open3d; import open3d.ml; import torch; print(open3d.__version__); print(torch.cuda.is_available())"
```

## Step 9: Keep Autoware separate

For `autoware_lidar_centerpoint` and related Autoware components:

- use Ubuntu 22.04 + ROS 2 Humble
- keep it in an Autoware workspace
- do not try to install it into the `lidar-3d-base` conda env

Workspace bootstrap scripts in this repo:

```bash
bash setup/bootstrap-wsl-autoware.sh --flavor core
```

or for the full repository:

```bash
bash setup/bootstrap-wsl-autoware.sh --flavor main
```

Then verify:

```bash
bash setup/wsl-autoware-doctor.sh --flavor core
```

Build Autoware from a clean WSL shell, not from an active Conda environment. If your prompt shows something like `(lidar-3d-base)`, run `conda deactivate` until no Conda env is active before `source /opt/ros/humble/setup.bash` and `colcon build`.

Official docs:

- https://autowarefoundation.github.io/autoware-documentation/latest/installation/autoware/source-installation/
- https://autowarefoundation.github.io/autoware-documentation/main/installation/autoware/core-source-installation/

## Native Windows environment

For preprocessing and lightweight testing on Windows:

```powershell
cd "d:\Dropbox\_Rutgers\2026.03.14 Open3D-ML LiDAR Analytics"
conda env create -f setup\lidar-win-prep.yml
conda activate lidar-win-prep
python -c "import open3d; print(open3d.__version__)"
```

Use this env for:

- loading and converting derived point-cloud files
- notebooks
- Open3D visualization
- general preprocessing around the detector pipeline

## VS Code

This workspace now includes [extensions.json](../.vscode/extensions.json) with recommended extensions for:

- Remote WSL
- Python
- Jupyter

Open the folder through VS Code's `Remote - WSL` extension after Ubuntu is ready.

## What I would do first on this machine

1. Install WSL2 Ubuntu 22.04.
2. Bring up `lidar-3d-base`.
3. Get PyTorch working with GPU inside WSL.
4. Install and smoke-test `MMDetection3D`.
5. Keep `VeloView` and Windows-side preprocessing native.
6. Only try `OpenPCDet` after the MMDetection3D path is stable.

## Package order on this machine

See [PACKAGE-LANDSCAPE.md](./PACKAGE-LANDSCAPE.md) for the current package roles and why the install order is:

1. `MMDetection3D`
2. `OpenPCDet` in a separate env
3. `Open3D-ML` in a separate auxiliary env if needed
4. `Autoware` in a separate ROS 2 workspace

## Primary sources

- OpenPCDet requirements: https://github.com/open-mmlab/OpenPCDet
- MMDetection3D install guide: https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/docs/en/get_started.md
- MMCV install notes: https://mmcv.readthedocs.io/en/latest/get_started/installation.html
- Autoware source installation: https://autowarefoundation.github.io/autoware-documentation/latest/installation/autoware/source-installation/
- Autoware Core source installation: https://autowarefoundation.github.io/autoware-documentation/main/installation/autoware/core-source-installation/
- VeloView releases: https://github.com/Kitware/VeloView/releases
- PyTorch install selector: https://pytorch.org/get-started/locally/

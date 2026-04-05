#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


SOURCE_ROOT = Path("/home/datacity/src/OpenPCDet")
PATCHED_ROOT = Path("/tmp/openpcdet_patched")
EXTRA_PY_ROOT = Path("/tmp/openpcdet_extra_py")
MARKER_PATH = PATCHED_ROOT / ".codex_bootstrap_marker"


PATCHED_DATASETS_INIT = """import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
from .once.once_dataset import ONCEDataset
from .custom.custom_dataset import CustomDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset,
    'ONCEDataset': ONCEDataset,
    'CustomDataset': CustomDataset,
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler
"""


def patch_kitti_dataset(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    header = "\n".join(text.splitlines()[:6])
    if "from pathlib import Path" not in header:
        text = text.replace("import pickle\n", "import pickle\nfrom pathlib import Path\n", 1)
    noisy_lines = [
        "            print('%s sample_idx: %s' % (self.split, sample_idx))\n",
        "            print('gt_database sample: %d/%d' % (k + 1, len(infos)))\n",
    ]
    for noisy in noisy_lines:
        if noisy in text:
            text = text.replace(noisy, "")
    path.write_text(text, encoding="utf-8")


def patch_detector3d_template(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "torch.load(filename, map_location=loc_type)": "torch.load(filename, map_location=loc_type, weights_only=False)",
        "torch.load(pre_trained_path, map_location=loc_type)": "torch.load(pre_trained_path, map_location=loc_type, weights_only=False)",
        "torch.load(optimizer_filename, map_location=loc_type)": "torch.load(optimizer_filename, map_location=loc_type, weights_only=False)",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    path.write_text(text, encoding="utf-8")


def patch_test_py(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)\n",
        "        model.load_params_from_file(\n"
        "            filename=cur_ckpt,\n"
        "            logger=logger,\n"
        "            to_cpu=dist_test,\n"
        "            pre_trained_path=args.pretrained_model,\n"
        "        )\n",
    )
    path.write_text(text, encoding="utf-8")


def patch_train_py(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    helper = """
def _freeze_modules_from_env(model, logger):
    raw_value = os.environ.get('OPENPCDET_FREEZE_MODULES', '').strip()
    if not raw_value:
        return []

    frozen = []
    for module_name in [item.strip() for item in raw_value.split(',') if item.strip()]:
        module = getattr(model, module_name, None)
        if module is None:
            logger.warning('Requested freeze for unknown module: %s', module_name)
            continue

        module.requires_grad_(False)
        module.eval()
        frozen.append(module_name)

    if frozen:
        logger.info('Frozen modules via OPENPCDET_FREEZE_MODULES: %s', ', '.join(frozen))
    return frozen


"""
    if "def _freeze_modules_from_env(model, logger):" not in text:
        text = text.replace("def main():\n", helper + "def main():\n", 1)

    optimizer_line = "    optimizer = build_optimizer(model, cfg.OPTIMIZATION)\n"
    freeze_line = "    frozen_modules = _freeze_modules_from_env(model, logger)\n"
    for line in (optimizer_line, freeze_line):
        while line in text:
            text = text.replace(line, "", 1)
    marker = """    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

"""
    injected = marker + freeze_line + optimizer_line + "\n"
    if marker in text:
        text = text.replace(marker, injected, 1)

    duplicate_eval = """    for module_name in frozen_modules:
        getattr(model, module_name).eval()
    for module_name in frozen_modules:
        getattr(model, module_name).eval()
"""
    single_eval = """    for module_name in frozen_modules:
        getattr(model, module_name).eval()
"""
    if duplicate_eval in text:
        text = text.replace(duplicate_eval, single_eval, 1)
    elif single_eval not in text:
        before = """    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
"""
        after = """    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    for module_name in frozen_modules:
        getattr(model, module_name).eval()
"""
        if before in text:
            text = text.replace(before, after, 1)

    path.write_text(text, encoding="utf-8")


def patch_train_utils(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    noisy = "                        gpu_info = os.popen('gpustat').read()\n                        logger.info(gpu_info)\n"
    quiet = "                        if shutil.which('gpustat'):\n                            gpu_info = os.popen('gpustat').read()\n                            logger.info(gpu_info)\n"
    if "import shutil" not in text:
        text = text.replace("import os\n", "import os\nimport shutil\n", 1)
    if noisy in text:
        text = text.replace(noisy, quiet)
    path.write_text(text, encoding="utf-8")


def patch_common_utils(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    before = """def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \\
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask
"""
    after = """def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \\
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \\
           & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])
    return mask
"""
    if before in text:
        text = text.replace(before, after)
    path.write_text(text, encoding="utf-8")


def patch_data_processor(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "def _generate_voxels_numpy" not in text:
        marker = "class VoxelGeneratorWrapper():\n"
        helper = """class VoxelGeneratorWrapper():
    @staticmethod
    def _generate_voxels_numpy(points, voxel_size, coors_range, max_num_points_per_voxel, max_num_voxels):
        points = np.asarray(points, dtype=np.float32)
        voxel_size = np.asarray(voxel_size, dtype=np.float32)
        coors_range = np.asarray(coors_range, dtype=np.float32)
        grid_size = np.floor((coors_range[3:6] - coors_range[0:3]) / voxel_size).astype(np.int32)

        if points.size == 0:
            empty_voxels = np.zeros((0, max_num_points_per_voxel, points.shape[-1] if points.ndim == 2 else 0), dtype=np.float32)
            empty_coords = np.zeros((0, 3), dtype=np.int32)
            empty_num_points = np.zeros((0,), dtype=np.int32)
            return empty_voxels, empty_coords, empty_num_points

        point_mask = (
            (points[:, 0] >= coors_range[0]) & (points[:, 0] < coors_range[3]) &
            (points[:, 1] >= coors_range[1]) & (points[:, 1] < coors_range[4]) &
            (points[:, 2] >= coors_range[2]) & (points[:, 2] < coors_range[5])
        )
        points = points[point_mask]
        if points.shape[0] == 0:
            empty_voxels = np.zeros((0, max_num_points_per_voxel, points.shape[-1]), dtype=np.float32)
            empty_coords = np.zeros((0, 3), dtype=np.int32)
            empty_num_points = np.zeros((0,), dtype=np.int32)
            return empty_voxels, empty_coords, empty_num_points

        point_coords = np.floor((points[:, :3] - coors_range[:3]) / voxel_size).astype(np.int32)
        coord_mask = (
            (point_coords[:, 0] >= 0) & (point_coords[:, 0] < grid_size[0]) &
            (point_coords[:, 1] >= 0) & (point_coords[:, 1] < grid_size[1]) &
            (point_coords[:, 2] >= 0) & (point_coords[:, 2] < grid_size[2])
        )
        points = points[coord_mask]
        point_coords = point_coords[coord_mask]

        voxel_map = {}
        voxel_coords = []
        voxels = []
        num_points = []

        for point, coord_xyz in zip(points, point_coords):
            coord_zyx = (int(coord_xyz[2]), int(coord_xyz[1]), int(coord_xyz[0]))
            voxel_idx = voxel_map.get(coord_zyx)
            if voxel_idx is None:
                if len(voxel_coords) >= max_num_voxels:
                    continue
                voxel_idx = len(voxel_coords)
                voxel_map[coord_zyx] = voxel_idx
                voxel_coords.append(coord_zyx)
                voxels.append(np.zeros((max_num_points_per_voxel, points.shape[1]), dtype=np.float32))
                num_points.append(0)

            cur_num_points = num_points[voxel_idx]
            if cur_num_points < max_num_points_per_voxel:
                voxels[voxel_idx][cur_num_points] = point
                num_points[voxel_idx] = cur_num_points + 1

        if not voxel_coords:
            empty_voxels = np.zeros((0, max_num_points_per_voxel, points.shape[-1]), dtype=np.float32)
            empty_coords = np.zeros((0, 3), dtype=np.int32)
            empty_num_points = np.zeros((0,), dtype=np.int32)
            return empty_voxels, empty_coords, empty_num_points

        return (
            np.stack(voxels, axis=0),
            np.asarray(voxel_coords, dtype=np.int32),
            np.asarray(num_points, dtype=np.int32),
        )

"""
        text = text.replace(marker, helper, 1)

    init_before = """    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
"""
    init_after = """    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        self.vsize_xyz = np.asarray(vsize_xyz, dtype=np.float32)
        self.coors_range_xyz = np.asarray(coors_range_xyz, dtype=np.float32)
        self.max_num_points_per_voxel = int(max_num_points_per_voxel)
        self.max_num_voxels = int(max_num_voxels)
        try:
"""
    if init_before in text:
        text = text.replace(init_before, init_after, 1)

    old = """    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \\
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f\"Unexpected error, library: 'cumm' wasn't imported properly.\"
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points
"""
    new = """    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \\
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            voxels, coordinates, num_points = self._generate_voxels_numpy(
                points=points,
                voxel_size=self.vsize_xyz,
                coors_range=self.coors_range_xyz,
                max_num_points_per_voxel=self.max_num_points_per_voxel,
                max_num_voxels=self.max_num_voxels,
            )
        return voxels, coordinates, num_points
"""
    if old in text:
        text = text.replace(old, new)
    text = text.replace("voxel_size=self._voxel_generator.vsize", "voxel_size=self.vsize_xyz")
    text = text.replace("coors_range=self._voxel_generator.coors_range", "coors_range=self.coors_range_xyz")
    text = text.replace(
        "max_num_points_per_voxel=self._voxel_generator.voxels.shape[1]",
        "max_num_points_per_voxel=self.max_num_points_per_voxel",
    )
    text = text.replace(
        "max_num_voxels=self._voxel_generator.voxels.shape[0]",
        "max_num_voxels=self.max_num_voxels",
    )
    path.write_text(text, encoding="utf-8")


def needs_refresh() -> bool:
    if not (PATCHED_ROOT / "tools").is_dir():
        return True
    if not MARKER_PATH.is_file():
        return True
    kitti_dataset_path = PATCHED_ROOT / "pcdet" / "datasets" / "kitti" / "kitti_dataset.py"
    if not kitti_dataset_path.is_file():
        return True
    kitti_text = kitti_dataset_path.read_text(encoding="utf-8")
    kitti_header = "\n".join(kitti_text.splitlines()[:6])
    if "from pathlib import Path" not in kitti_header:
        return True
    detector_template_path = PATCHED_ROOT / "pcdet" / "models" / "detectors" / "detector3d_template.py"
    if not detector_template_path.is_file():
        return True
    detector_text = detector_template_path.read_text(encoding="utf-8")
    if "weights_only=False" not in detector_text:
        return True
    test_py_path = PATCHED_ROOT / "tools" / "test.py"
    if not test_py_path.is_file():
        return True
    test_py_text = test_py_path.read_text(encoding="utf-8")
    if "pre_trained_path=args.pretrained_model" not in test_py_text:
        return True
    common_utils_path = PATCHED_ROOT / "pcdet" / "utils" / "common_utils.py"
    if not common_utils_path.is_file():
        return True
    common_utils_text = common_utils_path.read_text(encoding="utf-8")
    if "(points[:, 2] >= limit_range[2])" not in common_utils_text:
        return True
    data_processor_path = PATCHED_ROOT / "pcdet" / "datasets" / "processor" / "data_processor.py"
    if not data_processor_path.is_file():
        return True
    data_processor_text = data_processor_path.read_text(encoding="utf-8")
    if "def _generate_voxels_numpy" not in data_processor_text:
        return True
    marker = MARKER_PATH.read_text(encoding="utf-8").strip()
    source_stamp = str(int(SOURCE_ROOT.stat().st_mtime))
    return marker != source_stamp


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"Missing OpenPCDet source tree: {SOURCE_ROOT}")

    if needs_refresh():
        if PATCHED_ROOT.exists():
            shutil.rmtree(PATCHED_ROOT)
        shutil.copytree(SOURCE_ROOT, PATCHED_ROOT, symlinks=True)
    (PATCHED_ROOT / "pcdet" / "datasets" / "__init__.py").write_text(
        PATCHED_DATASETS_INIT,
        encoding="utf-8",
    )
    patch_kitti_dataset(PATCHED_ROOT / "pcdet" / "datasets" / "kitti" / "kitti_dataset.py")
    patch_detector3d_template(PATCHED_ROOT / "pcdet" / "models" / "detectors" / "detector3d_template.py")
    patch_test_py(PATCHED_ROOT / "tools" / "test.py")
    patch_train_py(PATCHED_ROOT / "tools" / "train.py")
    patch_train_utils(PATCHED_ROOT / "tools" / "train_utils" / "train_utils.py")
    patch_common_utils(PATCHED_ROOT / "pcdet" / "utils" / "common_utils.py")
    patch_data_processor(PATCHED_ROOT / "pcdet" / "datasets" / "processor" / "data_processor.py")
    MARKER_PATH.write_text(str(int(SOURCE_ROOT.stat().st_mtime)), encoding="utf-8")

    EXTRA_PY_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"patched_root={PATCHED_ROOT}")
    print(f"extra_pythonpath={EXTRA_PY_ROOT}")


if __name__ == "__main__":
    main()

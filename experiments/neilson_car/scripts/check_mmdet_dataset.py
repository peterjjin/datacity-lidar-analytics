#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope


CONFIG_PATH = Path(
    "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics/"
    "experiments/neilson_car/configs/mmdet3d_pointpillars_car.py"
)


def main() -> None:
    from mmdet3d.registry import DATASETS

    init_default_scope("mmdet3d")
    cfg = Config.fromfile(str(CONFIG_PATH))

    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    val_dataset = DATASETS.build(cfg.val_dataloader.dataset)

    print("NEILSON_CAR_KITTI_ROOT", os.environ.get("NEILSON_CAR_KITTI_ROOT", "<default>"))
    print("train_len", len(train_dataset))
    print("val_len", len(val_dataset))

    sample = train_dataset[0]
    print("train_sample_keys", sorted(sample.keys()))
    if "inputs" in sample and "points" in sample["inputs"]:
        print("train_points_shape", tuple(sample["inputs"]["points"].shape))
    if "data_samples" in sample:
        ds = sample["data_samples"]
        if hasattr(ds, "gt_instances_3d"):
            print("gt_box_count", len(ds.gt_instances_3d))


if __name__ == "__main__":
    main()

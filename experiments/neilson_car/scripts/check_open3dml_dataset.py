#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from open3d._ml3d.datasets.kitti import KITTI


WORKSPACE_ROOT = Path("/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics")
CONFIG_PATH = WORKSPACE_ROOT / "data" / "neilson_car_kitti" / "open3dml_dataset.json"


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = KITTI(cfg["dataset_path"], val_split=int(cfg["val_split"]))

    print(f"dataset_path={cfg['dataset_path']}")
    for split in ("training", "validation", "test"):
        split_ds = dataset.get_split(split)
        print(f"{split}: {len(split_ds)}")

    for split in ("training", "validation"):
        split_ds = dataset.get_split(split)
        sample = split_ds.get_data(0)
        attr = split_ds.get_attr(0)
        print(
            f"{split} sample: name={attr['name']} "
            f"points={tuple(sample['point'].shape)} "
            f"boxes={len(sample['bounding_boxes'])}"
        )


if __name__ == "__main__":
    main()

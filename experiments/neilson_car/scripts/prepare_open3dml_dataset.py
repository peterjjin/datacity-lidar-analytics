#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "neilson_car_kitti"


def read_split_count(data_root: Path, name: str) -> int:
    path = data_root / "ImageSets" / f"{name}.txt"
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    data_root = Path(os.environ.get("NEILSON_CAR_KITTI_ROOT", str(DEFAULT_DATA_ROOT))).resolve()
    manifest_path = data_root / "neilson_car_manifest.json"
    output_path = data_root / "open3dml_dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    counts = manifest.get("counts", {})
    train_count = read_split_count(data_root, "train")
    val_count = read_split_count(data_root, "val")
    test_count = read_split_count(data_root, "test")
    payload = {
        "dataset_path": str(data_root),
        "dataset_class": "open3d._ml3d.datasets.kitti.KITTI",
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        # Open3D-ML's KITTI loader uses a numeric val_split instead of ImageSets/*.txt.
        "val_split": train_count,
        "notes": [
            "Neilson uses the existing KITTI-style export under data/neilson_car_kitti.",
            "Open3D-ML KITTI split boundaries are controlled by val_split.",
            "Use val_split=train_count to reproduce the Neilson train/val split.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

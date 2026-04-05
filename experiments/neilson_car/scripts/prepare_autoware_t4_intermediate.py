#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KITTI_ROOT = WORKSPACE_ROOT / "data" / "neilson_car_kitti"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "data" / "neilson_autoware_t4_intermediate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Neilson KITTI-normalized data into an Autoware/T4-oriented intermediate format."
    )
    parser.add_argument("--kitti-root", default=str(DEFAULT_KITTI_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--scene-name", default="AlbanyNeilson_DATA_20240201_213000")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def yaw_to_quaternion(yaw: float) -> dict[str, float]:
    half = yaw / 2.0
    return {"w": math.cos(half), "x": 0.0, "y": 0.0, "z": math.sin(half)}


def parse_kitti_labels(label_path: Path) -> list[dict]:
    labels: list[dict] = []
    if not label_path.exists():
        return labels
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 15:
            continue
        category = parts[0]
        h = float(parts[8])
        w = float(parts[9])
        l = float(parts[10])
        x = float(parts[11])
        y = float(parts[12])
        z = float(parts[13])
        yaw = float(parts[14])
        labels.append(
            {
                "category_name": category,
                "instance_id": "",
                "attribute_names": [],
                "three_d_bbox": {
                    "translation": {"x": x, "y": y, "z": z},
                    "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "size": {"width": w, "length": l, "height": h},
                    "rotation": yaw_to_quaternion(yaw),
                },
                "num_lidar_pts": 0,
                "num_radar_pts": 0,
            }
        )
    return labels


def export_split(kitti_root: Path, out_root: Path, split_name: str, frame_ids: list[str]) -> dict[str, int]:
    split_root = out_root / split_name
    if split_root.exists():
        shutil.rmtree(split_root)

    lidar_dir = split_root / "data" / "LIDAR_CONCAT"
    label_dir = split_root / "annotation"
    ensure_dir(lidar_dir)
    ensure_dir(label_dir)

    frame_annotations: dict[str, list[dict]] = {}
    nonempty = 0
    total_boxes = 0

    for frame_id in frame_ids:
        src_bin = kitti_root / ("training" if split_name != "test" else "testing") / "velodyne" / f"{frame_id}.bin"
        dst_bin = lidar_dir / f"{frame_id}.pcd.bin"

        points = np.fromfile(src_bin, dtype=np.float32).reshape(-1, 4)
        ring = np.full((points.shape[0], 1), -1.0, dtype=np.float32)
        np.concatenate([points, ring], axis=1).tofile(dst_bin)

        if split_name == "test":
            labels = []
        else:
            labels = parse_kitti_labels(kitti_root / "training" / "label_2" / f"{frame_id}.txt")
        frame_annotations[frame_id] = labels
        if labels:
            nonempty += 1
            total_boxes += len(labels)

    (label_dir / "frame_annotations.json").write_text(json.dumps(frame_annotations, indent=2) + "\n", encoding="utf-8")
    (out_root / split_name / "frames.txt").write_text("\n".join(frame_ids) + "\n", encoding="utf-8")
    return {"frames": len(frame_ids), "nonempty_frames": nonempty, "boxes": total_boxes}


def main() -> None:
    args = parse_args()
    kitti_root = Path(args.kitti_root).resolve()
    out_root = Path(args.output_root).resolve()
    ensure_dir(out_root)

    train_ids = read_split(kitti_root / "ImageSets" / "train.txt")
    val_ids = read_split(kitti_root / "ImageSets" / "val.txt")
    test_ids = read_split(kitti_root / "ImageSets" / "test.txt")

    counts = {
        "train": export_split(kitti_root, out_root, "train", train_ids),
        "val": export_split(kitti_root, out_root, "val", val_ids),
        "test": export_split(kitti_root, out_root, "test", test_ids),
    }

    manifest = {
        "source_kitti_root": str(kitti_root),
        "output_root": str(out_root),
        "scene_name": args.scene_name,
        "notes": [
            "This is an intermediate Autoware/T4-oriented export, not a complete official T4 dataset.",
            "Point clouds are converted from KITTI-style .bin (x,y,z,intensity) to .pcd.bin-style float32 (x,y,z,intensity,ring_idx).",
            "Labels are emitted in the per-frame structure expected by TIER IV annotation generation tools: category_name, instance_id, attribute_names, and three_d_bbox.",
        ],
        "counts": counts,
    }
    (out_root / "autoware_t4_intermediate_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

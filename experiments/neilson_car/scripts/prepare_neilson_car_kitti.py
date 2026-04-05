#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/mnt/d/Dropbox/SmartMobility/2022.11.23 3D-2D Object Integration/Code/"
    "RULabeler/data/AlbanyNeilson/20240201/DATA_20240201_213000/kitti_open3dml"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "data" / "neilson_car_kitti"
MIN_LABEL_JSON_BYTES = 2048

# 1x1 black PNG.
DUMMY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a normalized KITTI root for the Neilson Car experiment."
    )
    p.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument(
        "--label-json-root",
        default="",
        help="Optional raw RULabeler label directory used to skip supplemental frames by JSON size.",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=["Car"],
        help="Classes to keep in training/validation labels.",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def write_padded_calib(src: Path, dst: Path) -> None:
    lines = [line.rstrip("\n") for line in src.read_text(encoding="utf-8").splitlines()]
    if not any(line.startswith("Tr_imu_to_velo:") for line in lines):
        lines.append("Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dummy_png(path: Path) -> None:
    path.write_bytes(DUMMY_PNG_BYTES)


def read_split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_filtered_label_lines(src: Path, keep_classes: set[str]) -> list[str]:
    out_lines: list[str] = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts and parts[0] in keep_classes:
            out_lines.append(line)
    return out_lines


def write_filtered_label_file(dst: Path, filtered_lines: list[str]) -> None:
    dst.write_text("\n".join(filtered_lines) + ("\n" if filtered_lines else ""), encoding="utf-8")


def is_valid_label_json(sample_id: str, label_json_root: Path | None) -> bool:
    if label_json_root is None:
        return True
    json_path = label_json_root / f"{sample_id}.json"
    if not json_path.is_file():
        return True
    return json_path.stat().st_size >= MIN_LABEL_JSON_BYTES


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    keep_classes = set(args.classes)
    if args.label_json_root:
        label_json_root = Path(args.label_json_root).expanduser().resolve()
    else:
        candidate = source_root.parent / "label"
        label_json_root = candidate if candidate.is_dir() else None

    if not source_root.is_dir():
        raise SystemExit(f"Missing source root: {source_root}")

    imagesets_src = source_root / "ImageSets"
    train_ids_src = read_split_ids(imagesets_src / "train.txt")
    val_ids_src = read_split_ids(imagesets_src / "val.txt")
    test_ids = read_split_ids(imagesets_src / "test.txt")

    train_ids: list[str] = []
    val_ids: list[str] = []
    skipped_small_json: dict[str, list[str]] = {"train": [], "val": []}
    skipped_empty_after_filter: dict[str, list[str]] = {"train": [], "val": []}
    suspicious_large_json_empty_after_filter: dict[str, list[str]] = {"train": [], "val": []}

    ensure_dir(output_root / "ImageSets")
    for rel in [
        "training/calib",
        "training/label_2",
        "training/velodyne",
        "training/image_2",
        "testing/calib",
        "testing/velodyne",
        "testing/image_2",
    ]:
        ensure_dir(output_root / rel)

    def build_split(sample_ids: list[str], split_name: str) -> list[str]:
        kept_ids: list[str] = []
        for sample_id in sample_ids:
            valid_json = is_valid_label_json(sample_id, label_json_root)
            if not valid_json:
                skipped_small_json[split_name].append(sample_id)
                continue
            filtered_lines = read_filtered_label_lines(
                source_root / "training" / "label_2" / f"{sample_id}.txt",
                keep_classes,
            )
            if not filtered_lines:
                skipped_empty_after_filter[split_name].append(sample_id)
                if label_json_root is not None:
                    json_path = label_json_root / f"{sample_id}.json"
                    if json_path.is_file() and json_path.stat().st_size >= MIN_LABEL_JSON_BYTES:
                        suspicious_large_json_empty_after_filter[split_name].append(sample_id)
                continue

            write_padded_calib(
                source_root / "training" / "calib" / f"{sample_id}.txt",
                output_root / "training" / "calib" / f"{sample_id}.txt",
            )
            safe_symlink(
                source_root / "training" / "velodyne" / f"{sample_id}.bin",
                output_root / "training" / "velodyne" / f"{sample_id}.bin",
            )
            write_dummy_png(output_root / "training" / "image_2" / f"{sample_id}.png")
            write_filtered_label_file(
                output_root / "training" / "label_2" / f"{sample_id}.txt",
                filtered_lines,
            )
            kept_ids.append(sample_id)
        return kept_ids

    train_ids = build_split(train_ids_src, "train")
    val_ids = build_split(val_ids_src, "val")

    (output_root / "ImageSets" / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (output_root / "ImageSets" / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    (output_root / "ImageSets" / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")

    for sample_id in test_ids:
        write_padded_calib(
            source_root / "training" / "calib" / f"{sample_id}.txt",
            output_root / "testing" / "calib" / f"{sample_id}.txt",
        )
        safe_symlink(
            source_root / "training" / "velodyne" / f"{sample_id}.bin",
            output_root / "testing" / "velodyne" / f"{sample_id}.bin",
        )
        write_dummy_png(output_root / "testing" / "image_2" / f"{sample_id}.png")

    reduced = output_root / "training" / "velodyne_reduced"
    if reduced.exists() or reduced.is_symlink():
        reduced.unlink()
    reduced.symlink_to(output_root / "training" / "velodyne")

    train_label_count = sum(
        1
        for sample_id in train_ids
        if (output_root / "training" / "label_2" / f"{sample_id}.txt").read_text(encoding="utf-8").strip()
    )
    val_label_count = sum(
        1
        for sample_id in val_ids
        if (output_root / "training" / "label_2" / f"{sample_id}.txt").read_text(encoding="utf-8").strip()
    )

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "classes": sorted(keep_classes),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "train_nonempty_filtered_labels": train_label_count,
            "val_nonempty_filtered_labels": val_label_count,
            "train_skipped_small_json": len(skipped_small_json["train"]),
            "val_skipped_small_json": len(skipped_small_json["val"]),
            "train_skipped_empty_after_filter": len(skipped_empty_after_filter["train"]),
            "val_skipped_empty_after_filter": len(skipped_empty_after_filter["val"]),
            "train_suspicious_large_json_empty_after_filter": len(
                suspicious_large_json_empty_after_filter["train"]
            ),
            "val_suspicious_large_json_empty_after_filter": len(
                suspicious_large_json_empty_after_filter["val"]
            ),
        },
        "label_json_root": str(label_json_root) if label_json_root else "",
        "min_label_json_bytes": MIN_LABEL_JSON_BYTES,
        "suspicious_large_json_empty_after_filter": suspicious_large_json_empty_after_filter,
    }
    (output_root / "neilson_car_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

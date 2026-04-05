#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "data" / "neilson_car_kitti"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "data" / "neilson_car_kitti_subset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny Neilson KITTI subset using symlinked files.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose_with_labels(source_root: Path, split_name: str, count: int) -> list[str]:
    ids = read_ids(source_root / "ImageSets" / f"{split_name}.txt")
    selected: list[str] = []
    label_root = source_root / "training" / "label_2"
    for sample_id in ids:
        label_path = label_root / f"{sample_id}.txt"
        if split_name == "test" or label_path.read_text(encoding="utf-8").strip():
            selected.append(sample_id)
        if len(selected) >= count:
            break
    return selected


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def write_split(path: Path, sample_ids: list[str]) -> None:
    text = "\n".join(sample_ids)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()

    if output_root.exists():
        if not args.force:
            raise SystemExit(f"{output_root} already exists; pass --force to replace it")
        shutil.rmtree(output_root)

    train_ids = choose_with_labels(source_root, "train", args.train_count)
    val_ids = choose_with_labels(source_root, "val", args.val_count)
    test_ids = choose_with_labels(source_root, "test", args.test_count)

    for relative_dir in [
        "training/velodyne",
        "training/calib",
        "training/image_2",
        "training/label_2",
        "testing/velodyne",
        "testing/calib",
        "testing/image_2",
        "ImageSets",
    ]:
        ensure_dir(output_root / relative_dir)

    for sample_id in train_ids + val_ids:
        for relative in [
            f"training/velodyne/{sample_id}.bin",
            f"training/calib/{sample_id}.txt",
            f"training/image_2/{sample_id}.png",
            f"training/label_2/{sample_id}.txt",
        ]:
            link_file(source_root / relative, output_root / relative)

    for sample_id in test_ids:
        for relative in [
            f"testing/velodyne/{sample_id}.bin",
            f"testing/calib/{sample_id}.txt",
            f"testing/image_2/{sample_id}.png",
        ]:
            link_file(source_root / relative, output_root / relative)

    for name, sample_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        write_split(output_root / "ImageSets" / f"{name}.txt", sample_ids)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "sample_ids": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }
    (output_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

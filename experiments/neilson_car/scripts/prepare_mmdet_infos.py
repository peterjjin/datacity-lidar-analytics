#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import mmengine

MMDET3D_ROOT = Path("/home/datacity/src/mmdetection3d")
DEFAULT_DATA_ROOT = Path(
    "/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics/data/neilson_car_kitti"
)
INFO_PREFIX = "neilson_car"
MIN_LABEL_JSON_BYTES = 2048


def read_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_label_json_root(data_root: Path) -> Path | None:
    manifest_path = data_root / "neilson_car_manifest.json"
    if manifest_path.is_file():
        manifest = mmengine.load(manifest_path)
        label_json_root = manifest.get("label_json_root")
        if label_json_root:
            path = Path(label_json_root)
            if path.is_dir():
                return path
    return None


def filter_v2_infos(
    pkl_path: Path,
    split_name: str,
    label_json_root: Path | None,
) -> list[tuple[str, int]]:
    payload = mmengine.load(pkl_path)
    if not isinstance(payload, dict) or "data_list" not in payload:
        raise RuntimeError(f"Unexpected info payload format in {pkl_path}")

    filtered_data_list = []
    suspicious: list[tuple[str, int]] = []
    skipped_small = 0
    skipped_empty = 0

    for info in payload["data_list"]:
        sample_idx = str(info.get("sample_idx"))
        instances = info.get("instances", [])
        if instances:
            filtered_data_list.append(info)
            continue

        skipped_empty += 1
        if label_json_root is None:
            continue

        json_path = label_json_root / f"{sample_idx}.json"
        if not json_path.is_file():
            suspicious.append((sample_idx, -1))
            continue

        size = json_path.stat().st_size
        if size < MIN_LABEL_JSON_BYTES:
            skipped_small += 1
            continue

        suspicious.append((sample_idx, size))

    payload["data_list"] = filtered_data_list
    mmengine.dump(payload, pkl_path)

    print(
        f"[{split_name}] kept {len(filtered_data_list)} infos, "
        f"skipped_empty={skipped_empty}, skipped_small_json={skipped_small}"
    )
    if suspicious:
        print(f"[{split_name}] suspicious empty-instance samples with raw json >= {MIN_LABEL_JSON_BYTES} bytes:")
        for sample_idx, size in suspicious:
            if size >= 0:
                print(f"  sample_idx={sample_idx} json_size={size}")
            else:
                print(f"  sample_idx={sample_idx} json_missing=1")

    return suspicious


def normalize_to_legacy_list(pkl_path: Path) -> None:
    payload = mmengine.load(pkl_path)
    if isinstance(payload, dict) and "data_list" in payload:
        mmengine.dump(payload["data_list"], pkl_path)


def main() -> None:
    data_root = Path(os.environ.get("NEILSON_CAR_KITTI_ROOT", str(DEFAULT_DATA_ROOT))).resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Missing data root: {data_root}")
    label_json_root = infer_label_json_root(data_root)

    sys.path.insert(0, str(MMDET3D_ROOT))

    from tools.dataset_converters.kitti_converter import _calculate_num_points_in_gt
    from tools.dataset_converters.kitti_data_utils import get_kitti_image_info
    from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

    image_sets = data_root / "ImageSets"
    train_ids = read_ids(image_sets / "train.txt")
    val_ids = read_ids(image_sets / "val.txt")
    test_ids = read_ids(image_sets / "test.txt")

    print("Generating Neilson KITTI-style info files without camera-FOV cropping.")

    train_infos = get_kitti_image_info(
        str(data_root),
        training=True,
        label_info=True,
        velodyne=True,
        calib=True,
        with_plane=False,
        image_ids=train_ids,
        relative_path=True,
        with_imageshape=True,
    )
    _calculate_num_points_in_gt(str(data_root), train_infos, relative_path=True, remove_outside=False)
    mmengine.dump(train_infos, data_root / f"{INFO_PREFIX}_infos_train.pkl")

    val_infos = get_kitti_image_info(
        str(data_root),
        training=True,
        label_info=True,
        velodyne=True,
        calib=True,
        with_plane=False,
        image_ids=val_ids,
        relative_path=True,
        with_imageshape=True,
    )
    _calculate_num_points_in_gt(str(data_root), val_infos, relative_path=True, remove_outside=False)
    mmengine.dump(val_infos, data_root / f"{INFO_PREFIX}_infos_val.pkl")

    mmengine.dump(train_infos + val_infos, data_root / f"{INFO_PREFIX}_infos_trainval.pkl")

    test_infos = get_kitti_image_info(
        str(data_root),
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        with_plane=False,
        image_ids=test_ids,
        relative_path=True,
        with_imageshape=True,
    )
    mmengine.dump(test_infos, data_root / f"{INFO_PREFIX}_infos_test.pkl")

    for name in ("train", "val", "trainval", "test"):
        normalize_to_legacy_list(data_root / f"{INFO_PREFIX}_infos_{name}.pkl")
        update_pkl_infos(
            "kitti",
            out_dir=str(data_root),
            pkl_path=str(data_root / f"{INFO_PREFIX}_infos_{name}.pkl"),
        )

    train_pkl = data_root / f"{INFO_PREFIX}_infos_train.pkl"
    val_pkl = data_root / f"{INFO_PREFIX}_infos_val.pkl"
    train_payload = mmengine.load(train_pkl)
    val_payload = mmengine.load(val_pkl)
    suspicious_train = filter_v2_infos(train_pkl, "train", label_json_root)
    suspicious_val = filter_v2_infos(val_pkl, "val", label_json_root)

    filtered_train = mmengine.load(train_pkl)
    filtered_val = mmengine.load(val_pkl)
    trainval_payload = {
        "metainfo": filtered_train.get("metainfo", train_payload.get("metainfo", {})),
        "data_list": filtered_train.get("data_list", []) + filtered_val.get("data_list", []),
    }
    mmengine.dump(trainval_payload, data_root / f"{INFO_PREFIX}_infos_trainval.pkl")

    if suspicious_train or suspicious_val:
        print(
            "WARNING: found empty-instance samples whose raw json files were not supplemental "
            f"(<{MIN_LABEL_JSON_BYTES} bytes). Review the sample_idx values printed above."
        )

    print("Wrote:")
    for name in ("train", "val", "trainval", "test"):
        print(data_root / f"{INFO_PREFIX}_infos_{name}.pkl")


if __name__ == "__main__":
    main()

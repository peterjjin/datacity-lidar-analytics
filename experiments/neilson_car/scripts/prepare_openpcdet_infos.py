#!/usr/bin/env python3
from __future__ import annotations

import pickle
import sys
from pathlib import Path
import json
import yaml
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_openpcdet_runtime import main as bootstrap_openpcdet_runtime


import os


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
OPENPCDET_PATCHED_ROOT = Path("/tmp/openpcdet_patched")
OPENPCDET_SOURCE_ROOT = Path("/home/datacity/src/OpenPCDet")
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "neilson_car_kitti"
DATASET_CFG_PATH = OPENPCDET_SOURCE_ROOT / "tools" / "cfgs" / "dataset_configs" / "kitti_dataset.yaml"
MIN_LABEL_JSON_BYTES = 2048


def dump_pickle(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def infer_label_json_root() -> Path | None:
    data_root = Path(os.environ.get("NEILSON_CAR_KITTI_ROOT", str(DEFAULT_DATA_ROOT))).resolve()
    manifest_path = data_root / "neilson_car_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_json_root = manifest.get("label_json_root")
    if not label_json_root:
        return None
    path = Path(label_json_root)
    return path if path.is_dir() else None


def filter_infos(infos: list[dict], split_name: str, label_json_root: Path | None) -> list[dict]:
    filtered: list[dict] = []
    suspicious: list[tuple[str, int]] = []
    skipped_empty = 0
    skipped_small = 0

    for info in infos:
        annos = info.get("annos")
        names = annos.get("name", []) if isinstance(annos, dict) else []
        if len(names) > 0:
            filtered.append(info)
            continue

        skipped_empty += 1
        sample_idx = str(info.get("point_cloud", {}).get("lidar_idx", ""))
        if label_json_root is None or not sample_idx:
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

    print(
        f"[{split_name}] kept {len(filtered)} infos, "
        f"skipped_empty={skipped_empty}, skipped_small_json={skipped_small}"
    )
    if suspicious:
        print(f"[{split_name}] suspicious empty annos with raw json >= {MIN_LABEL_JSON_BYTES} bytes:")
        for sample_idx, size in suspicious:
            if size >= 0:
                print(f"  sample_idx={sample_idx} json_size={size}")
            else:
                print(f"  sample_idx={sample_idx} json_missing=1")

    return filtered


def main() -> None:
    bootstrap_openpcdet_runtime()
    sys.path.insert(0, str(OPENPCDET_PATCHED_ROOT))
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    data_root = Path(os.environ.get("NEILSON_CAR_KITTI_ROOT", str(DEFAULT_DATA_ROOT))).resolve()
    label_json_root = infer_label_json_root()

    dataset_cfg = EasyDict(yaml.safe_load(DATASET_CFG_PATH.read_text(encoding="utf-8")))
    dataset_cfg.DATA_PATH = str(data_root)
    dataset_cfg.FOV_POINTS_ONLY = False

    dataset = KittiDataset(
        dataset_cfg=dataset_cfg,
        class_names=["Car", "Pedestrian", "Cyclist"],
        root_path=data_root,
        training=False,
    )

    print("Generating native OpenPCDet KITTI info files from Neilson export")
    for split, has_label in (("train", True), ("val", True), ("test", False)):
        dataset.set_split(split)
        infos = dataset.get_infos(
            num_workers=4,
            has_label=has_label,
            count_inside_pts=False,
        )
        if has_label:
            infos = filter_infos(infos, split, label_json_root)
        out_path = data_root / f"kitti_infos_{split}.pkl"
        dump_pickle(out_path, infos)
        print(f"{split}: wrote {out_path} ({len(infos)} samples)")

    train_infos = pickle.loads((data_root / "kitti_infos_train.pkl").read_bytes())
    val_infos = pickle.loads((data_root / "kitti_infos_val.pkl").read_bytes())
    trainval_path = data_root / "kitti_infos_trainval.pkl"
    dump_pickle(trainval_path, train_infos + val_infos)
    print(f"trainval: wrote {trainval_path} ({len(train_infos) + len(val_infos)} samples)")

    dataset.set_split("train")
    dataset.create_groundtruth_database(data_root / "kitti_infos_train.pkl", split="train")
    print(f"db: wrote {(data_root / 'kitti_dbinfos_train.pkl')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from open3dml_gpu_compat import apply_open3dml_gpu_compat_patches
from open3d._ml3d.utils import Config
from open3d.ml.torch import datasets, models, pipelines


OPEN3D_CFG = Path(
    "/home/datacity/miniforge3/envs/lidar-open3d-ml/lib/python3.10/site-packages/open3d/_ml3d/configs/pointpillars_kitti.yml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--config", default=str(OPEN3D_CFG))
    parser.add_argument("--val-split", type=int, default=4164)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    return parser.parse_args()


def label_name(raw_label, classes) -> str:
    if isinstance(raw_label, str):
        return raw_label
    if isinstance(raw_label, int):
        return classes[raw_label]
    return str(raw_label)


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError(f"Open3D-ML inference is GPU-only in this workspace; received device={args.device!r}.")
    data_root = Path(args.data_root).resolve()
    pred_dir = Path(args.output_root).resolve() / args.scene_name / "label"
    pred_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.load_from_file(args.config)
    cfg.model.ckpt_path = str(Path(args.checkpoint).resolve())
    cfg.dataset.dataset_path = str(data_root)
    cfg.pipeline.test_compute_metric = False
    cfg.pipeline.summary.record_for = []

    apply_open3dml_gpu_compat_patches()
    model = models.PointPillars(device=args.device, **cfg.model)
    dataset = datasets.KITTI(cfg.dataset.dataset_path, val_split=args.val_split)
    pipeline = pipelines.ObjectDetection(
        model,
        dataset=dataset,
        device=args.device,
        split="validation",
        **cfg.pipeline,
    )
    pipeline.load_ckpt(ckpt_path=cfg.model.ckpt_path)

    split = dataset.get_split("validation")
    classes = list(cfg.model.classes)

    for idx in range(len(split)):
        sample = split.get_data(idx)
        attr = split.get_attr(idx)
        frame_id = str(int(attr["name"]))
        raw_boxes = pipeline.run_inference(sample)
        boxes = raw_boxes[0] if raw_boxes and isinstance(raw_boxes[0], list) else raw_boxes

        payload = []
        for box_idx, box in enumerate(boxes):
            label = label_name(box.label_class, classes)
            payload.append(
                {
                    "obj_type": label,
                    "obj_id": box_idx,
                    "score": float(box.confidence),
                    "psr": {
                        "position": {
                            "x": float(box.center[0]),
                            "y": float(box.center[1]),
                            "z": float(box.center[2]),
                        },
                        "rotation": {"x": 0.0, "y": 0.0, "z": float(box.yaw)},
                        "scale": {
                            "x": float(box.size[2]),
                            "y": float(box.size[0]),
                            "z": float(box.size[1]),
                        },
                    },
                }
            )

        out_path = pred_dir / f"{frame_id}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        if (idx + 1) % 50 == 0:
            print(f"processed {idx + 1}/{len(split)}")

    print(f"wrote predictions to {pred_dir}")


if __name__ == "__main__":
    main()

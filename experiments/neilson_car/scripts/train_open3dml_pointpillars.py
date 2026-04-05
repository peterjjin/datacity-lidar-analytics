#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from open3dml_gpu_compat import apply_open3dml_gpu_compat_patches
from open3d._ml3d.utils import Config
from open3d.ml.torch import datasets, models, pipelines


OPEN3D_CFG = Path(__file__).resolve().parents[1] / "configs" / "open3dml_pointpillars_car.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(OPEN3D_CFG))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--val-split", type=int, default=4164)
    parser.add_argument("--max-epoch", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=12)
    return parser.parse_args()


def require_supported_cuda(device: str) -> None:
    if device != "cuda":
        raise RuntimeError(f"Open3D-ML training is GPU-only in this workspace; received device={device!r}.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Open3D-ML training, but torch.cuda.is_available() is False.")

    major, minor = torch.cuda.get_device_capability(0)
    active_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())

    if active_arch not in supported_arches:
        raise RuntimeError(
            "Open3D-ML training is running on an unsupported GPU architecture for this Torch build: "
            f"{active_arch} not in {sorted(supported_arches)}. "
            "This environment currently uses the Open3D 0.19 wheel, which requires Torch 2.2.2+cu121. "
            "That Torch build does not support RTX 5090 / sm_120. Rebuild Open3D against a newer Torch "
            "toolchain or run this training on CPU or on an older supported GPU."
        )


def load_partial_checkpoint(model, checkpoint_path: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()

    matched = {}
    skipped = {}
    for key, value in state_dict.items():
        current = model_state.get(key)
        if current is None or tuple(current.shape) != tuple(value.shape):
            skipped[key] = {
                "checkpoint_shape": tuple(value.shape),
                "model_shape": tuple(current.shape) if current is not None else None,
            }
            continue
        matched[key] = value

    missing, unexpected = model.load_state_dict(matched, strict=False)
    return {
        "matched_keys": len(matched),
        "skipped_keys": skipped,
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()

    cfg = Config.load_from_file(args.config)
    cfg.dataset.dataset_path = str(Path(args.data_root).resolve())
    cfg.dataset.val_split = int(args.val_split)
    cfg.model.ckpt_path = None
    cfg.model.is_resume = False
    cfg.pipeline.main_log_dir = str(run_root)
    cfg.pipeline.train_sum_dir = str(run_root / "train_log")
    cfg.pipeline.max_epoch = int(args.max_epoch)
    cfg.pipeline.num_workers = int(args.num_workers)

    require_supported_cuda(args.device)
    if args.device == "cuda":
        apply_open3dml_gpu_compat_patches()
    model = models.PointPillars(device=args.device, **cfg.model)
    load_report = load_partial_checkpoint(model, checkpoint_path)
    dataset = datasets.KITTI(
        cfg.dataset.dataset_path,
        val_split=cfg.dataset.val_split,
    )
    pipeline = pipelines.ObjectDetection(
        model,
        dataset=dataset,
        device=args.device,
        split="train",
        **cfg.pipeline,
    )
    pipeline.run_train()

    checkpoint_dir = run_root / "PointPillars_KITTI_torch" / "checkpoint"
    checkpoints = sorted(checkpoint_dir.glob("ckpt_*.pth"))
    summary = {
        "config": str(Path(args.config).resolve()),
        "data_root": cfg.dataset.dataset_path,
        "run_root": str(run_root),
        "checkpoint_dir": str(checkpoint_dir),
        "latest_checkpoint": str(checkpoints[-1]) if checkpoints else "",
        "count_checkpoints": len(checkpoints),
        "checkpoint": str(checkpoint_path),
        "checkpoint_load": load_report,
        "device": args.device,
        "max_epoch": cfg.pipeline.max_epoch,
        "num_workers": cfg.pipeline.num_workers,
    }
    (run_root / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

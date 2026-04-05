#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


WORKSPACE_ROOT = Path("/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics")
DEFAULT_ALGO_CONFIG = WORKSPACE_ROOT / "experiments" / "neilson_car" / "algorithms.json"
DEFAULT_NEILSON_SCENE = Path(
    "/mnt/d/Dropbox/SmartMobility/2022.11.23 3D-2D Object Integration/Code/"
    "RULabeler/data/AlbanyNeilson/20240201/DATA_20240201_213000"
)
RULABELER_EVAL = Path(
    "/mnt/d/Dropbox/SmartMobility/2022.11.23 3D-2D Object Integration/Code/"
    "RULabeler/tools/eval_bev_benchmark.py"
)
MIN_LABEL_JSON_BYTES = 2048


@dataclass
class AlgoResult:
    name: str
    status: str
    message: str
    pred_root: str = ""
    eval_root: str = ""
    fp: int = 0
    fn: int = 0
    tp: int = 0
    gt_boxes: int = 0
    pred_boxes: int = 0
    type_i_rate: float = 0.0
    type_ii_rate: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Neilson benchmark across multiple algorithms.")
    p.add_argument("--algo-config", default=str(DEFAULT_ALGO_CONFIG))
    p.add_argument("--scene-dir", default=str(DEFAULT_NEILSON_SCENE))
    p.add_argument("--output-root", default=str(WORKSPACE_ROOT / "experiments" / "neilson_car" / "benchmark_runs"))
    p.add_argument(
        "--frame-list",
        default=str(WORKSPACE_ROOT / "data" / "neilson_car_kitti" / "ImageSets" / "val.txt"),
        help="Optional txt file with one frame id per line. Defaults to the prepared validation split.",
    )
    p.add_argument("--class-aware", action="store_true", default=True)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--score-threshold", type=float, default=0.0)
    p.add_argument("--max-algorithms", type=int, default=0)
    p.add_argument("--save-bev", action="store_true")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(cmd: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_json_files(path: Path) -> None:
    if not path.exists():
        return
    for json_file in path.rglob("*.json"):
        try:
            json_file.unlink()
        except FileNotFoundError:
            pass


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_valid_label_json(path: Path) -> bool:
    try:
        return path.stat().st_size >= MIN_LABEL_JSON_BYTES
    except FileNotFoundError:
        return False


def load_frame_filter(frame_list_path: Optional[Path]) -> Optional[List[str]]:
    if not frame_list_path:
        return None
    if not frame_list_path.is_file():
        return None
    frames = []
    for line in frame_list_path.read_text(encoding="utf-8").splitlines():
        frame = line.strip()
        if frame:
            frames.append(frame.lstrip("0") or "0")
    return frames or None


def prepare_subset_csv(output_root: Path, flat_scene_name: str, frames: Optional[List[str]]) -> Optional[Path]:
    if not frames:
        return None
    subset_path = output_root / "subset_frames.csv"
    with subset_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "frame"])
        writer.writeheader()
        for frame in frames:
            writer.writerow({"scene": flat_scene_name, "frame": frame})
    return subset_path


def prepare_flat_gt(
    scene_dir: Path, output_root: Path, flat_scene_name: str, frames: Optional[List[str]] = None
) -> Path:
    gt_root = output_root / "gt_flat"
    scene_out = gt_root / flat_scene_name
    label_out = scene_out / "label"
    source_label_dir = scene_dir / "label"
    ensure_dir(label_out)

    selected_frames = set(frames or [])
    marker_path = scene_out / ".source_scene"
    marker_payload = {
        "scene_dir": str(scene_dir),
        "frame_count": len(selected_frames),
        "frame_sample": sorted(selected_frames)[:5],
        "min_label_json_bytes": MIN_LABEL_JSON_BYTES,
    }
    marker_value = json.dumps(marker_payload, sort_keys=True)
    source_files = sorted(source_label_dir.glob("*.json"))
    expected_count = sum(
        1
        for src_file in source_files
        if (not selected_frames or src_file.stem in selected_frames) and is_valid_label_json(src_file)
    )
    if marker_path.is_file() and marker_path.read_text(encoding="utf-8") == marker_value:
        existing_count = sum(1 for _ in label_out.glob("*.json"))
        if existing_count >= expected_count if expected_count else existing_count == 0:
            return gt_root

    clear_json_files(label_out)
    for src_file in source_files:
        if selected_frames and src_file.stem not in selected_frames:
            continue
        if not is_valid_label_json(src_file):
            continue
        dst_file = label_out / src_file.name
        if dst_file.is_file():
            continue
        dst_file.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")

    marker_path.write_text(marker_value, encoding="utf-8")
    return gt_root


def iterate_gt_frames(label_dir: Path) -> Iterable[Tuple[str, list]]:
    for label_file in sorted(label_dir.glob("*.json")):
        if not is_valid_label_json(label_file):
            continue
        try:
            data = json.loads(label_file.read_text(encoding="utf-8"))
        except Exception:
            data = []
        yield label_file.stem, data if isinstance(data, list) else []


def kitti_label_line_to_rulabeler(line: str) -> Optional[dict]:
    parts = line.strip().split()
    if len(parts) < 15:
        return None
    obj_type = parts[0]
    h = float(parts[8])
    w = float(parts[9])
    l = float(parts[10])
    x = float(parts[11])
    y = float(parts[12])
    z = float(parts[13])
    yaw = float(parts[14])
    score = float(parts[15]) if len(parts) > 15 else 1.0
    return {
        "obj_type": obj_type,
        "obj_id": "",
        "score": score,
        "psr": {
            "position": {"x": x, "y": y, "z": z},
            "rotation": {"x": 0.0, "y": 0.0, "z": yaw},
            "scale": {"x": l, "y": w, "z": h},
        },
    }


def create_synthetic_predictions(mode: str, gt_root: Path, pred_root: Path, flat_scene_name: str) -> None:
    label_in = gt_root / flat_scene_name / "label"
    label_out = pred_root / flat_scene_name / "label"
    ensure_dir(label_out)
    for frame, data in iterate_gt_frames(label_in):
        if mode == "oracle":
            out = data
        elif mode == "empty":
            out = []
        else:
            raise ValueError(mode)
        (label_out / f"{frame}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def convert_kitti_txt_dir_to_rulabeler(
    src_dir: Path, pred_root: Path, flat_scene_name: str, frame_ids: Optional[List[str]] = None
) -> int:
    count = 0
    label_out = pred_root / flat_scene_name / "label"
    ensure_dir(label_out)
    txt_files = sorted(src_dir.glob("*.txt"))
    for idx, txt_file in enumerate(txt_files):
        objs = []
        for line in txt_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = kitti_label_line_to_rulabeler(line)
            if obj:
                objs.append(obj)
        target_stem = frame_ids[idx] if frame_ids and idx < len(frame_ids) else txt_file.stem
        (label_out / f"{target_stem}.json").write_text(json.dumps(objs, indent=2), encoding="utf-8")
        count += 1
    return count


def run_mmdet3d(
    spec: dict, pred_root: Path, args: argparse.Namespace, flat_scene_name: str, frame_ids: Optional[List[str]]
) -> Tuple[str, str]:
    checkpoint = str(spec.get("checkpoint", "")).strip()
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("missing checkpoint")
    submission_prefix = Path("/tmp") / f"{spec['name']}_submission"
    if submission_prefix.exists():
        shutil.rmtree(submission_prefix)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    env["NEILSON_CAR_KITTI_ROOT"] = str(WORKSPACE_ROOT / "data" / "neilson_car_kitti")
    conda_sh = str(Path.home() / "miniforge3" / "etc" / "profile.d" / "conda.sh")
    command = (
        f"source '{conda_sh}' && conda activate {spec['env']} && "
        f"cd /home/datacity/src/mmdetection3d && "
        f"python tools/test.py '{spec['config']}' '{checkpoint}' "
        f"--work-dir '{spec['work_dir']}' "
        f"--cfg-options test_evaluator.format_only=True "
        f"test_evaluator.submission_prefix='{submission_prefix}' "
        f"work_dir='{spec['work_dir']}' "
        f"test_dataloader.num_workers=0 "
        f"test_dataloader.persistent_workers=False "
        f"val_dataloader.num_workers=0 "
        f"val_dataloader.persistent_workers=False"
    )
    run(["bash", "-lc", command], env=env)
    convert_kitti_txt_dir_to_rulabeler(
        submission_prefix / "pred_instances_3d", pred_root, flat_scene_name, frame_ids=frame_ids
    )
    return str(submission_prefix), "pred_instances_3d"


def run_openpcdet(spec: dict, pred_root: Path, flat_scene_name: str, frame_ids: Optional[List[str]]) -> Tuple[str, str]:
    checkpoint = str(spec.get("checkpoint", "")).strip()
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("missing checkpoint")
    env = os.environ.copy()
    extra_pythonpath = str(spec.get("extra_pythonpath", "")).strip()
    if extra_pythonpath:
        existing = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = f"{existing}:{extra_pythonpath}" if existing else extra_pythonpath
    conda_sh = str(Path.home() / "miniforge3" / "etc" / "profile.d" / "conda.sh")
    set_cfgs = " ".join(shlex.quote(str(x)) for x in spec.get("set_cfgs", []))
    command = (
        f"source '{conda_sh}' && conda activate {spec['env']} && "
        f"cd /tmp/openpcdet_patched/tools && "
        f"python test.py --cfg_file '{spec['config']}' --ckpt '{checkpoint}' "
        f"--batch_size {int(spec.get('batch_size', 1))} "
        f"--workers {int(spec.get('workers', 0))} "
        f"--extra_tag '{spec['extra_tag']}' --eval_tag '{spec['eval_tag']}' --save_to_file "
        f"--set {set_cfgs}"
    )
    run(["bash", "-lc", command], env=env)
    final_data = None
    search_roots = [Path(spec.get("work_dir", "")).resolve(), Path("/tmp/openpcdet_patched/output")]
    for base in search_roots:
        if not str(base) or not base.exists():
            continue
        patterns = [
            f"**/{spec['extra_tag']}/eval/epoch_*/val/{spec['eval_tag']}/final_result/data",
            f"**/{spec['extra_tag']}/eval/epoch_*/test/{spec['eval_tag']}/final_result/data",
            "epoch_*/val/*/final_result/data",
            "epoch_*/test/*/final_result/data",
        ]
        for pattern in patterns:
            final_data = next(base.glob(pattern), None)
            if final_data is not None:
                break
        if final_data is not None:
            break
    if final_data is None:
        raise FileNotFoundError("missing OpenPCDet final_result/data")
    convert_kitti_txt_dir_to_rulabeler(final_data, pred_root, flat_scene_name, frame_ids=frame_ids)
    return str(final_data), "final_result/data"


def run_open3dml(spec: dict, pred_root: Path, flat_scene_name: str) -> Tuple[str, str]:
    checkpoint = str(spec.get("checkpoint", "")).strip()
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError("missing checkpoint")
    device = str(spec.get("device", "cuda")).strip()
    if device != "cuda":
        raise RuntimeError(f"Open3D-ML inference is GPU-only in this workspace; received device={device!r}.")
    env = os.environ.copy()
    extra_pythonpath = str(spec.get("extra_pythonpath", "")).strip()
    if extra_pythonpath:
        existing = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = f"{extra_pythonpath}:{existing}" if existing else extra_pythonpath
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/mplcache")
    env["XDG_CACHE_HOME"] = env.get("XDG_CACHE_HOME", "/tmp")
    conda_sh = str(Path.home() / "miniforge3" / "etc" / "profile.d" / "conda.sh")
    command = (
        f"source '{conda_sh}' && conda activate {spec['env']} && "
        f"python '{WORKSPACE_ROOT / 'experiments' / 'neilson_car' / 'scripts' / 'run_open3dml_pointpillars.py'}' "
        f"--data-root '{spec['data_root']}' "
        f"--checkpoint '{checkpoint}' "
        f"--output-root '{pred_root}' "
        f"--scene-name '{flat_scene_name}' "
        f"--device '{device}' "
        f"--val-split {int(spec.get('val_split', 4164))}"
    )
    run(["bash", "-lc", command], env=env)
    return str(pred_root), "open3dml_rulabeler_json"


def evaluate_predictions(
    gt_root: Path,
    pred_root: Path,
    output_root: Path,
    method_name: str,
    args: argparse.Namespace,
    subset_csv: Optional[Path],
) -> None:
    cmd = [
        sys.executable,
        str(RULABELER_EVAL),
        "--gt-root",
        str(gt_root),
        "--pred-root",
        str(pred_root),
        "--method-name",
        method_name,
        "--output-dir",
        str(output_root),
        "--iou-threshold",
        str(args.iou_threshold),
        "--score-threshold",
        str(args.score_threshold),
    ]
    if subset_csv and subset_csv.is_file():
        cmd.extend(["--subset-csv", str(subset_csv)])
    if args.class_aware:
        cmd.append("--class-aware")
    if args.save_bev:
        cmd.append("--save-bev")
    run(cmd)


def summarize_eval(output_root: Path, algo_name: str) -> AlgoResult:
    scene_csv = output_root / algo_name / "scene_metrics.csv"
    rows = list(csv.DictReader(scene_csv.open("r", encoding="utf-8")))
    if not rows:
        return AlgoResult(name=algo_name, status="failed", message="empty scene metrics")
    row = rows[0]
    tp = int(row["tp"])
    fp = int(row["fp"])
    fn = int(row["fn"])
    gt_boxes = int(row["gt_boxes"])
    pred_boxes = int(row["pred_boxes"])
    type_i = fp / max(tp + fp, 1)
    type_ii = fn / max(tp + fn, 1)
    return AlgoResult(
        name=algo_name,
        status="ok",
        message="evaluated",
        eval_root=str(output_root / algo_name),
        tp=tp,
        fp=fp,
        fn=fn,
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        type_i_rate=type_i,
        type_ii_rate=type_ii,
    )


def write_reports(results: List[AlgoResult], output_root: Path) -> None:
    rows = []
    for r in results:
        rows.append(
            {
                "algorithm": r.name,
                "status": r.status,
                "message": r.message,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "gt_boxes": r.gt_boxes,
                "pred_boxes": r.pred_boxes,
                "type_i_rate": f"{r.type_i_rate:.6f}",
                "type_ii_rate": f"{r.type_ii_rate:.6f}",
                "pred_root": r.pred_root,
                "eval_root": r.eval_root,
            }
        )
    csv_path = output_root / "algorithm_error_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "status",
                "message",
                "tp",
                "fp",
                "fn",
                "gt_boxes",
                "pred_boxes",
                "type_i_rate",
                "type_ii_rate",
                "pred_root",
                "eval_root",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    md_lines = [
        "# Neilson Algorithm Error Report",
        "",
        "| Algorithm | Status | TP | FP | FN | Type I | Type II | Message |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        md_lines.append(
            f"| {r.name} | {r.status} | {r.tp} | {r.fp} | {r.fn} | {r.type_i_rate:.4f} | {r.type_ii_rate:.4f} | {r.message} |"
        )
    md_lines.extend(
        [
            "",
            "Type I error is reported here as `FP / (TP + FP)`.",
            "Type II error is reported here as `FN / (TP + FN)`.",
        ]
    )
    (output_root / "algorithm_error_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    algo_config = read_json(Path(args.algo_config))
    scene_dir = Path(args.scene_dir).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    flat_scene_name = algo_config.get("scene_name", "AlbanyNeilson_DATA_20240201_213000")
    frame_filter = load_frame_filter(Path(args.frame_list) if args.frame_list else None)
    gt_root = prepare_flat_gt(scene_dir, output_root, flat_scene_name, frames=frame_filter)
    subset_csv = prepare_subset_csv(output_root, flat_scene_name, frame_filter)

    selected = []
    only = set(args.only or [])
    for spec in algo_config["algorithms"]:
        if not spec.get("enabled", True):
            continue
        if only and spec["name"] not in only:
            continue
        selected.append(spec)
    if args.max_algorithms > 0:
        selected = selected[: args.max_algorithms]

    results: List[AlgoResult] = []

    for spec in selected:
        name = spec["name"]
        pred_root = output_root / "predictions" / name
        ensure_dir(pred_root)
        clear_json_files(pred_root)

        if args.dry_run:
            results.append(AlgoResult(name=name, status="dry_run", message="not executed"))
            continue

        try:
            if spec["type"] == "synthetic":
                create_synthetic_predictions(spec["mode"], gt_root, pred_root, flat_scene_name)
                src_hint = spec["mode"]
            elif spec["type"] == "mmdet3d_kitti":
                src_hint, _ = run_mmdet3d(spec, pred_root, args, flat_scene_name, frame_filter)
            elif spec["type"] == "openpcdet_kitti":
                src_hint, _ = run_openpcdet(spec, pred_root, flat_scene_name, frame_filter)
            elif spec["type"] == "open3dml_kitti":
                src_hint, _ = run_open3dml(spec, pred_root, flat_scene_name)
            else:
                raise NotImplementedError(spec["type"])

            evaluate_predictions(gt_root, pred_root, output_root, name, args, subset_csv)
            summary = summarize_eval(output_root, name)
            summary.pred_root = str(pred_root)
            summary.message = src_hint
            results.append(summary)
        except FileNotFoundError as e:
            results.append(
                AlgoResult(
                    name=name,
                    status="skipped",
                    message=str(e),
                    pred_root=str(pred_root),
                )
            )
        except subprocess.CalledProcessError as e:
            results.append(
                AlgoResult(
                    name=name,
                    status="failed",
                    message=f"command failed with exit {e.returncode}",
                    pred_root=str(pred_root),
                )
            )
        except Exception as e:
            results.append(
                AlgoResult(
                    name=name,
                    status="failed",
                    message=repr(e),
                    pred_root=str(pred_root),
                )
            )

    write_reports(results, output_root)
    write_json(
        output_root / "run_manifest.json",
        {
            "scene_dir": str(scene_dir),
            "gt_root": str(gt_root),
            "algorithms": [r.__dict__ for r in results],
            "frame_list": str(args.frame_list) if args.frame_list else "",
            "frame_count": len(frame_filter or []),
            "iou_threshold": args.iou_threshold,
            "score_threshold": args.score_threshold,
            "class_aware": args.class_aware,
        },
    )
    print(f"wrote report: {output_root / 'algorithm_error_report.md'}")


if __name__ == "__main__":
    main()

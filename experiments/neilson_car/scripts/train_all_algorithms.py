#!/usr/bin/env python3
from __future__ import annotations

import argparse
import codecs
from datetime import datetime
import json
import os
import re
import shlex
import signal
import selectors
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = WORKSPACE_ROOT / "experiments" / "neilson_car" / "configs"
CHECKPOINT_ROOT = WORKSPACE_ROOT / "experiments" / "neilson_car" / "checkpoints"
RUN_ROOT = WORKSPACE_ROOT / "experiments" / "neilson_car" / "runs"
LOG_ROOT = WORKSPACE_ROOT / "logs"
PROGRESS_RE = re.compile(r"^\[[<>=\-\s>]+\]\s+\d+/\d+,")
TQDM_PROGRESS_RE = re.compile(r"(?:^|\s)\d{1,3}%\|.*\|\s*\d+/\d+\s*\[")
PROGRESS_MIN_REDRAW_SECONDS = 0.25
CONDA_SH = Path.home() / "miniforge3" / "etc" / "profile.d" / "conda.sh"
MMDET3D_ROOT = Path.home() / "src" / "mmdetection3d"
OPENPCDET_PATCHED_ROOT = Path("/tmp/openpcdet_patched")
MIRROR_WORKSPACE_ROOT = Path("/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics")


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _python_runpy(script_path: Path | str, *argv: object) -> str:
    args = [str(script_path), *(str(value) for value in argv)]
    payload = ", ".join(repr(item) for item in args)
    code = f'import runpy, sys; sys.argv = [{payload}]; runpy.run_path(sys.argv[0], run_name="__main__")'
    return f"python -c {shlex.quote(code)}"


def _require_cuda_command(label: str, *, require_arch: bool = False) -> str:
    argv: list[object] = [
        "--label",
        label,
    ]
    if require_arch:
        argv.append("--require-arch")
    return _python_runpy(
        WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "require_cuda.py",
        *argv,
    )


def _guard_openpcdet_runtime_command(config_path: Path, label: str) -> str:
    return _python_runpy(
        WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "guard_openpcdet_runtime.py",
        "--cfg-file",
        config_path,
        "--label",
        label,
    )


def _join_shell(*parts: str) -> str:
    return " && ".join(part for part in parts if part)


def _optional_cuda_command(label: str, *, require_arch: bool, enabled: bool) -> str:
    if not enabled:
        return ""
    return _require_cuda_command(label, require_arch=require_arch)


def _workspace_targets() -> list[Path]:
    targets = [WORKSPACE_ROOT]
    if MIRROR_WORKSPACE_ROOT != WORKSPACE_ROOT and MIRROR_WORKSPACE_ROOT.is_dir():
        targets.append(MIRROR_WORKSPACE_ROOT)
    return targets


def _copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _write_json_targets(rel_path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2) + "\n"
    for root in _workspace_targets():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")


def _persist_openpcdet_run(name: str, run_tag: str, epochs: int) -> None:
    run_map = {
        "openpcdet_pointpillar_car": ("openpcdet_pointpillar_car.subset", f"neilson_pointpillar_{run_tag}"),
        "openpcdet_second_car": ("openpcdet_second_car.subset", f"neilson_second_{run_tag}"),
    }
    if name not in run_map:
        return

    tag_root, extra_tag = run_map[name]
    src_run_dir = OPENPCDET_PATCHED_ROOT / "output" / "tmp" / tag_root / extra_tag
    if not src_run_dir.is_dir():
        return

    latest_ckpt = src_run_dir / "ckpt" / f"checkpoint_epoch_{epochs}.pth"
    summary = {
        "algorithm": name,
        "run_tag": run_tag,
        "source_run_dir": str(src_run_dir),
        "latest_checkpoint": str(latest_ckpt) if latest_ckpt.is_file() else "",
        "has_eval_dir": (src_run_dir / "eval").is_dir(),
    }

    for root in _workspace_targets():
        dst_run_dir = root / "experiments" / "neilson_car" / "runs" / extra_tag
        _copy_tree_contents(src_run_dir, dst_run_dir)

        if latest_ckpt.is_file():
            checkpoint_dir = root / "experiments" / "neilson_car" / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            finetuned_ckpt = checkpoint_dir / f"{extra_tag}_epoch_{epochs}.pth"
            shutil.copy2(latest_ckpt, finetuned_ckpt)
            summary[f"persistent_checkpoint@{root}"] = str(finetuned_ckpt)

    _write_json_targets(
        Path("experiments") / "neilson_car" / "runs" / extra_tag / "training_summary.json",
        summary,
    )


def _openpcdet_command(
    *,
    config_path: Path,
    pretrained_model: Path,
    extra_tag: str,
    epochs: int,
    workers: int,
    freeze_modules: str = "",
) -> str:
    return _join_shell(
        f"source {_q(CONDA_SH)}",
        "conda activate lidar-openpcdet",
        "export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS-}",
        f"export NEILSON_CAR_KITTI_ROOT={_q(WORKSPACE_ROOT / 'data' / 'neilson_car_kitti')}",
        f"export OPENPCDET_FREEZE_MODULES={_q(freeze_modules)}",
        _python_runpy(WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "bootstrap_openpcdet_runtime.py"),
        "export PYTHONPATH=/tmp/openpcdet_patched:/tmp/openpcdet_extra_py${PYTHONPATH:+:$PYTHONPATH}",
        _require_cuda_command("OpenPCDet (lidar-openpcdet)", require_arch=True),
        _guard_openpcdet_runtime_command(config_path, "OpenPCDet (lidar-openpcdet)"),
        "test -d /tmp/openpcdet_patched/tools",
        _python_runpy(WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "prepare_openpcdet_infos.py"),
        f"cd {_q(OPENPCDET_PATCHED_ROOT / 'tools')}",
        _python_runpy(
            OPENPCDET_PATCHED_ROOT / "tools" / "train.py",
            "--cfg_file",
            config_path,
            "--extra_tag",
            extra_tag,
            "--workers",
            workers,
            "--batch_size",
            4,
            "--epochs",
            epochs,
            "--pretrained_model",
            pretrained_model,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Neilson training across all supported 3D frameworks.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "mmdet3d_pointpillars_car",
            "openpcdet_pointpillar_car",
            "openpcdet_second_car",
            "open3dml_pointpillars_kitti",
        ],
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--open3d-device", default="cuda", choices=["cuda"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-tag", default="finetune")
    return parser.parse_args()


def recipe_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    mmdet_work_dir = RUN_ROOT / "mmdet3d_pointpillars_car_finetune"
    pointpillar_tag = f"neilson_pointpillar_{args.run_tag}"
    second_tag = f"neilson_second_{args.run_tag}"
    open3d_run_root = RUN_ROOT / "open3dml_pointpillars_car_finetune"

    return {
        "mmdet3d_pointpillars_car": [
            "bash",
            "-lc",
            _join_shell(
                f"source {_q(CONDA_SH)}",
                "conda activate lidar-3d-base",
                f"export NEILSON_CAR_KITTI_ROOT={_q(WORKSPACE_ROOT / 'data' / 'neilson_car_kitti')}",
                "export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS-}",
                "export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}",
                "export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/.cache}",
                f"export MMDET3D_LOAD_FROM={_q(CHECKPOINT_ROOT / 'mmdet3d_pointpillars_kitti.pth')}",
                f"export MMDET3D_TRAIN_WORKERS={_q(args.workers)}",
                f"export MMDET3D_VAL_WORKERS={_q(args.workers)}",
                f"export MMDET3D_VAL_INTERVAL={_q(args.epochs + 1)}",
                _require_cuda_command("MMDetection3D (lidar-3d-base)", require_arch=True),
                _python_runpy(WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "prepare_mmdet_infos.py"),
                f"mkdir -p {_q(mmdet_work_dir)}",
                f"cd {_q(MMDET3D_ROOT)}",
                f"PYTHONPATH={_q(MMDET3D_ROOT)} "
                + _python_runpy(
                    MMDET3D_ROOT / "tools" / "train.py",
                    WORKSPACE_ROOT / "experiments" / "neilson_car" / "configs" / "mmdet3d_pointpillars_car.py",
                    "--work-dir",
                    mmdet_work_dir,
                    "--cfg-options",
                    f"train_cfg.val_interval={args.epochs + 1}",
                    f"train_cfg.max_epochs={args.epochs}",
                ),
            ),
        ],
        "openpcdet_pointpillar_car": [
            "bash",
            "-lc",
            _openpcdet_command(
                config_path=CONFIG_ROOT / "openpcdet_pointpillar_car.yaml",
                pretrained_model=CHECKPOINT_ROOT / "openpcdet_pointpillar_kitti.pth",
                extra_tag=pointpillar_tag,
                epochs=args.epochs,
                workers=args.workers,
                freeze_modules="",
            ),
        ],
        "openpcdet_second_car": [
            "bash",
            "-lc",
            _openpcdet_command(
                config_path=CONFIG_ROOT / "openpcdet_second_car.yaml",
                pretrained_model=CHECKPOINT_ROOT / "openpcdet_second_kitti.pth",
                extra_tag=second_tag,
                epochs=args.epochs,
                workers=args.workers,
                freeze_modules="vfe,backbone_3d",
            ),
        ],
        "open3dml_pointpillars_kitti": [
            "bash",
            "-lc",
            _join_shell(
                f"source {_q(CONDA_SH)}",
                "conda activate lidar-open3d-ml",
                "export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS-}",
                f"export NEILSON_CAR_KITTI_ROOT={_q(WORKSPACE_ROOT / 'data' / 'neilson_car_kitti')}",
                _optional_cuda_command(
                    "Open3D-ML (lidar-open3d-ml)",
                    require_arch=True,
                    enabled=args.open3d_device == "cuda",
                ),
                _python_runpy(WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "prepare_open3dml_dataset.py"),
                _python_runpy(
                    WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "train_open3dml_pointpillars.py",
                    "--config",
                    CONFIG_ROOT / "open3dml_pointpillars_car.yml",
                    "--data-root",
                    WORKSPACE_ROOT / "data" / "neilson_car_kitti",
                    "--checkpoint",
                    CHECKPOINT_ROOT / "open3dml_pointpillars_kitti.pth",
                    "--run-root",
                    open3d_run_root,
                    "--device",
                    args.open3d_device,
                    "--val-split",
                    2687,
                    "--max-epoch",
                    args.epochs,
                    "--num-workers",
                    args.workers,
                ),
            ),
        ],
}


def emit(message: str, log_handle) -> None:
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def emit_timed(message: str, log_handle) -> None:
    emit(f"[{timestamp_now()}] {message}", log_handle)


def play_failure_sound(log_handle) -> None:
    commands = [
        ["paplay", "/usr/share/sounds/freedesktop/stereo/dialog-error.oga"],
        ["canberra-gtk-play", "--id=dialog-error"],
        ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
    ]
    for cmd in commands:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            emit_timed(f"failure alert via {' '.join(cmd)}", log_handle)
            return
        except FileNotFoundError:
            continue
    sys.stdout.write("\a")
    sys.stdout.flush()
    log_handle.write("[terminal-bell]\n")
    log_handle.flush()


def _new_progress_state() -> dict[str, object]:
    return {
        "active": False,
        "last_line": "",
        "last_redraw": 0.0,
    }


def _flush_terminal_progress(progress_state: dict[str, object]) -> dict[str, object]:
    if progress_state["active"]:
        sys.stdout.write("\n")
        sys.stdout.flush()
    progress_state["active"] = False
    progress_state["last_line"] = ""
    progress_state["last_redraw"] = 0.0
    return progress_state


def _handle_stream_text(text: str, log_handle, progress_state: dict[str, object]) -> dict[str, object]:
    stripped = text.rstrip("\r\n")
    if not stripped:
        return progress_state
    if PROGRESS_RE.match(stripped) or TQDM_PROGRESS_RE.search(stripped):
        now = time.monotonic()
        is_final_update = "100%" in stripped
        last_line = str(progress_state["last_line"])
        last_redraw = float(progress_state["last_redraw"])
        should_redraw = (
            not progress_state["active"]
            or stripped != last_line
            and (is_final_update or now - last_redraw >= PROGRESS_MIN_REDRAW_SECONDS)
        )
        if should_redraw:
            sys.stdout.write("\r\033[K" + stripped)
            sys.stdout.flush()
            progress_state["last_redraw"] = now
            progress_state["last_line"] = stripped
        progress_state["active"] = True
        return progress_state
    progress_state = _flush_terminal_progress(progress_state)
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()
    log_handle.write(text if text.endswith("\n") else text + "\n")
    log_handle.flush()
    return progress_state


def _drain_pending_text(
    pending: str, log_handle, progress_state: dict[str, object]
) -> tuple[str, dict[str, object]]:
    while True:
        newline_pos = pending.find("\n")
        carriage_pos = pending.find("\r")
        positions = [pos for pos in (newline_pos, carriage_pos) if pos != -1]
        if not positions:
            break
        split_at = min(positions)
        text = pending[:split_at]
        separator = pending[split_at]
        pending = pending[split_at + 1 :]
        if separator == "\n":
            progress_state = _handle_stream_text(text + "\n", log_handle, progress_state)
        else:
            progress_state = _handle_stream_text(text, log_handle, progress_state)
    return pending, progress_state


def _terminate_process(process: subprocess.Popen[bytes], log_handle) -> None:
    if process.poll() is not None:
        return

    emit_timed(f"interrupt received, stopping pid {process.pid}", log_handle)
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        emit_timed(f"pid {process.pid} did not exit after SIGINT; terminating", log_handle)

    process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        emit_timed(f"pid {process.pid} did not exit after terminate; killing", log_handle)
        process.kill()
        process.wait()


def run_with_live_log(cmd: list[str], env: dict[str, str], log_handle) -> None:
    process = subprocess.Popen(
        cmd,
        cwd=str(WORKSPACE_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )

    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    progress_state = _new_progress_state()
    try:
        while True:
            events = selector.select(timeout=0.2)
            if not events:
                if process.poll() is not None:
                    break
                continue

            for key, _ in events:
                chunk = os.read(key.fd, 4096)
                if not chunk:
                    selector.unregister(process.stdout)
                    break
                pending += decoder.decode(chunk)
                pending, progress_state = _drain_pending_text(pending, log_handle, progress_state)
    except KeyboardInterrupt:
        progress_state = _flush_terminal_progress(progress_state)
        _terminate_process(process, log_handle)
        raise
    finally:
        selector.close()

    pending += decoder.decode(b"", final=True)
    if pending:
        progress_state = _handle_stream_text(pending, log_handle, progress_state)
    progress_state = _flush_terminal_progress(progress_state)

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _format_process_failure(exc: subprocess.CalledProcessError) -> str:
    if exc.returncode < 0:
        signal_num = -exc.returncode
        try:
            signal_name = signal.Signals(signal_num).name
        except ValueError:
            signal_name = f"SIG{signal_num}"
        return f"subprocess terminated by signal {signal_num} ({signal_name})"
    return f"subprocess exited with status {exc.returncode}"


def _write_training_manifest(
    out_path: Path,
    *,
    run_manifest: list[dict[str, object]],
    epochs: int,
    run_tag: str,
    log_path: Path,
    started_at: str,
    total_start: float,
    final_status: str,
) -> None:
    out_path.write_text(
        json.dumps(
            {
                "recipes": run_manifest,
                "epochs": epochs,
                "run_tag": run_tag,
                "log_path": str(log_path),
                "started_at": started_at,
                "updated_at": timestamp_now(),
                "total_elapsed": format_duration(time.monotonic() - total_start),
                "status": final_status,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    commands = recipe_commands(args)
    run_manifest: list[dict[str, object]] = []
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = timestamp_now()
    log_path = LOG_ROOT / f"training_log_{timestamp}.log"
    total_start = time.monotonic()
    out_path = WORKSPACE_ROOT / "experiments" / "neilson_car" / "training_manifest.json"

    with log_path.open("w", encoding="utf-8") as log_handle:
        emit_timed(f"logging to {log_path}", log_handle)
        emit_timed(f"workspace root: {WORKSPACE_ROOT}", log_handle)
        emit_timed(
            "command: " + " ".join(_q(part) for part in [sys.executable, *sys.argv]),
            log_handle,
        )
        _write_training_manifest(
            out_path,
            run_manifest=run_manifest,
            epochs=args.epochs,
            run_tag=args.run_tag,
            log_path=log_path,
            started_at=started_at,
            total_start=total_start,
            final_status="running",
        )

        for name in args.algorithms:
            if name not in commands:
                raise SystemExit(f"Unsupported training target: {name}")
            cmd = commands[name]
            env = os.environ.copy()
            entry: dict[str, object] = {
                "name": name,
                "cmd": cmd,
                "status": "running",
                "started_at": timestamp_now(),
            }
            run_manifest.append(entry)
            _write_training_manifest(
                out_path,
                run_manifest=run_manifest,
                epochs=args.epochs,
                run_tag=args.run_tag,
                log_path=log_path,
                started_at=started_at,
                total_start=total_start,
                final_status="running",
            )
            algo_start = time.monotonic()
            emit_timed(f"=== [{name}] starting ===", log_handle)
            if args.dry_run:
                entry["status"] = "dry-run"
                entry["completed_at"] = timestamp_now()
                entry["elapsed"] = "00:00"
                emit_timed(f"[dry-run] {' '.join(cmd)}", log_handle)
                continue
            try:
                run_with_live_log(cmd, env, log_handle)
            except KeyboardInterrupt:
                elapsed = format_duration(time.monotonic() - algo_start)
                entry["status"] = "interrupted"
                entry["completed_at"] = timestamp_now()
                entry["elapsed"] = elapsed
                emit_timed(f"=== [{name}] interrupted after {elapsed} ===", log_handle)
                _write_training_manifest(
                    out_path,
                    run_manifest=run_manifest,
                    epochs=args.epochs,
                    run_tag=args.run_tag,
                    log_path=log_path,
                    started_at=started_at,
                    total_start=total_start,
                    final_status="interrupted",
                )
                raise
            except subprocess.CalledProcessError as exc:
                elapsed = format_duration(time.monotonic() - algo_start)
                entry["status"] = "failed"
                entry["completed_at"] = timestamp_now()
                entry["elapsed"] = elapsed
                entry["failure_reason"] = _format_process_failure(exc)
                emit_timed(entry["failure_reason"], log_handle)
                emit_timed(f"=== [{name}] failed after {elapsed} ===", log_handle)
                _write_training_manifest(
                    out_path,
                    run_manifest=run_manifest,
                    epochs=args.epochs,
                    run_tag=args.run_tag,
                    log_path=log_path,
                    started_at=started_at,
                    total_start=total_start,
                    final_status="failed",
                )
                play_failure_sound(log_handle)
                raise
            _persist_openpcdet_run(name, args.run_tag, args.epochs)
            elapsed = format_duration(time.monotonic() - algo_start)
            entry["status"] = "completed"
            entry["completed_at"] = timestamp_now()
            entry["elapsed"] = elapsed
            emit_timed(f"=== [{name}] completed in {elapsed} ===", log_handle)
            _write_training_manifest(
                out_path,
                run_manifest=run_manifest,
                epochs=args.epochs,
                run_tag=args.run_tag,
                log_path=log_path,
                started_at=started_at,
                total_start=total_start,
                final_status="running",
            )

    _write_training_manifest(
        out_path,
        run_manifest=run_manifest,
        epochs=args.epochs,
        run_tag=args.run_tag,
        log_path=log_path,
        started_at=started_at,
        total_start=total_start,
        final_status="completed",
    )
    print(f"wrote {out_path}")
    print(f"log saved to {log_path}")
    print(f"total elapsed: {format_duration(time.monotonic() - total_start)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import codecs
from datetime import datetime
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import sys
import time
from typing import Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = WORKSPACE_ROOT / "experiments" / "neilson_car" / "configs" / "openpcdet_second_car.yaml"
CHECKPOINT_PATH = WORKSPACE_ROOT / "experiments" / "neilson_car" / "checkpoints" / "openpcdet_second_kitti.pth"
CONDA_SH = Path.home() / "miniforge3" / "etc" / "profile.d" / "conda.sh"
OPENPCDET_PATCHED_ROOT = Path("/tmp/openpcdet_patched")
LOG_ROOT = WORKSPACE_ROOT / "logs"
PROGRESS_RE = re.compile(r"^\[[<>=\-\s>]+\]\s+\d+/\d+,")
PROGRESS_MIN_REDRAW_SECONDS = 0.25


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _python_runpy(script_path: Path | str, *argv: object) -> str:
    args = [str(script_path), *(str(value) for value in argv)]
    payload = ", ".join(repr(item) for item in args)
    code = f'import runpy, sys; sys.argv = [{payload}]; runpy.run_path(sys.argv[0], run_name="__main__")'
    return f"python -X faulthandler -c {shlex.quote(code)}"


def _join_shell(*parts: str) -> str:
    return " && ".join(part for part in parts if part)


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def emit(message: str, log_handle) -> None:
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


def emit_timed(message: str, log_handle) -> None:
    emit(f"[{timestamp_now()}] {message}", log_handle)


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
    if PROGRESS_RE.match(stripped):
        now = time.monotonic()
        last_line = str(progress_state["last_line"])
        last_redraw = float(progress_state["last_redraw"])
        if (
            not progress_state["active"]
            or stripped != last_line and now - last_redraw >= PROGRESS_MIN_REDRAW_SECONDS
        ):
            sys.stdout.write("\r\033[K" + stripped)
            sys.stdout.flush()
            progress_state["last_line"] = stripped
            progress_state["last_redraw"] = now
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


def run_with_live_log(cmd: list[str], env: dict[str, str], log_handle, *, cwd: Path) -> int:
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
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
    finally:
        selector.close()

    pending += decoder.decode(b"", final=True)
    if pending:
        progress_state = _handle_stream_text(pending, log_handle, progress_state)
    _flush_terminal_progress(progress_state)
    return process.wait()


def format_return_code(return_code: int) -> str:
    if return_code < 0:
        signal_num = -return_code
        try:
            signal_name = signal.Signals(signal_num).name
        except ValueError:
            signal_name = f"SIG{signal_num}"
        return f"signal {signal_num} ({signal_name})"
    return f"exit status {return_code}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused diagnostics for OpenPCDet SECOND on the Neilson car dataset.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--keep-going", action="store_true", help="continue through the scenario matrix even after failures")
    return parser.parse_args()


def build_shell_env_exports(extra_env: dict[str, str]) -> Iterable[str]:
    for key, value in extra_env.items():
        yield f"export {key}={_q(value)}"


def build_common_prefix(extra_env: dict[str, str], freeze_modules: str) -> str:
    return _join_shell(
        f"source {_q(CONDA_SH)}",
        "conda activate lidar-openpcdet",
        "export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS-}",
        f"export NEILSON_CAR_KITTI_ROOT={_q(WORKSPACE_ROOT / 'data' / 'neilson_car_kitti')}",
        f"export OPENPCDET_FREEZE_MODULES={_q(freeze_modules)}",
        *build_shell_env_exports(extra_env),
        _python_runpy(WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "bootstrap_openpcdet_runtime.py"),
        "export PYTHONPATH=/tmp/openpcdet_patched:/tmp/openpcdet_extra_py${PYTHONPATH:+:$PYTHONPATH}",
    )


def build_env_probe_command(label: str) -> str:
    probe_code = """
import inspect
import os
import platform
import sys

print(f"[diag] python={sys.version.split()[0]}")
print(f"[diag] executable={sys.executable}")
print(f"[diag] platform={platform.platform()}")
print(f"[diag] cwd={os.getcwd()}")
for key in [
    "CUDA_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING",
    "TORCH_SHOW_CPP_STACKTRACES",
    "PYTHONFAULTHANDLER",
    "OPENPCDET_FREEZE_MODULES",
]:
    print(f"[diag] env:{key}={os.environ.get(key, '<unset>')}")

import torch
print(f"[diag] torch={torch.__version__}")
print(f"[diag] torch.version.cuda={torch.version.cuda}")
print(f"[diag] torch.cuda.is_available={torch.cuda.is_available()}")
print(f"[diag] torch.cuda.arch_list={torch.cuda.get_arch_list()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"[diag] gpu_name={props.name}")
    print(f"[diag] gpu_capability=sm_{props.major}{props.minor}")
    print(f"[diag] total_memory_bytes={props.total_memory}")

import spconv
import cumm
print(f"[diag] spconv={getattr(spconv, '__version__', 'unknown')}")
print(f"[diag] spconv_file={inspect.getfile(spconv)}")
print(f"[diag] cumm={getattr(cumm, '__version__', 'unknown')}")
print(f"[diag] cumm_file={inspect.getfile(cumm)}")
"""
    return _join_shell(
        build_common_prefix(
            {
                "PYTHONFAULTHANDLER": "1",
                "TORCH_SHOW_CPP_STACKTRACES": "1",
            },
            "vfe,backbone_3d",
        ),
        f"python -X faulthandler -c {_q(probe_code)}",
        _python_runpy(WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "require_cuda.py", "--label", label, "--require-arch"),
    )


def build_spconv_smoke_command() -> str:
    smoke_code = """
import inspect
import torch
import spconv.pytorch as spconv

print(f"[diag] spconv_smoke torch={torch.__version__}")
print(f"[diag] spconv_smoke module={inspect.getfile(spconv)}")
torch.manual_seed(0)
device = "cuda"
count = 2048
features = torch.randn((count, 4), device=device, dtype=torch.float32, requires_grad=True)
indices = torch.zeros((count, 4), dtype=torch.int32, device=device)
indices[:, 1] = torch.arange(count, device=device) % 8
indices[:, 2] = (torch.arange(count, device=device) // 8) % 8
indices[:, 3] = (torch.arange(count, device=device) // 64) % 8
tensor = spconv.SparseConvTensor(features, indices, spatial_shape=[8, 8, 8], batch_size=1)
module = spconv.SubMConv3d(4, 8, kernel_size=3, padding=1, bias=False).to(device)
output = module(tensor)
loss = output.features.square().mean()
loss.backward()
print(f"[diag] spconv_smoke output_shape={tuple(output.features.shape)}")
print(f"[diag] spconv_smoke grad_norm={float(features.grad.norm().item()):.6f}")
"""
    return _join_shell(
        build_common_prefix(
            {
                "PYTHONFAULTHANDLER": "1",
                "TORCH_SHOW_CPP_STACKTRACES": "1",
                "CUDA_LAUNCH_BLOCKING": "1",
            },
            "vfe,backbone_3d",
        ),
        _python_runpy(WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "require_cuda.py", "--label", "OpenPCDet SECOND spconv smoke", "--require-arch"),
        f"cd {_q(OPENPCDET_PATCHED_ROOT / 'tools')}",
        f"python -X faulthandler -c {_q(smoke_code)}",
    )


def build_runtime_guard_command(label: str) -> str:
    return _join_shell(
        build_common_prefix(
            {
                "PYTHONFAULTHANDLER": "1",
                "TORCH_SHOW_CPP_STACKTRACES": "1",
            },
            "vfe,backbone_3d",
        ),
        _python_runpy(
            WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "guard_openpcdet_runtime.py",
            "--cfg-file",
            CONFIG_PATH,
            "--label",
            label,
        ),
    )


def build_train_command(*, extra_tag: str, workers: int, batch_size: int, freeze_modules: str, epochs: int) -> str:
    return _join_shell(
        build_common_prefix(
            {
                "PYTHONFAULTHANDLER": "1",
                "TORCH_SHOW_CPP_STACKTRACES": "1",
                "CUDA_LAUNCH_BLOCKING": "1",
            },
            freeze_modules,
        ),
        _python_runpy(WORKSPACE_ROOT / "experiments" / "neilson_car" / "scripts" / "require_cuda.py", "--label", f"OpenPCDet SECOND train {extra_tag}", "--require-arch"),
        f"cd {_q(OPENPCDET_PATCHED_ROOT / 'tools')}",
        _python_runpy(
            OPENPCDET_PATCHED_ROOT / "tools" / "train.py",
            "--cfg_file",
            CONFIG_PATH,
            "--extra_tag",
            extra_tag,
            "--workers",
            workers,
            "--batch_size",
            batch_size,
            "--epochs",
            epochs,
            "--pretrained_model",
            CHECKPOINT_PATH,
            "--logger_iter_interval",
            1,
            "--ckpt_save_interval",
            1,
            "--wo_gpu_stat",
            "--fix_random_seed",
        ),
    )


def main() -> int:
    args = parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path or LOG_ROOT / f"diagnose_openpcdet_second_{timestamp}.log"
    scenarios = [
        {
            "name": "env_probe",
            "cmd": ["bash", "-lc", build_env_probe_command("OpenPCDet SECOND diagnostics")],
            "cwd": WORKSPACE_ROOT,
        },
        {
            "name": "runtime_guard",
            "cmd": ["bash", "-lc", build_runtime_guard_command("OpenPCDet SECOND diagnostics")],
            "cwd": WORKSPACE_ROOT,
        },
        {
            "name": "spconv_smoke",
            "cmd": ["bash", "-lc", build_spconv_smoke_command()],
            "cwd": WORKSPACE_ROOT,
        },
        {
            "name": "train_workers12_freeze",
            "cmd": ["bash", "-lc", build_train_command(extra_tag="diag_w12_freeze", workers=12, batch_size=4, freeze_modules="vfe,backbone_3d", epochs=args.epochs)],
            "cwd": WORKSPACE_ROOT,
        },
        {
            "name": "train_workers1_freeze",
            "cmd": ["bash", "-lc", build_train_command(extra_tag="diag_w1_freeze", workers=1, batch_size=4, freeze_modules="vfe,backbone_3d", epochs=args.epochs)],
            "cwd": WORKSPACE_ROOT,
        },
        {
            "name": "train_workers0_freeze",
            "cmd": ["bash", "-lc", build_train_command(extra_tag="diag_w0_freeze", workers=0, batch_size=4, freeze_modules="vfe,backbone_3d", epochs=args.epochs)],
            "cwd": WORKSPACE_ROOT,
        },
        {
            "name": "train_workers0_no_freeze",
            "cmd": ["bash", "-lc", build_train_command(extra_tag="diag_w0_nofreeze", workers=0, batch_size=4, freeze_modules="", epochs=args.epochs)],
            "cwd": WORKSPACE_ROOT,
        },
        {
            "name": "train_workers0_batch1_freeze",
            "cmd": ["bash", "-lc", build_train_command(extra_tag="diag_w0_b1_freeze", workers=0, batch_size=1, freeze_modules="vfe,backbone_3d", epochs=args.epochs)],
            "cwd": WORKSPACE_ROOT,
        },
    ]

    overall_status = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        emit_timed(f"logging to {log_path}", log_handle)
        emit_timed(f"workspace root: {WORKSPACE_ROOT}", log_handle)
        emit_timed(f"python executable: {sys.executable}", log_handle)
        emit_timed(f"config path: {CONFIG_PATH}", log_handle)
        emit_timed(f"checkpoint path: {CHECKPOINT_PATH}", log_handle)
        for scenario in scenarios:
            emit_timed(f"=== [{scenario['name']}] starting ===", log_handle)
            emit_timed("command: " + " ".join(_q(part) for part in scenario["cmd"]), log_handle)
            env = os.environ.copy()
            return_code = run_with_live_log(scenario["cmd"], env, log_handle, cwd=scenario["cwd"])
            if return_code == 0:
                emit_timed(f"=== [{scenario['name']}] completed successfully ===", log_handle)
                continue

            overall_status = return_code
            emit_timed(f"[{scenario['name']}] failed with {format_return_code(return_code)}", log_handle)
            emit_timed(f"=== [{scenario['name']}] failed ===", log_handle)
            if not args.keep_going:
                break

        emit_timed(f"diagnostic run finished with {format_return_code(overall_status) if overall_status else 'success'}", log_handle)

    return 0 if overall_status == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

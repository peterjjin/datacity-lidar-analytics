#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail fast unless PyTorch CUDA is available.")
    parser.add_argument("--label", default="training environment")
    parser.add_argument("--require-arch", action="store_true")
    return parser.parse_args()


def emit_driver_diagnostics(label: str) -> None:
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        print(f"[cuda-check] {label}: failed to load libcuda.so.1: {exc}", file=sys.stderr)
        return

    cu_init = libcuda.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    rc = cu_init(0)
    print(f"[cuda-check] {label}: libcuda cuInit(0) -> {rc}", file=sys.stderr)

    if Path("/dev/dxg").exists():
        print(f"[cuda-check] {label}: /dev/dxg is present", file=sys.stderr)
    else:
        print(f"[cuda-check] {label}: /dev/dxg is missing", file=sys.stderr)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"[cuda-check] {label}: nvidia-smi -> {result.stdout.strip()}", file=sys.stderr)
            else:
                stderr = result.stderr.strip() or f"exit {result.returncode}"
                print(f"[cuda-check] {label}: nvidia-smi check failed: {stderr}", file=sys.stderr)
        except Exception as exc:
            print(f"[cuda-check] {label}: nvidia-smi probe failed: {exc!r}", file=sys.stderr)

    if rc == 304:
        print(
            f"[cuda-check] {label}: CUDA driver init failed before PyTorch could see the GPU. "
            "In WSL this usually indicates a host driver / dxg / firmware issue rather than a training-script bug.",
            file=sys.stderr,
        )
        print(
            f"[cuda-check] {label}: Try `wsl --shutdown`, reopen Ubuntu, then retest. "
            "If cuInit still returns 304, check the Windows NVIDIA driver, WSL updates, and BIOS GPU settings such as Resizable BAR.",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"[cuda-check] {args.label}: torch import failed: {exc!r}", file=sys.stderr)
        return 2

    print(f"[cuda-check] {args.label}: torch={torch.__version__}")
    print(f"[cuda-check] {args.label}: torch.version.cuda={torch.version.cuda}")

    if not torch.cuda.is_available():
        print(f"[cuda-check] {args.label}: torch.cuda.is_available() is False", file=sys.stderr)
        emit_driver_diagnostics(args.label)
        return 1

    try:
        device_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
    except Exception as exc:
        print(f"[cuda-check] {args.label}: CUDA probe failed: {exc!r}", file=sys.stderr)
        return 1

    print(f"[cuda-check] {args.label}: gpu={device_name}")
    print(f"[cuda-check] {args.label}: arch={arch}")

    if args.require_arch and arch not in torch.cuda.get_arch_list():
        print(
            f"[cuda-check] {args.label}: active architecture {arch} is missing from "
            f"{torch.cuda.get_arch_list()}",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

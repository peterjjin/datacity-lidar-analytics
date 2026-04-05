#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
NATIVE_LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "neilson_car"
    / "scripts"
    / "train_all_algorithms_native.sh"
)

DEFAULT_ALGORITHMS = [
    "mmdet3d_pointpillars_car",
    "openpcdet_pointpillar_car",
    "openpcdet_second_car",
    "open3dml_pointpillars_kitti",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top-level wrapper for the Neilson native multi-framework trainer."
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--run-tag", default="finetune")
    parser.add_argument("--open3d-device", default="cuda")
    parser.add_argument("--data-root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        help="Optional list of algorithm names, matching the legacy wrapper.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        default=[],
        help="Repeatable single algorithm flag, matching the native launcher.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    algorithms = list(args.algorithm)
    if args.algorithms:
        algorithms.extend(args.algorithms)
    if not algorithms:
        algorithms = list(DEFAULT_ALGORITHMS)

    command: list[str] = [
        "bash",
        str(NATIVE_LAUNCHER),
        "--epochs",
        str(args.epochs),
        "--workers",
        str(args.workers),
        "--run-tag",
        args.run_tag,
        "--open3d-device",
        args.open3d_device,
    ]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    if args.dry_run:
        command.append("--dry-run")
    for algorithm in algorithms:
        command.extend(["--algorithm", algorithm])
    return command


def main() -> int:
    args = parse_args()
    if not NATIVE_LAUNCHER.is_file():
        print(f"missing launcher: {NATIVE_LAUNCHER}", file=sys.stderr)
        return 2

    command = build_command(args)
    print("launching:", " ".join(shlex.quote(part) for part in command))

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

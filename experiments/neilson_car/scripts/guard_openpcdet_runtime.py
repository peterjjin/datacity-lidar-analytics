#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SECOND_NAME_RE = re.compile(r"^\s*NAME:\s*SECONDNet\s*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail fast for known-bad OpenPCDet runtime/model combinations."
    )
    parser.add_argument("--cfg-file", required=True, type=Path)
    parser.add_argument("--label", default="OpenPCDet guard")
    return parser.parse_args()


def config_uses_second(cfg_file: Path) -> bool:
    return bool(SECOND_NAME_RE.search(cfg_file.read_text(encoding="utf-8")))


def main() -> int:
    args = parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"[openpcdet-guard] {args.label}: torch import failed: {exc!r}", file=sys.stderr)
        return 2

    if not args.cfg_file.is_file():
        print(f"[openpcdet-guard] {args.label}: missing config: {args.cfg_file}", file=sys.stderr)
        return 2

    if not config_uses_second(args.cfg_file):
        print(f"[openpcdet-guard] {args.label}: config is not SECONDNet; continuing")
        return 0

    if not torch.cuda.is_available():
        print(f"[openpcdet-guard] {args.label}: CUDA is unavailable; continuing to existing CUDA checks")
        return 0

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    device_name = torch.cuda.get_device_name(0)
    print(f"[openpcdet-guard] {args.label}: detected {device_name} ({arch})")

    if arch == "sm_120":
        print(
            f"[openpcdet-guard] {args.label}: blocking launch for SECONDNet on {arch}. "
            "The latest diagnostic run reproduced a deterministic SIGFPE inside spconv/cumm "
            "during the standalone sparse-convolution smoke test and again during the first "
            "forward/backward pass, so OPENPCDET_FREEZE_MODULES is not a working fix in this environment.",
            file=sys.stderr,
        )
        print(
            f"[openpcdet-guard] {args.label}: use openpcdet_pointpillar_car or a different GPU/toolchain "
            "until spconv/cumm is updated for this stack.",
            file=sys.stderr,
        )
        return 4

    print(f"[openpcdet-guard] {args.label}: SECONDNet runtime check passed for {arch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

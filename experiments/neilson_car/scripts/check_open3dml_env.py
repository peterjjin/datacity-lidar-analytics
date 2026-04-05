#!/usr/bin/env python3
from __future__ import annotations

import traceback

import open3d


def main() -> None:
    print(f"open3d={open3d.__version__}")
    try:
        import torch

        print(f"torch={torch.__version__}")
    except Exception as exc:
        print(f"torch_import_failed={exc!r}")

    try:
        import open3d.ml.torch as ml3d  # noqa: F401

        print("open3d_ml_torch=ok")
    except Exception as exc:
        print(f"open3d_ml_torch_failed={exc!r}")
        print("traceback:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

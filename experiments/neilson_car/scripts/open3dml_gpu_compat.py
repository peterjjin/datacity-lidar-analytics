#!/usr/bin/env python3
from __future__ import annotations

import logging

import torch


_LOG = logging.getLogger(__name__)
_PATCHED = False
_EMPTY_VOXEL_WARNINGS = 0


def _ragged_to_dense_torch(
    values: torch.Tensor,
    row_splits: torch.Tensor,
    out_col_size: int,
    default_value,
) -> torch.Tensor:
    device = values.device
    values_cpu = values.detach().to(device="cpu")
    row_splits_cpu = row_splits.detach().to(device="cpu", dtype=torch.int64)
    rows = max(int(row_splits_cpu.numel()) - 1, 0)
    out_col_size = int(out_col_size)
    tail_shape = tuple(values_cpu.shape[1:])

    default = torch.as_tensor(default_value, device="cpu", dtype=values_cpu.dtype)
    out = default.expand((rows, out_col_size, *tail_shape)).clone()
    if rows == 0 or out_col_size == 0:
        return out.to(device=device)

    if row_splits_cpu.ndim != 1:
        raise RuntimeError(f"ragged_to_dense expects 1D row_splits, received shape={tuple(row_splits_cpu.shape)}")

    lengths = row_splits_cpu[1:] - row_splits_cpu[:-1]
    if torch.any(lengths < 0):
        raise RuntimeError("ragged_to_dense received non-monotonic row_splits from Open3D voxelization")

    if row_splits_cpu.numel() > 0:
        required = int(row_splits_cpu[-1].item())
        if required > int(values_cpu.shape[0]):
            raise RuntimeError(
                "ragged_to_dense received out-of-range row_splits from Open3D voxelization: "
                f"required={required}, values_rows={int(values_cpu.shape[0])}"
            )

    lengths = lengths.clamp(min=0)
    if not torch.any(lengths > 0):
        return out.to(device=device)

    col_index = torch.arange(out_col_size, device="cpu", dtype=torch.int64).unsqueeze(0)
    lengths = lengths.unsqueeze(1)
    mask = col_index < lengths
    if not torch.any(mask):
        return out.to(device=device)

    row_index, col_index = torch.nonzero(mask, as_tuple=True)
    value_index = row_splits_cpu[:-1].unsqueeze(1) + torch.arange(
        out_col_size, device="cpu", dtype=torch.int64
    ).unsqueeze(0)
    out[row_index, col_index] = values_cpu[value_index[mask]]
    return out.to(device=device)


def _pointpillars_voxelization_forward_cpu_fallback(self, points_feats: torch.Tensor):
    # Open3D 0.19's CUDA voxelize path is not reliable on RTX 5090 / sm_120.
    # Keep the rest of PointPillars on the model device and only materialize the
    # voxelization stage on CPU before copying the compact outputs back.
    from open3d.ml.torch.ops import voxelize

    device = points_feats.device
    points_feats_cpu = points_feats.detach().to(device="cpu")
    points_cpu = points_feats_cpu[:, :3]

    if self.training:
        max_voxels = self.max_voxels[0]
    else:
        max_voxels = self.max_voxels[1]

    num_voxels = ((self.points_range_max - self.points_range_min) / self.voxel_size).type(torch.int32)
    ans = voxelize(
        points_cpu,
        torch.tensor([0, points_cpu.shape[0]], device="cpu", dtype=torch.int64),
        self.voxel_size.to(device="cpu"),
        self.points_range_min.to(device="cpu"),
        self.points_range_max.to(device="cpu"),
        self.max_num_points,
        max_voxels,
    )

    feats = torch.cat([torch.zeros_like(points_feats_cpu[0:1, :]), points_feats_cpu])
    voxels_point_indices_dense = _ragged_to_dense_torch(
        ans.voxel_point_indices,
        ans.voxel_point_row_splits,
        self.max_num_points,
        torch.tensor(-1),
    ) + 1

    out_voxels = feats[voxels_point_indices_dense]
    out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
    out_num_points = ans.voxel_point_row_splits[1:] - ans.voxel_point_row_splits[:-1]

    in_bounds_y = out_coords[:, 1] < num_voxels[1]
    in_bounds_x = out_coords[:, 2] < num_voxels[0]
    in_bounds = torch.logical_and(in_bounds_x, in_bounds_y)

    out_coords = out_coords[in_bounds].to(device=device)
    out_voxels = out_voxels[in_bounds].to(device=device)
    out_num_points = out_num_points[in_bounds].to(device=device)
    return out_voxels, out_coords, out_num_points


def _pointpillars_extract_feats_safe(self, points):
    """Handle batches that voxelize to zero pillars without crashing PointPillars."""
    global _EMPTY_VOXEL_WARNINGS

    voxels, num_points, coors = self.voxelize(points)
    voxel_features = self.voxel_encoder(voxels, num_points, coors)

    if coors.numel() == 0:
        batch_size = len(points)
        if batch_size <= 0:
            raise RuntimeError("PointPillars received an empty batch.")

        if _EMPTY_VOXEL_WARNINGS < 5:
            point_counts = [int(sample.shape[0]) for sample in points]
            _LOG.warning(
                "PointPillars voxelization produced zero pillars; using an empty pseudo-image fallback "
                "(batch_size=%d, point_counts=%s).",
                batch_size,
                point_counts,
            )
            _EMPTY_VOXEL_WARNINGS += 1

        x = torch.zeros(
            (
                batch_size,
                self.middle_encoder.in_channels,
                self.middle_encoder.ny,
                self.middle_encoder.nx,
            ),
            dtype=voxel_features.dtype,
            device=points[0].device,
        )
    else:
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)

    x = self.backbone(x)
    x = self.neck(x)
    return x


def apply_open3dml_gpu_compat_patches() -> bool:
    global _PATCHED
    if _PATCHED:
        return False

    import open3d.ml.torch.ops as ml_ops
    import open3d.ml.torch.python.ops as py_ops
    from open3d._ml3d.torch.models import point_pillars

    ml_ops.ragged_to_dense = _ragged_to_dense_torch
    py_ops.ragged_to_dense = _ragged_to_dense_torch
    point_pillars.ragged_to_dense = _ragged_to_dense_torch
    point_pillars.PointPillarsVoxelization.forward = _pointpillars_voxelization_forward_cpu_fallback
    point_pillars.PointPillars.extract_feats = _pointpillars_extract_feats_safe

    _PATCHED = True
    _LOG.info("Applied Open3D-ML GPU compatibility patches for voxelize/ragged_to_dense via CPU fallback")
    return True

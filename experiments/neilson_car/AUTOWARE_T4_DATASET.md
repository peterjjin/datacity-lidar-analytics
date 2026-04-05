# Autoware / T4 Dataset Notes

Neilson is already normalized into a KITTI-style tree under
`data/neilson_car_kitti`. That is enough for MMDetection3D, OpenPCDet, and
Open3D-ML. Autoware's current training stack is different.

## What Autoware training expects

Autoware's current 3D model training path is the TIER IV / AWML pipeline:

- raw Autoware sensor recordings are converted into a **T4 dataset**
- AWML then creates `info` pickle files from that T4 dataset
- training is launched with `python tools/detection3d/train.py {config}`

This means Autoware does **not** train directly from packet captures or the
Neilson label JSON files.

## Does Autoware take `pcap` as model-training input?

No, not directly.

- Sensor packet captures or bags are source material for conversion.
- TIER IV's public dataset tooling converts **rosbag2 / MCAP** recordings into
  non-annotated or annotated T4 datasets.
- The T4 dataset keeps point clouds on disk in `data/LIDAR_CONCAT/*.pcd.bin`
  plus scene-level annotation tables in `annotation/*.json`.

For Neilson, the practical path is:

1. keep the existing KITTI export for the three framework baselines
2. export a T4-oriented intermediate dataset for Autoware
3. generate official T4 tables and info pickles with TIER IV tooling
4. train AWML / Autoware models from the generated T4 dataset

## How labels are organized for Autoware

Autoware / AWML organizes 3D labels as **scene-level tables**, not one JSON per
frame.

The important T4 annotation files are:

- `annotation/category.json`
- `annotation/instance.json`
- `annotation/sample.json`
- `annotation/sample_data.json`
- `annotation/sample_annotation.json`
- `annotation/ego_pose.json`
- `annotation/calibrated_sensor.json`
- `annotation/sensor.json`
- `annotation/log.json`
- `annotation/scene.json`
- `annotation/attribute.json`
- `annotation/visibility.json`

The actual 3D boxes live in `sample_annotation.json`, linked by tokens to
instances, categories, frames, ego poses, and sample data.

## Neilson mapping

Neilson KITTI export:

- point cloud: `training/velodyne/{frame}.bin`
- label: `training/label_2/{frame}.txt`
- calibration: `training/calib/{frame}.txt`

Autoware / T4 target:

- point cloud: `data/LIDAR_CONCAT/{frame}.pcd.bin`
- labels: scene tables under `annotation/*.json`

For the lidar payload, T4 expects float32 rows:

- `[x, y, z, intensity, ring_idx]`

Neilson currently has:

- `[x, y, z, intensity]`

The intermediate exporter adds a placeholder `ring_idx = -1`.

## Intermediate export in this repo

Use:

```bash
python3 experiments/neilson_car/scripts/prepare_autoware_t4_intermediate.py
```

This writes:

- `data/neilson_autoware_t4_intermediate/train/data/LIDAR_CONCAT/*.pcd.bin`
- `data/neilson_autoware_t4_intermediate/val/data/LIDAR_CONCAT/*.pcd.bin`
- `data/neilson_autoware_t4_intermediate/test/data/LIDAR_CONCAT/*.pcd.bin`
- `data/neilson_autoware_t4_intermediate/*/annotation/frame_annotations.json`

`frame_annotations.json` is an intermediate per-frame structure compatible with
the fields used by TIER IV's annotation generation code:

- `category_name`
- `instance_id`
- `attribute_names`
- `three_d_bbox.translation`
- `three_d_bbox.velocity`
- `three_d_bbox.acceleration`
- `three_d_bbox.size`
- `three_d_bbox.rotation`
- `num_lidar_pts`
- `num_radar_pts`

## Constraints

Neilson does not currently provide the full metadata stack Autoware usually
expects for a production T4 dataset:

- ego poses across the scene
- calibrated sensor chain
- scene/log tokens and timestamps in T4 schema
- rosbag2 / MCAP provenance
- camera topology compatible with T4 camera tables

So the intermediate export is the right first step, but a production Autoware
training dataset still needs either:

- a full T4 table generator that supplies those records, or
- access to the original capture metadata so TIER IV tooling can reconstruct
  them cleanly.

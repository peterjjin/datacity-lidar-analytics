# Neilson Pretrained Benchmark Summary

Validation split:
- frames: 892
- ground-truth boxes: 5,426
- benchmark scene: `AlbanyNeilson_DATA_20240201_213000`

Current pretrained baseline results:

| Algorithm | Status | TP | FP | FN | Type I | Type II | Report |
|---|---:|---:|---:|---:|---:|---:|---|
| `mmdet3d_pointpillars_car` | `ok` | 0 | 640 | 5426 | 1.0000 | 1.0000 | `experiments/neilson_car/benchmark_runs_mmdet_rerun/algorithm_error_report.md` |
| `openpcdet_pointpillar_car` | `ok` | 11 | 50679 | 5415 | 0.9998 | 0.9980 | `experiments/neilson_car/benchmark_runs_openpcdet_pointpillar_final/algorithm_error_report.md` |
| `openpcdet_second_car` | `ok` | 8 | 29203 | 5418 | 0.9997 | 0.9985 | `experiments/neilson_car/benchmark_runs_openpcdet_second/algorithm_error_report.md` |
| `open3dml_pointpillars_kitti` | `ok` | 0 | 267 | 5426 | 1.0000 | 1.0000 | `experiments/neilson_car/benchmark_runs_open3dml_pointpillars/algorithm_error_report.md` |

Notes:
- OpenPCDet now has native Neilson `kitti_infos_*.pkl` and `kitti_dbinfos_train.pkl` under `data/neilson_car_kitti/`.
- Open3D-ML runs through a compatibility overlay in `/tmp/open3dml_compat` with `torch 2.2.2` and `numpy 1.26.4`.
- All current pretrained baselines are domain-mismatched to Neilson and behave as high-false-positive / high-false-negative references, not usable final models.

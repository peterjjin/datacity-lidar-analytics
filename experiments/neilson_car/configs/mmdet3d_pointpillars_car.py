import os

_base_ = '/home/datacity/src/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'

data_root = os.environ.get(
    'NEILSON_CAR_KITTI_ROOT',
    '/mnt/d/Dropbox/_Rutgers/2026.03.14 Open3D-ML LiDAR Analytics/data/neilson_car_kitti/',
)
data_root = data_root.rstrip('/\\')
load_from = os.environ.get('MMDET3D_LOAD_FROM') or None
train_workers = int(os.environ.get('MMDET3D_TRAIN_WORKERS', '12'))
val_workers = int(os.environ.get('MMDET3D_VAL_WORKERS', '12'))
val_interval = int(os.environ.get('MMDET3D_VAL_INTERVAL', '1000'))

class_names = ['Car']
metainfo = dict(classes=class_names)
backend_args = None
input_modality = dict(use_lidar=True, use_camera=False)

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

# Neilson does not provide KITTI ground-plane files or a true image-FOV camera
# setup, so keep the first experiment LiDAR-only and avoid ObjectSample.
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=train_workers,
    persistent_workers=train_workers > 0,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='KittiDataset',
            data_root=data_root,
            ann_file='neilson_car_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=val_workers,
    persistent_workers=val_workers > 0,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file='neilson_car_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=val_workers,
    persistent_workers=val_workers > 0,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file='neilson_car_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    type='KittiMetric',
    ann_file=os.path.join(data_root, 'neilson_car_infos_val.pkl'),
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))

train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=val_interval)

_base_ = ['../_base_/default_runtime.py',
          '../_base_/tests/segmentation.py']
# misc custom setting
batch_size = 8  # bs: total bs in all gpus
batch_size_val = 8
batch_size_test = 6
mix_prob = 0.0
empty_cache = False
enable_amp = True
num_worker = 8
find_unused_parameters = False
sync_bn = True
param_dicts = [dict(keyword="block", lr_scale=0.1)]
eval = False
seed = 3407
# CUDA_VISIBLE_DEVICES=0 sh ./scripts/test.sh -d s3dis -c conda-former -n test -w 

# model settings
model = dict(
    type="ConDaFormer-small-v1m1",
    in_channels=6,
    out_channels=13,
    voxel_size=0.04,
    window_size=8,
    quant_size=0.25,
    base_channels=48,
    head_dim=16,
    drop_path_rate=0.5,
    channels=(96, 192, 384, 192, 96, 48),
    layers=(2, 6, 2, 0, 0, 0),
)

# scheduler settings
epoch = 3000
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(type='MultiStepLR', milestones=[0.6, 0.8], gamma=0.1)

# dataset settings
dataset_type = "S3DISDataset"
data_root = "data/s3dis/"
data = dict(
    num_classes=13,
    ignore_label=255,
    names=['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
           'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'],
    train=dict(
        type='S3DISDataset',
        split=('Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.8, 1.2]),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="Voxelize", voxel_size=0.04, hash_type='fnv', mode='train',
                 keys=("coord", "color", "label"), return_discrete_coord=True),
            dict(type="SphereCrop", point_max=80000, mode='random'),
            dict(type="PositiveShift"),
            dict(type="NormalizeColor01"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "label"), feat_keys=["coord", "color"])
        ],
        test_mode=False
    ),
    val=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root=data_root,
        transform=[
            dict(type="Voxelize", voxel_size=0.04, hash_type='fnv', mode='val',
                 keys=("coord", "color", "label"), return_discrete_coord=True, 
                 return_inverse=True
                 ),
            dict(type="PositiveShift"),
            dict(type="NormalizeColor01"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "discrete_coord", "label", "inverse", "length"
                       ),
                 offset_keys_dict=dict(offset="coord"),
                 feat_keys=["coord", "color"])
        ],
        test_mode=False),
    test=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root=data_root,
        transform=[
            dict(type='NormalizeColor01')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='Voxelize',
                voxel_size=0.04,
                hash_type='fnv',
                mode='test',
                keys=('coord', 'color'),
                return_discrete_coord=True),
            crop=None,
            post_transform=[
                dict(type='PositiveShift'),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'discrete_coord', 'index'),
                    feat_keys=('coord', 'color'))
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis='z', center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis='z', center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis='z', center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis='z', center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis='z', center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis='z', center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis='z', center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis='z', center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis='z', center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis='z', center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis='z', center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis='z', center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
            ]
        )
    )
)

criteria = [
    dict(type="CrossEntropyLoss",
         loss_weight=1.0,
         ignore_index=data["ignore_label"])
]

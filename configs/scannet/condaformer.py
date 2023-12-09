_base_ = ['../_base_/default_runtime.py',
          '../_base_/tests/segmentation.py']
# misc custom setting
batch_size = 12  # bs: total bs in all gpus
batch_size_val = 12
batch_size_test = 4
mix_prob = 0.0
empty_cache = True
enable_amp = True
num_worker = 16
find_unused_parameters = False
sync_bn = True
param_dicts = [dict(keyword="block", lr_scale=0.1)]
eval = False
seed = 3407
# CUDA_VISIBLE_DEVICES=0 sh ./scripts/test.sh -d s3dis -c conda-former -n test -w 

# model settings
model = dict(
    type="ConDaFormer-v1m1",
    in_channels=9,
    out_channels=20,
    voxel_size=0.02,
    window_size=10,
    quant_size=0.25,
    base_channels=48,
    drop_path_rate=0.3,
    channels=(96, 192, 384, 384, 384, 192, 96, 48),
    layers=(2, 2, 6, 2, 0, 0, 0, 0),
)

# scheduler settings
epoch = 900

optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.02)
scheduler = dict(type='MultiStepWithWarmupLR', milestones=[0.6, 0.8], gamma=0.1, warmup_rate=0.02, warmup_scale=1e-6)

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"
data = dict(
    num_classes=20,
    ignore_label=255,
    names=["wall", "floor", "cabinet", "bed", "chair",
           "sofa", "table", "door", "window", "bookshelf",
           "picture", "counter", "desk", "curtain", "refridgerator",
           "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis='x', p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis='y', p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="Voxelize", voxel_size=0.02, hash_type='fnv', mode='train',
                 keys=("coord", "normal", "color", "label"), return_discrete_coord=True),
            dict(type="SphereCrop", point_max=100000, mode='random'),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "label"), feat_keys=["coord", "normal", "color"])
        ],
        test_mode=False
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Voxelize", voxel_size=0.02, hash_type='fnv', mode='val',
                 keys=("coord", "normal", "color", "label"), return_discrete_coord=True, return_inverse=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "discrete_coord", "label", "inverse", "length"),
                 offset_keys_dict=dict(offset="coord"),
                 feat_keys=["coord", "normal", "color"])
        ],
        test_mode=False),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='Voxelize',
                voxel_size=0.02,
                hash_type='fnv',
                mode='test',
                keys=('coord', 'normal', 'color'),
                return_discrete_coord=True,
            ),
            test_range=[0,320],
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'discrete_coord', 'index'#, 'inverse', 'length'
                          ),
                    feat_keys=('coord', 'normal', 'color'))
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
                [dict(type="RandomFlip", p=1)]
            ]
        )
    )
)

criteria = [
    dict(type="CrossEntropyLoss",
         loss_weight=1.0,
         ignore_index=data["ignore_label"])
]

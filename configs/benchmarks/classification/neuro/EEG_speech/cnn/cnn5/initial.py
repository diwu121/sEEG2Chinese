_base_ = [
    '../../../../_base_/datasets/neuro/cls.py',
    '../../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='PlainCNN',
        depth=5,
        in_channels=117,
        patch_size=15,
        kernel_size=7,
        base_channels=128, out_channels=256,
        drop_rate=0.1,
        out_indices=(3,),  # no conv-1, x-1: stage-x
    ),
    head=dict(
        type='GNN_ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=256, num_classes=24)
)

# data
data_root = '/usr/data/DATA/EEG_SPEECH/sub1/initial/'
data = dict(
    train=dict(
        data_source=dict(
            type='ProcessedDataset', root=data_root, split='train', seed=42,
    )),
    val=dict(
        data_source=dict(
            type='ProcessedDataset', root=data_root, split='test', seed=42,
    )),
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.1, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

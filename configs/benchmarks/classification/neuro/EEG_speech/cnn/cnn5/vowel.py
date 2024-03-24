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
        patch_size=7,
        kernel_size=4,
        base_channels=128, out_channels=256,
        drop_rate=0.1,
        out_indices=(3,),  # no conv-1, x-1: stage-x
    ),
    head=dict(
        type='NAR_ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=256, num_classes=12)
)
train_pipeline_1 = [
    # dict(type='RandomScaling', sigma=0.001, p=0.1),
    dict(type='RandomPermutation', max_segments=5, p=0.2),
    # dict(type='RandomJitter', sigma=0.001, p=0.1,),
    dict(type='ToTensor'),
]
# train_pipeline_2 = [
#     dict(type='RandomScaling', sigma=1.8, p=1.0),
#     dict(type='RandomPermutation', max_segments=2, p=0.8),
#     dict(type='RandomJitter', sigma=2, p=1.0),
#     dict(type='ToTensor'),
# ]
test_pipeline = [
    dict(type='ToTensor'),
]
# data
data_root = '/usr/data/DATA/EEG_SPEECH/sub1/vowel/'
data = dict(
    train=dict(
        data_source=dict(
            type='ProcessedDataset', root=data_root, split='train', seed=23,),
        pipeline=train_pipeline_1,
    ),
    val=dict(
        data_source=dict(
            type='ProcessedDataset', root=data_root, split='test', seed=23,),
        pipeline=test_pipeline,
    ),
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=4e-3,
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)

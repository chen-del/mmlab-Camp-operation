
# convnext 显存不够
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(type='ConvNeXt', arch='base', drop_path_rate=0.5),
#     head=dict(
#         type='LinearClsHead',
#         num_classes=30,
#         in_channels=1024,
#         loss=dict(
#             type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
#     ),
#     init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in21k_20220124-13b83eec.pth'),
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0.8),
    #     dict(type='CutMix', alpha=1.0),
#     # ]),
# )

# resnet50
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=30,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
    init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth')
)




############################################ dataset settings###########################################
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=30,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=12,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='/home/hanhan/cc_wrod/develop/mmlab/mmpretrain/datasets/fruit30/train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='/home/hanhan/cc_wrod/develop/mmlab/mmpretrain/datasets/fruit30/val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

############################################ schedules ###########################################
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))




# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.25, by_epoch=False, begin=0, end=2500),
    dict(
        type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
]




# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=5)
val_cfg = dict()
test_cfg = dict()

############################################ runtime ###########################################
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=5, save_best='auto'),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
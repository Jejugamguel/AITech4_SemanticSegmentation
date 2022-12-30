_base_ = [
    '../../../mmsegmentation/configs/_base_/models/segmenter_vit-b16_mask.py',
    '../../_base_/custom.py', '../../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../_base_/scheduler_epochs_60.py'
]
import wandb
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'  # noqa

model = dict(
    pretrained=checkpoint,
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        num_classes=11),
    )

optimizer = dict(lr=0.001, weight_decay=0.0)

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    # num_gpus: 8 -> batch_size: 8
    samples_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))


wandb.login()
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='WandbLoggerHook',interval=100,
            init_kwargs=dict(
                project='Segmentation_project',
                entity = 'aitech4_cv3',
                name = "segmenter_vit_l_60e"),)
        # log_checkpoint=True,
        # log_checkpoint_metadata=True,
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])


# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

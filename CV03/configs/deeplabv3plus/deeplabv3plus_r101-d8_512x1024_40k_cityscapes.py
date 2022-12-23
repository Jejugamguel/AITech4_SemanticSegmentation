import wandb



_base_ = [
    '../../../mmsegmentation/configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/custom.py', '../../../mmsegmentation/configs/_base_/default_runtime.py',
    # '../../../mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
model = dict(
    decode_head=dict(num_classes=11), 
    auxiliary_head=dict(num_classes=11))



# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# # runtime settings
# runner = dict(type='IterBasedRunner', max_iters=40000)
# checkpoint_config = dict(by_epoch=False, interval=4000)
# evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=5, max_keep_ckpts=5)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)


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
                name = "DeepLabv3+_resnet101_60e"),)
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

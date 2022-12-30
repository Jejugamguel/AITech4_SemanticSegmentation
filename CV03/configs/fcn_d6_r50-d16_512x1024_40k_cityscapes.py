import wandb
_base_ = [
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/mmsegmentation/configs/_base_/models/fcn_r50-d8.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/_base_/custom.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/default_runtime.py', 
    # '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(dilation=6,num_classes=11),
    auxiliary_head=dict(dilation=6, num_classes=11))




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
                name = "test_JN"),)
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

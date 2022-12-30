import wandb
_base_ = [
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/mmsegmentation/configs/_base_/models/deeplabv3_r50-d8.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/configs/Augmentation/Aug_NCmxG2TTA3.py', 
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/mmsegmentation/configs/_base_/default_runtime.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/_base_/scheduler_epochs_60.py'
]
model = dict(
    decode_head=dict(num_classes=11), 
    auxiliary_head=dict(num_classes=11))

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
                name = "Hae"),)
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
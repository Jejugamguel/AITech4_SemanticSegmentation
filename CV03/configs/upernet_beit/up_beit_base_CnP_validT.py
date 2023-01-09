import wandb
import random
_base_ = [
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/mmsegmentation/configs/_base_/models/upernet_beit.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/_base_/CnP_custom_validT.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/default_runtime.py', 
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/_base_/scheduler_epochs_60.py'
]

## model 수정
model = dict(pretrained = '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/pretrain/beit_base_patch16_224_pt22k_ft22k.pth',
             decode_head = dict(num_classes = 11),
             auxiliary_head = dict(num_classes = 11))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
        betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

# learning policy
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


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
                name = "upernet_vit-b16_AdamW"),)
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

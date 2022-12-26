_base_ = [
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/mmsegmentation/configs/_base_/models/deeplabv3_r50-d8.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/configs/Augmentation/Aug_base.py', 
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/mmsegmentation/configs/_base_/default_runtime.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/_base_/scheduler_epochs_60.py'
]
model = dict(
    decode_head=dict(num_classes=11), 
    auxiliary_head=dict(num_classes=11))

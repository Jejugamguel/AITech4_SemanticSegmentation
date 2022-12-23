_base_ = [
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/mmsegmentation/configs/_base_/models/fcn_r50-d8.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/_base_/custom.py',
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/CV03/default_runtime.py', 
    '/opt/ml/level2_semanticsegmentation_cv-level2-cv-03/mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(dilation=6,num_classes=11),
    auxiliary_head=dict(dilation=6, num_classes=11))

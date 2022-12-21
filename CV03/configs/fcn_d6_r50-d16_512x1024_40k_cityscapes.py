_base_ = [
    '../../mmsegmentation/configs/_base_/models/fcn_r50-d8.py', '../_base_/custom.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py', '../../mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(dilation=6,num_classes=11),
    auxiliary_head=dict(dilation=6, num_classes=11))

_base_ = [
    '../../mmsegmentation/configs/_base_/models/deeplabv3_r50-d8.py',
    '../_base_/custom.py', 
    '../../mmsegmentation/configs/_base_/default_runtime.py',
    '../../mmsegmentation/configs/_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=11), 
    auxiliary_head=dict(num_classes=11))

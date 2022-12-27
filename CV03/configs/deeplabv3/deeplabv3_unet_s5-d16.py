_base_ = [
    '../../mmsegmentation/configs/_base_/models/deeplabv3_unet_s5-d16.py',
    '../_base_/custom.py', 
    '../../mmsegmentation/configs/_base_/default_runtime.py',
    '../_base_/scheduler_epochs_60.py'
]
model = dict(
    decode_head=dict(num_classes=11), 
    auxiliary_head=dict(num_classes=11))

import json

# dataset settings
dataset_type = "CustomDataset"
data_root = "/opt/ml/input/data"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)

cfg = json.load(open("/opt/ml/config.json", "r"))

classes = [
    "Background",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]

palette = [
    [0, 0, 0],
    [192, 0, 128],
    [0, 128, 192],
    [0, 128, 64],
    [128, 0, 0],
    [64, 0, 128],
    [64, 0, 192],
    [192, 128, 64],
    [192, 192, 128],
    [64, 64, 128],
    [128, 0, 192],
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomFlip", prob=0),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomFlip", prob=0),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img"]),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir= cfg['train_img_path'],
        ann_dir= cfg['train_mask_path'],
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir=cfg['valid_img_path'],
        ann_dir=cfg['valid_mask_path'],
        pipeline=val_pipeline,
        classes=classes,
        palette=palette,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # reduce_zero_label=True,
        img_dir=cfg['test_path'],
        # ann_dir="",
        pipeline=val_pipeline,
        classes=classes,
        palette=palette,
    ),
)

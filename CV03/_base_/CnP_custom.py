import json

# dataset settings
dataset_type = "CustomDataset"
data_root = "/opt/ml/input/data"
img_norm_cfg = dict(
    mean=[117.323, 112.092, 106.659],
    std=[59.772, 58.859, 62.124],
    to_rgb=True
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
    dict(type="CopyPaste", prob=1, mode='all', patch_scale_ratio=0.75),
    dict(type="PhotoMetricDistortion"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="RandomRotate", prob=0.8, degree=30),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    
    dict(type="Resize", img_scale=(640, 640)),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        classes=classes,
        palette=palette,
        type="CustomDataset",
        # reduce_zero_label=False
        data_root=data_root,
        img_dir=cfg['train_img_path'],
        ann_dir=cfg['train_mask_path'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
        ]
    ),
    pipeline=train_pipeline)


val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(640,640),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=train_dataset,
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
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
    ),
)

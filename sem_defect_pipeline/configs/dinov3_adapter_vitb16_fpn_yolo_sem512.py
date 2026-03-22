"""
MMSegmentation config — DINOv3 ViT-Adapter + FPN with YOLO segmentation labels.

Combines:
  - ViT-Adapter backbone (from dinov3_adapter_vitb16_fpn_sem512.py)
  - YOLO .txt label loading (from dinov3_vitb16_fpn_yolo_sem512.py)
"""

_base_ = ['./dinov3_adapter_vitb16_fpn_sem512.py']

custom_imports = dict(
    imports=[
        'sem_defect_pipeline.dinov3_backbone',
        'sem_defect_pipeline.transforms',
    ],
    allow_failed_imports=False,
)

# ─── Override pipelines to use YOLO labels ───────────────────────────────────
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='LoadYOLOAnnotations', binary=True),
    dict(type='RandomResize',
         scale=crop_size,
         ratio_range=(0.75, 1.25),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion',
         brightness_delta=20,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.9, 1.1),
         hue_delta=5),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadYOLOAnnotations', binary=True),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='labels/train',
        ),
        seg_map_suffix='.txt',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='labels/val',
        ),
        seg_map_suffix='.txt',
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

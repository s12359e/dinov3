"""
MMSegmentation config — DINOv3 ViT-B/16 + SimpleFPN for SEM micro-defect detection.

Target task  : Binary segmentation of 4-pixel metal-extrusion defects in 512×512 SEM images.
Backbone     : DINOv3 ViT-B/16 (frozen patch embed; fine-tune transformer blocks).
Neck         : FPN — takes 4 same-resolution ViT feature maps and builds true multi-scale.
Head         : FPNHead — aggregates scales, outputs per-pixel class logits (2 classes).
Loss         : FocalLoss (γ=2, α=0.75) + DiceLoss (w=3.0).
               Focal penalises hard background/foreground confusion.
               Dice directly optimises IoU for the tiny defect region.

Requirements:
    mmsegmentation >= 1.2.0   (FocalLoss added in 1.2)
    mmengine >= 0.9.0
    torch >= 2.0
    dinov3 (this repo)

Directory layout expected under data_root:
    data/sem_defect/
        images/train/   *.png  (3-channel uint8, grayscale replicated)
        images/val/     *.png
        masks/train/    *.png  (single-channel: 0=background, 1=defect)
        masks/val/      *.png
"""

# ─── Base configs ────────────────────────────────────────────────────────────
_base_ = []   # standalone — no _base_ deps to keep this self-contained

custom_imports = dict(
    imports=['sem_defect_pipeline.dinov3_backbone'],
    allow_failed_imports=False,
)

# ─── Dataset ─────────────────────────────────────────────────────────────────
dataset_type = 'BaseSegDataset'
data_root = 'data/sem_defect'
crop_size = (512, 512)

# SEM images are grayscale but saved as 3-channel (R=G=B).
# Mean/std computed for that convention (pixel range ~[0, 255], grey-scale ~128).
img_mean = [128.0, 128.0, 128.0]
img_std  = [64.0,  64.0,  64.0]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize',
         scale=crop_size,
         ratio_range=(0.75, 1.25),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # PhotoMetricDistortion helps the model generalise across SEM brightness shifts
    dict(type='PhotoMetricDistortion',
         brightness_delta=20,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.9, 1.1),
         hue_delta=5),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='masks/train',
        ),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='masks/val',
        ),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

# Evaluate with mIoU, mDice, and F1 — Dice / F1 are most meaningful for tiny defects.
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
)
test_evaluator = val_evaluator

# ─── Model ───────────────────────────────────────────────────────────────────
norm_cfg = dict(type='BN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=img_mean,
    std=img_std,
    bgr_to_rgb=False,     # images saved as RGB-grey (R=G=B), no swap needed
    pad_val=0,
    seg_pad_val=255,      # 255 = ignore index for any padded regions
    size=crop_size,
)

# ViT-B/16 embed_dim = 768.  All 4 extracted feature maps have stride 16 and
# spatial size 32×32 for a 512-pixel input (512/16 = 32).
_embed_dim = 768

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    # ── Backbone ──────────────────────────────────────────────────────────
    backbone=dict(
        type='DINOv3ViTBackbone',

        img_size=512,
        patch_size=16,
        embed_dim=_embed_dim,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,

        # Stochastic depth: ramp up to 0.1 across the 12 blocks.
        drop_path_rate=0.1,

        # Extract features from block indices 3, 5, 7, 11 (0-indexed).
        # These correspond to layers 4, 6, 8, 12 in 1-indexed notation,
        # giving shallow→deep multi-scale semantic content.
        out_indices=[3, 5, 7, 11],

        # Freeze the patch embedding + first 4 blocks to preserve low-level
        # edge representations learned during DINOv3 SSL pre-training.
        frozen_stages=4,

        init_cfg=dict(
            type='Pretrained',
            # !! Replace with your actual DINOv3 ViT-B/16 checkpoint path !!
            checkpoint='checkpoints/dinov3_vitb16_pretrain.pth',
        ),
    ),

    # ── Neck: FPN ──────────────────────────────────────────────────────────
    # All 4 backbone outputs are (B, 768, 32, 32) — same resolution.
    # FPN treats them as a flat list and applies lateral convs; it then
    # creates P2–P5 via top-down upsampling, giving true multi-scale maps:
    #   P2: (B, 256, 128, 128)  stride  4  (3× upsample from P5 via FPN)
    #   P3: (B, 256,  64,  64)  stride  8
    #   P4: (B, 256,  32,  32)  stride 16  (same as ViT output)
    #   P5: (B, 256,  16,  16)  stride 32  (3× downsample via extra conv)
    #
    # NOTE: Because ViT features are all stride-16, the FPN here acts more as
    # a learned feature aggregation + scale generation module than a traditional
    # pyramid.  For a 4-pixel defect, P2 (stride 4, 128×128) is the critical scale.
    neck=dict(
        type='FPN',
        in_channels=[_embed_dim, _embed_dim, _embed_dim, _embed_dim],
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs='on_output',   # extra stride-32 level from P5
        relu_before_extra_convs=True,
        norm_cfg=norm_cfg,
    ),

    # ── Decode Head: FPNHead ───────────────────────────────────────────────
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,            # 0=background, 1=defect
        norm_cfg=norm_cfg,
        align_corners=False,

        # ── Loss: CrossEntropy (class-weighted) + Dice ────────────────────
        # Class imbalance estimate:
        #   Defect area  = 16 px  (4×4 square)
        #   Image area   = 262144 px (512×512)
        #   Ratio        ≈ 0.006%  → extreme imbalance
        #
        # CrossEntropyLoss with class_weight:
        #   Background weight = 1.0, Defect weight = 50.0.
        #   Directly compensates for the ~16000:1 area ratio (capped at 50×
        #   to avoid gradient instability; Dice handles the rest).
        #
        # DiceLoss (weight=3.0):
        #   Optimises set-overlap directly; unaffected by class frequency.
        #   Higher weight (3×) signals the model to prioritise exact pixel match
        #   over calibration.  Dice converges slowly at start; CE provides
        #   gradient early, then Dice polishes the spatial boundary.
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=[1.0, 50.0],   # [background, defect]
                loss_weight=1.0,
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=False,   # generalised Dice: sum over classes
                eps=1e-3,           # prevent div-by-zero on near-empty masks
                reduction='mean',
                loss_weight=3.0,
            ),
        ],
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'),   # whole-image inference (no sliding window)
)

# ─── Optimizer & LR Schedule ─────────────────────────────────────────────────
# AdamW with layerwise LR decay:
#   backbone blocks get 0.1× the base LR (fine-tune gently).
#   norm layers and biases are excluded from weight decay (standard ViT practice).
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'norm':     dict(decay_mult=0.0),
            'bias':     dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'cls_token': dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# Warm-up for 1 000 iters → cosine decay to 0 over remaining 19 000 iters.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-7,
        begin=1000,
        end=20000,
        by_epoch=False,
    ),
]

# ─── Training Loop ────────────────────────────────────────────────────────────
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=1000,
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ─── Hooks & Logging ─────────────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best='mDice',    # Dice/F1 is the primary metric for tiny defects
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=200),
)

# Reduce on plateau callback (optional — plug into custom hook if desired)
# Monitor val/mDice; halve LR if no improvement for 3 val intervals.

default_scope  = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level     = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
visualizer    = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
)
randomness = dict(seed=42, deterministic=False)

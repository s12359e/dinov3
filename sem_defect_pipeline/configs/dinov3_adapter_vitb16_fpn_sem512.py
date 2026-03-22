"""
MMSegmentation config — DINOv3 ViT-B/16 + Adapter + FPN for SEM micro-defect detection.

Target task  : Binary segmentation of 4×4 metal-extrusion defects in 512×512 SEM images.
Backbone     : DINOv3 ViT-B/16 + ViT-Adapter (frozen ViT; train adapter layers).
               Adapter produces true multi-scale features at strides 4, 8, 16, 32
               via Spatial Prior Module (SPM) + deformable-attention interaction blocks.
Neck         : FPN — refines the already multi-scale adapter outputs.
Head         : FPNHead — aggregates scales, outputs per-pixel class logits (2 classes).
Loss         : CrossEntropyLoss (class-weighted) + DiceLoss.

Compared to the plain ViT backbone (dinov3_vitb16_fpn_sem512.py):
  - True multi-scale from the backbone (not just FPN-generated).
  - ViT is fully frozen → only adapter + neck + head are trained → faster convergence.
  - Better spatial detail at stride 4 → critical for 4-pixel defects.
  - Higher parameter count due to adapter layers (~+15M params).
  - Requires MSDeformAttn (multi-scale deformable attention).

Requirements:
    mmsegmentation >= 1.2.0
    mmengine >= 0.9.0
    torch >= 2.0
    dinov3 (this repo)
"""

# ─── Base: inherit dataset, loss, schedule from plain config ─────────────────
_base_ = ['./dinov3_vitb16_fpn_sem512.py']

# ─── Override backbone to use ViT-Adapter ────────────────────────────────────
_embed_dim = 768

model = dict(
    backbone=dict(
        _delete_=True,    # fully replace base backbone (plain ViT keys don't apply)
        type='DINOv3AdapterBackbone',

        img_size=512,
        patch_size=16,
        embed_dim=_embed_dim,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,

        # Adapter-specific parameters
        # interaction_indexes: ViT block indices where adapter injects
        # deformable-attention interaction layers.  For ViT-B/12, evenly
        # spaced across the 12 blocks.
        interaction_indexes=[2, 5, 8, 11],

        # SPM initial conv channels
        conv_inplane=64,

        # Deformable attention config
        n_points=4,
        deform_num_heads=16,
        adapter_drop_path_rate=0.3,

        # ConvFFN in interaction blocks
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,

        # Add ViT features to adapter multi-scale outputs
        add_vit_feature=True,
        use_extra_extractor=True,

        # Gradient checkpointing (saves VRAM)
        with_cp=True,

        init_cfg=dict(
            type='Pretrained',
            # !! Replace with your actual DINOv3 ViT-B/16 checkpoint path !!
            checkpoint='checkpoints/dinov3_vitb16_pretrain.pth',
        ),
    ),

    # ── Neck: FPN ──────────────────────────────────────────────────────────
    # Adapter outputs are already true multi-scale at strides 4, 8, 16, 32.
    # FPN refines them and produces 4 output scales at 256 channels.
    neck=dict(
        type='FPN',
        in_channels=[_embed_dim, _embed_dim, _embed_dim, _embed_dim],
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
)

# ─── Optimizer: higher LR for adapter, ViT is frozen anyway ─────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,           # slightly higher LR — adapter trains from scratch
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.adapter.backbone': dict(lr_mult=0.0),   # ViT is frozen
            'norm':      dict(decay_mult=0.0),
            'bias':      dict(decay_mult=0.0),
            'level_embed': dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

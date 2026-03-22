"""
DINOv3 backbone wrappers for MMSegmentation.

Two backbone variants:

1. DINOv3ViTBackbone (plain ViT)
   - Extracts intermediate ViT features at specified block indices.
   - All feature maps share the same stride (patch_size).
   - Relies on FPN neck to create multi-scale outputs.

2. DINOv3AdapterBackbone (ViT + Adapter)
   - Wraps ViT with the DINOv3_Adapter module (deformable attention + SPM).
   - Produces true multi-scale feature maps at strides 4, 8, 16, 32.
   - ViT backbone is frozen; only adapter parameters are trained.

Usage: set init_cfg.checkpoint to a DINOv3 pretrained .pth file.
"""

import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from mmengine.logging import print_log

from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter


@MODELS.register_module()
class DINOv3ViTBackbone(BaseModule):
    """
    MMSeg-compatible wrapper around DINOv3's DinoVisionTransformer.

    Args:
        img_size (int): Input image size. Default: 512.
        patch_size (int): Patch size. Default: 16.
        embed_dim (int): ViT embedding dimension. Default: 768 (ViT-B).
        depth (int): Number of transformer blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        ffn_ratio (float): MLP hidden-dim ratio. Default: 4.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        out_indices (list[int]): Zero-indexed block indices to extract.
            Default: [3, 5, 7, 11] — quarters of the 12-block ViT-B.
        frozen_stages (int): Freeze the first N transformer blocks + patch
            embed to prevent catastrophic forgetting. -1 = freeze nothing.
            Recommended: 0 (freeze patch embed only) or 4.
        init_cfg (dict): Pretrained weight config.
            Example::
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='path/to/dinov3_vitb16.pth'
                )
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        out_indices: list = [3, 5, 7, 11],
        frozen_stages: int = -1,
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.vit = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            drop_path_rate=drop_path_rate,
        )

    def init_weights(self):
        super().init_weights()  # loads checkpoint via init_cfg
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze patch embed and optionally early transformer blocks."""
        # Always freeze patch embedding if frozen_stages >= 0
        if self.frozen_stages >= 0:
            self.vit.patch_embed.eval()
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False
            if hasattr(self.vit, 'cls_token'):
                self.vit.cls_token.requires_grad = False
            if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
                self.vit.pos_embed.requires_grad = False

        for i in range(min(self.frozen_stages, len(self.vit.blocks))):
            block = self.vit.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W)  — SEM images replicated to 3 channels.
        Returns:
            tuple of tensors, one per out_index, each (B, embed_dim, H/16, W/16).
        """
        features = self.vit.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,   # → (B, C, h, w)
            norm=True,
        )
        # features is a tuple of len(out_indices) tensors
        return tuple(features)


@MODELS.register_module()
class DINOv3AdapterBackbone(BaseModule):
    """
    MMSeg-compatible wrapper around DINOv3_Adapter.

    The adapter adds a Spatial Prior Module (SPM) and deformable-attention
    interaction blocks around a frozen DINOv3 ViT backbone.  It produces
    true multi-scale feature maps:

        out[0]: stride  4  — (B, embed_dim, H/4,  W/4)
        out[1]: stride  8  — (B, embed_dim, H/8,  W/8)
        out[2]: stride 16  — (B, embed_dim, H/16, W/16)
        out[3]: stride 32  — (B, embed_dim, H/32, W/32)

    The ViT backbone is always frozen inside the adapter; only the adapter
    parameters (SPM, interaction blocks, norms) are trained.

    Args:
        img_size (int): Input image size. Default: 512.
        patch_size (int): Patch size. Default: 16.
        embed_dim (int): ViT embedding dimension. Default: 768.
        depth (int): Number of transformer blocks. Default: 12.
        num_heads (int): Number of ViT attention heads. Default: 12.
        ffn_ratio (float): MLP hidden-dim ratio. Default: 4.0.
        interaction_indexes (list[int]): ViT block indices where the adapter
            injects interaction layers. Default: [2, 5, 8, 11] for ViT-B/12.
        conv_inplane (int): SPM initial conv channels. Default: 64.
        n_points (int): Deformable attention sampling points. Default: 4.
        deform_num_heads (int): Deformable attention heads. Default: 16.
        adapter_drop_path_rate (float): Drop path in adapter blocks. Default: 0.3.
        with_cffn (bool): Use ConvFFN in interaction blocks. Default: True.
        cffn_ratio (float): ConvFFN hidden ratio. Default: 0.25.
        deform_ratio (float): Deformable attention ratio. Default: 0.5.
        add_vit_feature (bool): Add ViT features to adapter outputs. Default: True.
        use_extra_extractor (bool): Extra extractor at last interaction. Default: True.
        with_cp (bool): Use gradient checkpointing. Default: True.
        init_cfg (dict): Pretrained weight config for the ViT backbone.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        interaction_indexes: list = [2, 5, 8, 11],
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: int = 16,
        adapter_drop_path_rate: float = 0.3,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 0.5,
        add_vit_feature: bool = True,
        use_extra_extractor: bool = True,
        with_cp: bool = True,
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.embed_dim = embed_dim

        vit = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
        )

        self.adapter = DINOv3_Adapter(
            backbone=vit,
            interaction_indexes=interaction_indexes,
            pretrain_size=img_size,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            drop_path_rate=adapter_drop_path_rate,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            add_vit_feature=add_vit_feature,
            use_extra_extractor=use_extra_extractor,
            with_cp=with_cp,
        )

    def init_weights(self):
        super().init_weights()  # loads ViT checkpoint via init_cfg

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            tuple of 4 tensors at strides 4, 8, 16, 32.
        """
        out_dict = self.adapter(x)
        # DINOv3_Adapter returns {"1": f1, "2": f2, "3": f3, "4": f4}
        return (out_dict["1"], out_dict["2"], out_dict["3"], out_dict["4"])

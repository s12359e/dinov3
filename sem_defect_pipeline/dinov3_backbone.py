"""
DINOv3 ViT-B/16 backbone wrapper for MMSegmentation.

Wraps DinoVisionTransformer so MMSeg can call it like any other backbone.
Returns a tuple of 4 feature maps (one per extracted layer) with shape
(B, embed_dim, H/patch_size, W/patch_size) — all at the same stride.
The FPN neck downstream is responsible for creating true multi-scale outputs.

Usage: set init_cfg.checkpoint to a DINOv3 pretrained .pth file.
"""

import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from mmengine.logging import print_log

from dinov3.models.vision_transformer import DinoVisionTransformer


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

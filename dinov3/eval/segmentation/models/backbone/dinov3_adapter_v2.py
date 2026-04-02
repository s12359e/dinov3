# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Refactored ViT-Adapter with configurable MSDA modes, Dynamic U-Net Decoder,
# and Periodic Frequency Gating for semiconductor micro-anomaly detection.
#
# PART 1: DINOv3_Adapter  -- msda_mode in {'original', '4-level', '5-level'}
# PART 2: DynamicUNetDecoder -- pixel-shuffle & gated-attention options
# PART 3: PeriodicFrequencyGating -- spectral notch filter for AOI

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn


# ====================================================================
#  Shared Utilities
# ====================================================================

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_reference_points(spatial_shapes, device):
    """Generate normalized reference points for each spatial level."""
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, patch_size, msda_mode="original"):
    """Build deformable-attention reference points and spatial shapes.

    Returns (deform_inputs1, deform_inputs2):
      * deform_inputs1 -- for an *injector* direction (multi-scale source -> ViT query)
      * deform_inputs2 -- for the *extractor* direction (ViT source -> multi-scale query)

    The number of multi-scale levels depends on ``msda_mode``.
    """
    bs, c, h, w = x.shape

    if msda_mode == "original":
        msda_shapes = [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
    elif msda_mode == "4-level":
        msda_shapes = [
            (h // 4, w // 4), (h // 8, w // 8),
            (h // 16, w // 16), (h // 32, w // 32),
        ]
    elif msda_mode == "5-level":
        msda_shapes = [
            (h // 2, w // 2), (h // 4, w // 4), (h // 8, w // 8),
            (h // 16, w // 16), (h // 32, w // 32),
        ]
    else:
        raise ValueError(f"Unknown msda_mode: {msda_mode}")

    # -- deform_inputs1 (injector direction, unused in InteractionBlockWithCls
    #    but kept for forward-compatibility) -----------------------------------
    spatial_shapes1 = torch.as_tensor(msda_shapes, dtype=torch.long, device=x.device)
    level_start_index1 = torch.cat(
        (spatial_shapes1.new_zeros((1,)), spatial_shapes1.prod(1).cumsum(0)[:-1])
    )
    reference_points1 = get_reference_points(
        [(h // patch_size, w // patch_size)], x.device
    )
    deform_inputs1 = [reference_points1, spatial_shapes1, level_start_index1]

    # -- deform_inputs2 (extractor direction) ----------------------------------
    spatial_shapes2 = torch.as_tensor(
        [(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device
    )
    level_start_index2 = torch.cat(
        (spatial_shapes2.new_zeros((1,)), spatial_shapes2.prod(1).cumsum(0)[:-1])
    )
    reference_points2 = get_reference_points(msda_shapes, x.device)
    deform_inputs2 = [reference_points2, spatial_shapes2, level_start_index2]

    return deform_inputs1, deform_inputs2


# ====================================================================
#  PART 1 -- Configurable ViT-Adapter
# ====================================================================

class DWConv(nn.Module):
    """Depth-wise convolution over *variable* multi-scale token sequences.

    Unlike the original hard-coded 3-level split, this version accepts an
    explicit ``level_hw_list`` so it works with 3 / 4 / 5 levels.
    """

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, level_hw_list):
        """
        Args:
            x: (B, N, C) concatenated tokens from all MSDA levels.
            level_hw_list: list of (H_i, W_i) for each level (finest first).
        """
        B, N, C = x.shape
        split_sizes = [H * W for H, W in level_hw_list]
        xs = x.split(split_sizes, dim=1)
        outs = []
        for xi, (H, W) in zip(xs, level_hw_list):
            xi = xi.transpose(1, 2).view(B, C, H, W).contiguous()
            xi = self.dwconv(xi).flatten(2).transpose(1, 2)
            outs.append(xi)
        return torch.cat(outs, dim=1)


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, level_hw_list):
        x = self.fc1(x)
        x = self.dwconv(x, level_hw_list)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Extractor(nn.Module):
    def __init__(
        self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
        with_cffn=True, cffn_ratio=0.25, drop=0.0, drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads,
            n_points=n_points, ratio=deform_ratio,
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=dim,
                hidden_features=int(dim * cffn_ratio),
                drop=drop,
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index, level_hw_list):
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query), reference_points,
                self.feat_norm(feat), spatial_shapes,
                level_start_index, None,
            )
            query = query + attn
            if self.with_cffn:
                query = query + self.drop_path(
                    self.ffn(self.ffn_norm(query), level_hw_list)
                )
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        return query


class InteractionBlockWithCls(nn.Module):
    def __init__(
        self, dim, num_heads=6, n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0, drop_path=0.0, with_cffn=True, cffn_ratio=0.25,
        init_values=0.0, deform_ratio=1.0, extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()
        self.extractor = Extractor(
            dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
            norm_layer=norm_layer, deform_ratio=deform_ratio,
            with_cffn=with_cffn, cffn_ratio=cffn_ratio,
            drop=drop, drop_path=drop_path, with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(
                    dim=dim, num_heads=num_heads, n_points=n_points,
                    norm_layer=norm_layer, with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                    drop=drop, drop_path=drop_path, with_cp=with_cp,
                )
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, deform_inputs1, deform_inputs2,
                level_hw_list, H_toks, W_toks):
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            level_hw_list=level_hw_list,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    level_hw_list=level_hw_list,
                )
        return x, c, cls


class SpatialPriorModule(nn.Module):
    """Multi-scale CNN stem producing spatial features at various strides.

    Always returns a **list of 4-D spatial tensors** (B, embed_dim, H_i, W_i)
    ordered from finest to coarsest.  Tokenisation is left to the adapter.

    For ``msda_mode='5-level'`` an extra 1/2-resolution feature is intercepted
    *before* the stride-2 MaxPool in the stem.
    """

    def __init__(self, inplanes=64, embed_dim=384, msda_mode="original",
                 with_cp=False):
        super().__init__()
        self.msda_mode = msda_mode
        self.with_cp = with_cp

        # Split stem so we can intercept the 1/2-resolution feature
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, inplanes, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False),
            nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False),
            nn.SyncBatchNorm(inplanes), nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes), nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes), nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes), nn.ReLU(inplace=True),
        )

        # 1x1 projections to embed_dim
        if msda_mode == "5-level":
            self.fc0 = nn.Conv2d(inplanes, embed_dim, 1, bias=True)       # 1/2
        self.fc1 = nn.Conv2d(inplanes, embed_dim, 1, bias=True)           # 1/4
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, 1, bias=True)       # 1/8
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, 1, bias=True)       # 1/16
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, 1, bias=True)       # 1/32

    def forward(self, x):
        def _inner_forward(x):
            c_half = self.stem_conv(x)       # stride 2  (1/2)
            c1_raw = self.stem_pool(c_half)  # stride 4  (1/4)
            c2_raw = self.conv2(c1_raw)      # stride 8  (1/8)
            c3_raw = self.conv3(c2_raw)      # stride 16 (1/16)
            c4_raw = self.conv4(c3_raw)      # stride 32 (1/32)

            feats = []
            if self.msda_mode == "5-level":
                feats.append(self.fc0(c_half))
            feats.extend([
                self.fc1(c1_raw),
                self.fc2(c2_raw),
                self.fc3(c3_raw),
                self.fc4(c4_raw),
            ])
            return feats

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)


class DINOv3_Adapter(nn.Module):
    """Configurable ViT-Adapter with three MSDA modes.

    ``msda_mode`` controls which SPM levels participate in deformable attention:

    * ``'original'`` -- 3 levels (1/8, 1/16, 1/32) in MSDA; 1/4 bypasses.
      Output: 4 feature maps at strides {4, 8, 16, 32}.
    * ``'4-level'`` -- 4 levels (1/4, 1/8, 1/16, 1/32) all in MSDA.
      Output: 4 feature maps at strides {4, 8, 16, 32}.
    * ``'5-level'`` -- 5 levels (1/2, 1/4, 1/8, 1/16, 1/32) all in MSDA.
      Output: 5 feature maps at strides {2, 4, 8, 16, 32}.

    NOTE: 5-level mode produces significantly more tokens and requires
    proportionally more GPU memory.
    """

    def __init__(
        self,
        backbone,
        msda_mode="original",
        interaction_indexes=(9, 19, 29, 39),
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=True,
    ):
        super().__init__()
        assert msda_mode in ("original", "4-level", "5-level")
        self.msda_mode = msda_mode
        self.backbone = backbone
        self.backbone.requires_grad_(False)

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = list(interaction_indexes)
        self.add_vit_feature = add_vit_feature
        embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        n_msda_levels = {"original": 3, "4-level": 4, "5-level": 5}[msda_mode]
        n_output_levels = {"original": 4, "4-level": 4, "5-level": 5}[msda_mode]

        self.level_embed = nn.Parameter(torch.zeros(n_msda_levels, embed_dim))
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dim=embed_dim,
            msda_mode=msda_mode, with_cp=False,
        )

        block_fn = InteractionBlockWithCls
        self.interactions = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                init_values=init_values, drop_path=drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                with_cffn=with_cffn, cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=(
                    (i == len(self.interaction_indexes) - 1) and use_extra_extractor
                ),
                with_cp=with_cp,
            )
            for i in range(len(self.interaction_indexes))
        ])

        # 'original' mode creates 1/4 output via ConvTranspose upsample of 1/8
        if msda_mode == "original":
            self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)

        self.norms = nn.ModuleList(
            [nn.SyncBatchNorm(embed_dim) for _ in range(n_output_levels)]
        )

        # -- weight init -------------------------------------------------------
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        if msda_mode == "original":
            self.up.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        nn.init.normal_(self.level_embed)

    # -- init helpers ----------------------------------------------------------

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _init_deform_weights(m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    # -- forward ---------------------------------------------------------------

    def forward(self, x):
        bs, C_in, h, w = x.shape
        embed_dim = self.backbone.embed_dim
        H_toks = h // self.patch_size
        W_toks = w // self.patch_size

        # 1) Spatial Prior Module  (list of 4-D spatial tensors, finest first)
        spm_feats = self.spm(x)

        # 2) Separate bypass (original only) from MSDA-participating levels
        if self.msda_mode == "original":
            bypass_feat = spm_feats[0]          # (B, dim, H/4, W/4)
            msda_feats_4d = spm_feats[1:]       # [1/8, 1/16, 1/32]
        else:
            bypass_feat = None
            msda_feats_4d = spm_feats           # all levels

        # 3) Tokenise & add level embeddings
        level_hw_list = [(f.shape[2], f.shape[3]) for f in msda_feats_4d]
        tokens = []
        for i, f in enumerate(msda_feats_4d):
            t = f.reshape(bs, embed_dim, -1).transpose(1, 2)   # (B, N_i, dim)
            t = t + self.level_embed[i]
            tokens.append(t)
        c = torch.cat(tokens, dim=1)                            # (B, N_total, dim)

        # 4) Deformable-attention inputs
        deform_in1, deform_in2 = deform_inputs(x, self.patch_size, self.msda_mode)

        # 5) Frozen ViT intermediate features
        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                all_layers = self.backbone.get_intermediate_layers(
                    x, n=self.interaction_indexes, return_class_token=True,
                )

        # 6) Interaction blocks
        outs = []
        for i, block in enumerate(self.interactions):
            vit_x, cls = all_layers[i]
            _, c, _ = block(
                vit_x, c, cls,
                deform_in1, deform_in2,
                level_hw_list, H_toks, W_toks,
            )
            outs.append(
                vit_x.transpose(1, 2)
                .reshape(bs, embed_dim, H_toks, W_toks)
                .contiguous()
            )

        # 7) Split concatenated tokens back to per-level spatial maps
        split_sizes = [H_i * W_i for H_i, W_i in level_hw_list]
        c_splits = c.split(split_sizes, dim=1)
        msda_out = [
            c_i.transpose(1, 2).reshape(bs, embed_dim, H_i, W_i).contiguous()
            for c_i, (H_i, W_i) in zip(c_splits, level_hw_list)
        ]

        # 8) Build final output list (finest -> coarsest)
        if self.msda_mode == "original":
            # msda_out = [1/8, 1/16, 1/32]
            c1_out = self.up(msda_out[0]) + bypass_feat
            final_feats = [c1_out] + msda_out       # [1/4, 1/8, 1/16, 1/32]
        else:
            final_feats = msda_out

        # 9) Add interpolated ViT features
        if self.add_vit_feature:
            n_vit = len(outs)
            n_out = len(final_feats)
            offset = n_out - n_vit          # 0 for original/4-level, 1 for 5-level
            for i in range(n_vit):
                j = i + offset
                target_size = (final_feats[j].shape[2], final_feats[j].shape[3])
                final_feats[j] = final_feats[j] + F.interpolate(
                    outs[i], size=target_size, mode="bilinear", align_corners=False,
                )

        # 10) Normalize & return as dict keyed by level index
        return {
            str(i): self.norms[i](feat) for i, feat in enumerate(final_feats)
        }


# ====================================================================
#  PART 2 -- Dynamic U-Net Decoder
# ====================================================================

class SemanticSpatialGate(nn.Module):
    """Sigmoid spatial gate generated from the upsampled low-res feature.

    Multiplies the high-res skip feature element-wise, suppressing
    spatially irrelevant skip content before concatenation.
    """

    def __init__(self, gate_channels):
        super().__init__()
        mid = max(gate_channels // 4, 1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),
        )

    def forward(self, low_res_up, skip_feat):
        """Returns ``skip_feat * sigmoid(gate)``."""
        gate = torch.sigmoid(self.gate_conv(low_res_up))   # (B, 1, H, W)
        return skip_feat * gate


class DecoderBlock(nn.Module):
    """Single decoder stage: upsample -> (optional gate) -> concat skip -> conv."""

    def __init__(self, in_ch, skip_ch, out_ch,
                 use_pixel_shuffle=False, use_gated_attention=False):
        super().__init__()
        self.use_pixel_shuffle = use_pixel_shuffle
        self.use_gated_attention = use_gated_attention

        if use_pixel_shuffle:
            # Sub-pixel convolution: expand channels then rearrange spatially
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 1, bias=False),
                nn.PixelShuffle(upscale_factor=2),
            )
            up_out_ch = out_ch
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                  align_corners=False)
            up_out_ch = in_ch

        if use_gated_attention:
            self.gate = SemanticSpatialGate(up_out_ch)

        self.conv = nn.Sequential(
            nn.Conv2d(up_out_ch + skip_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Ensure spatial alignment after upsample
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        if self.use_gated_attention:
            skip = self.gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DynamicUNetDecoder(nn.Module):
    """U-Net decoder that reconstructs adapter features to full resolution.

    Builds its internal stages dynamically from ``adapter_channels``.

    Args:
        adapter_channels: channel dimensions for each adapter level, ordered
            **finest to coarsest** (e.g. ``[384]*4`` for 4-level).
        num_classes: number of segmentation classes.
        use_pixel_shuffle: replace bilinear upsample with sub-pixel conv.
        use_gated_attention: add Semantic Spatial Gate at skip connections.
        use_novel_enhancement: enable Periodic Frequency Gating (Part 3).
        periodic_pitch: physical periodic pitch in *original image pixels*.
        finest_stride: stride of the finest adapter feature (4 or 2).
    """

    def __init__(
        self,
        adapter_channels,
        num_classes,
        use_pixel_shuffle=False,
        use_gated_attention=False,
        use_novel_enhancement=False,
        periodic_pitch=22,
        finest_stride=4,
    ):
        super().__init__()
        n_levels = len(adapter_channels)
        self.use_novel_enhancement = use_novel_enhancement

        # Decoder stages (coarsest -> finest)
        self.stages = nn.ModuleList()
        in_ch = adapter_channels[-1]
        for i in range(n_levels - 1):
            skip_ch = adapter_channels[-(i + 2)]
            out_ch = skip_ch
            self.stages.append(
                DecoderBlock(in_ch, skip_ch, out_ch,
                             use_pixel_shuffle, use_gated_attention)
            )
            in_ch = out_ch

        # Light conv before head
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        if use_novel_enhancement:
            self.pfg = PeriodicFrequencyGating(
                channels=in_ch,
                periodic_pitch_pixels=periodic_pitch,
                feature_stride=finest_stride,
            )

        self.head = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, features, original_size=None):
        """
        Args:
            features: dict ``{str(i): Tensor}`` or list, finest-first.
            original_size: ``(H, W)`` of the input image for final upsample.
        """
        if isinstance(features, dict):
            features = [features[str(i)]
                        for i in range(len(features))]

        x = features[-1]                               # coarsest
        for i, stage in enumerate(self.stages):
            skip = features[-(i + 2)]
            x = stage(x, skip)

        x = self.final_conv(x)

        if self.use_novel_enhancement:
            x = self.pfg(x)

        if original_size is not None:
            x = F.interpolate(x, size=original_size, mode="bilinear",
                              align_corners=False)

        return self.head(x)


# ====================================================================
#  PART 3 -- Periodic Frequency Gating  (Novel Enhancement)
# ====================================================================
# -----------------------------------------------------------------------
# Physical / Mathematical Intuition
# -----------------------------------------------------------------------
# Semiconductor metal-gate structures exhibit a strict 22-pixel periodic
# pitch, creating strong spectral peaks at harmonic frequencies
#     f_k = k / P   (k = 1, 2, ..., K)
# where P is the pitch in feature-map pixels (= physical_pitch / stride).
#
# A 4-pixel micro-defect is a *localised* disruption.  Its energy spreads
# broadly across the spectrum rather than concentrating at any harmonic.
#
# Strategy:
#   1. 2-D FFT of each channel.
#   2. Multiply by a *learnable* spectral mask with Gaussian notches at
#      each harmonic f_k.  Notch amplitude and bandwidth are trainable
#      so the network learns how aggressively to suppress each harmonic.
#   3. Inverse FFT -> anomaly-enhanced feature.
#   4. Channel-wise SE gating recalibrates the enhanced signal.
#   5. Sigmoid spatial attention gate modulates the original features.
#
# Why this works for 4-pixel defects:
#   - The defect's characteristic frequency (~1/4 = 0.25 cyc/px) is far
#     from the dominant pitch harmonics (1/22 ~ 0.045, 2/22 ~ 0.091, ...),
#     so notching the pitch preserves virtually all defect energy.
#   - After notching, the background power drops dramatically while the
#     defect signal remains, increasing the signal-to-background ratio.
# -----------------------------------------------------------------------


class PeriodicFrequencyGating(nn.Module):
    """Learnable spectral notch filter for periodic-background suppression."""

    def __init__(self, channels, periodic_pitch_pixels=22, feature_stride=4,
                 num_harmonics=8):
        super().__init__()
        self.channels = channels
        self.pitch = periodic_pitch_pixels / feature_stride     # pitch at feature scale
        self.f0 = 1.0 / self.pitch                             # fundamental freq

        # Only keep harmonics below Nyquist (0.5 cyc/px)
        max_k = max(int(0.5 / self.f0), 1)
        self.num_harmonics = min(num_harmonics, max_k)

        # Learnable notch parameters (amplitude, bandwidth) per harmonic
        self.notch_amplitudes = nn.Parameter(torch.zeros(self.num_harmonics))
        self.notch_widths = nn.Parameter(torch.zeros(self.num_harmonics))

        # Channel recalibration (SE-style)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, max(channels // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // 4, 1), channels),
            nn.Sigmoid(),
        )

        # Spatial attention projection
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)

    def _build_notch_mask(self, H, W, device):
        freq_y = torch.fft.fftfreq(H, device=device)
        freq_x = torch.fft.fftfreq(W, device=device)
        fy, fx = torch.meshgrid(freq_y, freq_x, indexing="ij")
        freq_mag = torch.sqrt(fy ** 2 + fx ** 2 + 1e-8)

        mask = torch.ones(H, W, device=device)
        for k in range(1, self.num_harmonics + 1):
            f_k = k * self.f0
            amp = torch.sigmoid(self.notch_amplitudes[k - 1])          # [0, 1]
            bw = F.softplus(self.notch_widths[k - 1]) * self.f0        # > 0
            notch = amp * torch.exp(-0.5 * ((freq_mag - f_k) / bw) ** 2)
            mask = mask * (1.0 - notch)
        return mask

    def forward(self, x):
        B, C, H, W = x.shape

        # Spectral filtering
        x_fft = torch.fft.fft2(x)
        mask = self._build_notch_mask(H, W, x.device)                  # (H, W)
        x_filtered = torch.fft.ifft2(x_fft * mask[None, None]).real

        # Channel recalibration
        ch_gate = self.channel_gate(x_filtered).unsqueeze(-1).unsqueeze(-1)
        x_enhanced = x_filtered * ch_gate

        # Spatial attention
        spatial_gate = torch.sigmoid(self.proj(x_enhanced))
        return x * spatial_gate


# ====================================================================
#  Mock Integration / Demo
# ====================================================================

class _MockDINOv3Backbone(nn.Module):
    """Minimal mock that satisfies the adapter's backbone interface."""

    def __init__(self, embed_dim=384, patch_size=14):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self._dummy = nn.Parameter(torch.zeros(1))  # non-empty parameters

    def requires_grad_(self, flag):
        for p in self.parameters():
            p.requires_grad_(flag)
        return self

    def get_intermediate_layers(self, x, n, return_class_token=False):
        bs, _, h, w = x.shape
        H_toks, W_toks = h // self.patch_size, w // self.patch_size
        n_tokens = H_toks * W_toks
        indices = n if isinstance(n, (list, tuple)) else list(range(n))
        results = []
        for _ in indices:
            tokens = torch.randn(
                bs, n_tokens, self.embed_dim,
                device=x.device, dtype=x.dtype,
            )
            if return_class_token:
                cls = torch.randn(
                    bs, 1, self.embed_dim,
                    device=x.device, dtype=x.dtype,
                )
                results.append((tokens, cls))
            else:
                results.append(tokens)
        return results


def build_pipeline(
    msda_mode="original",
    embed_dim=384,
    patch_size=14,
    num_classes=2,
    use_pixel_shuffle=False,
    use_gated_attention=False,
    use_novel_enhancement=False,
    periodic_pitch=22,  
    backbone=None,
):
    """Convenience builder for the full Adapter -> Decoder pipeline."""
    if backbone is None:
        backbone = _MockDINOv3Backbone(embed_dim=embed_dim, patch_size=patch_size)

    adapter = DINOv3_Adapter(
        backbone=backbone,
        msda_mode=msda_mode,
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        with_cp=False,
    )

    n_out = {"original": 4, "4-level": 4, "5-level": 5}[msda_mode]
    finest_stride = {"original": 4, "4-level": 4, "5-level": 2}[msda_mode]

    decoder = DynamicUNetDecoder(
        adapter_channels=[embed_dim] * n_out,
        num_classes=num_classes,
        use_pixel_shuffle=use_pixel_shuffle,
        use_gated_attention=use_gated_attention,
        use_novel_enhancement=use_novel_enhancement,
        periodic_pitch=periodic_pitch,
        finest_stride=finest_stride,
    )

    return adapter, decoder


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use an image size divisible by 32 AND by patch_size (14)
    # 448 = 14*32 = 32*14, satisfies both
    img = torch.randn(1, 3, 448, 448, device=device)

    for mode in ("original", "4-level", "5-level"):
        print(f"\n{'='*60}")
        print(f"  msda_mode = {mode!r}")
        print(f"{'='*60}")

        adapter, decoder = build_pipeline(
            msda_mode=mode,
            embed_dim=384,
            num_classes=2,
            use_pixel_shuffle=True,
            use_gated_attention=True,
            use_novel_enhancement=True,
            periodic_pitch=22,
        )
        adapter = adapter.to(device)
        decoder = decoder.to(device)

        feats = adapter(img)
        print("Adapter outputs:")
        for k in sorted(feats.keys(), key=int):
            print(f"  level {k}: {tuple(feats[k].shape)}")

        logits = decoder(feats, original_size=(448, 448))
        print(f"Decoder logits: {tuple(logits.shape)}")

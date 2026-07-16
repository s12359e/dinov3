"""Canonical DINOv3 ViT-B/16 construction and checkpoint contract."""

from __future__ import annotations

from collections.abc import Mapping

from dinov3.hub.backbones import dinov3_vitb16


BACKBONE_CONFIG = {
    "name": "dinov3_vitb16",
    "patch_size": 16,
    "embed_dim": 768,
    "n_storage_tokens": 4,
    "pos_embed_rope_rescale_coords": 2,
    "pos_embed_rope_dtype": "fp32",
    "layerscale_init": 1.0e-5,
    "norm_layer": "layernormbf16",
    "mask_k_bias": True,
}


def build_canonical_backbone():
    """Build the exact architecture expected by official DINOv3 ViT-B/16 weights."""
    return dinov3_vitb16(pretrained=False)


def export_backbone_config() -> dict:
    return dict(BACKBONE_CONFIG)


def validate_backbone_config(config: Mapping | None) -> None:
    """Reject a checkpoint that explicitly declares a different architecture."""
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise ValueError("backbone_config must be a mapping")
    for key, expected in BACKBONE_CONFIG.items():
        if key not in config:
            raise ValueError(f"backbone_config missing {key!r}")
        if config[key] != expected:
            raise ValueError(
                f"unsupported backbone_config[{key!r}]={config[key]!r}; "
                f"expected {expected!r}")

"""Small task-specific heads deployed on top of the DINOv3 backbone."""

from .backbone import (
    build_canonical_backbone,
    export_backbone_config,
    validate_backbone_config,
)
from .order_aware_fusion import (
    OrderAwareTripletFusionHead,
    is_target_unique_presence,
    select_safe_background_negatives,
    target_unique_fusion_loss,
)

__all__ = [
    "build_canonical_backbone",
    "export_backbone_config",
    "validate_backbone_config",
    "OrderAwareTripletFusionHead",
    "is_target_unique_presence",
    "select_safe_background_negatives",
    "target_unique_fusion_loss",
]

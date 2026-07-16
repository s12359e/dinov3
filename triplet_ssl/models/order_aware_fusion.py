"""Order-aware fusion of target/ref1/ref2 DINO patch tokens.

The semiconductor inspection truth table is directional: only a PSF that is
present in ``target`` and absent from both references is a defect.  A symmetric
distance cannot distinguish ``(target, ref1, ref2) == (1, 0, 0)`` from
``(0, 1, 1)``.  This head therefore treats target as a distinct role while
keeping ref1/ref2 exchangeable.

Inputs and output are pointwise in the patch-token dimension, so one trained
head works for any tile/grid size::

    target, ref1, ref2: (B, N, D), L2-normalized internally
    target_unique_logit: (B, N)
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_NAME = "order_aware_target_unique_v1"
TARGET_UNIQUE_PRESENCE = (1, 0, 0)
ALL_PRESENCE_PATTERNS = tuple(product((0, 1), repeat=3))


def is_target_unique_presence(presence: Sequence[int | bool]) -> bool:
    """Return the production label for presence ordered as target/ref1/ref2."""
    if len(presence) != 3:
        raise ValueError("presence must contain exactly (target, ref1, ref2)")
    p = tuple(int(bool(v)) for v in presence)
    return p == TARGET_UNIQUE_PRESENCE


class OrderAwareTripletFusionHead(nn.Module):
    """Target-asymmetric, reference-symmetric per-patch relation head.

    A shared projection prevents the head from assigning arbitrary feature
    spaces to the two references.  Symmetric mean/absolute-difference terms make
    ref1/ref2 swapping an exact architectural invariant.  The signed
    ``target - reference_mean`` term preserves the direction needed to separate
    target-extra ``100`` from target-missing ``011``.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < prior_prob < 1.0:
            raise ValueError("prior_prob must be in (0, 1)")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.prior_prob = float(prior_prob)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(5 * self.hidden_dim),
            nn.Linear(5 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        # A rare-positive prior avoids a random sea of high anomaly scores at
        # initialization without constraining what the supervised head can learn.
        nn.init.constant_(self.fusion[-1].bias, math.log(prior_prob / (1.0 - prior_prob)))

    @staticmethod
    def _validate_tokens(target: torch.Tensor, ref1: torch.Tensor, ref2: torch.Tensor) -> None:
        if target.ndim != 3:
            raise ValueError(f"fusion inputs must be (B,N,D), got {tuple(target.shape)}")
        if ref1.shape != target.shape or ref2.shape != target.shape:
            raise ValueError(
                "target/ref1/ref2 token shapes must match: "
                f"{tuple(target.shape)}, {tuple(ref1.shape)}, {tuple(ref2.shape)}"
            )

    def fused_features(
        self, target: torch.Tensor, ref1: torch.Tensor, ref2: torch.Tensor
    ) -> torch.Tensor:
        """Return the role-aware representation before the final fusion MLP."""
        self._validate_tokens(target, ref1, ref2)
        if target.shape[-1] != self.in_dim:
            raise ValueError(f"expected token dim {self.in_dim}, got {target.shape[-1]}")

        target = F.normalize(target, dim=-1)
        ref1 = F.normalize(ref1, dim=-1)
        ref2 = F.normalize(ref2, dim=-1)
        pt = self.token_proj(target)
        p1 = self.token_proj(ref1)
        p2 = self.token_proj(ref2)
        pref = 0.5 * (p1 + p2)
        signed = pt - pref
        return torch.cat(
            (pt, pref, signed, signed.abs(), (p1 - p2).abs()), dim=-1
        )

    def forward(
        self, target: torch.Tensor, ref1: torch.Tensor, ref2: torch.Tensor
    ) -> torch.Tensor:
        logits = self.fusion(self.fused_features(target, ref1, ref2)).squeeze(-1)
        if logits.shape != target.shape[:2]:
            raise RuntimeError(
                f"fusion head returned {tuple(logits.shape)}, expected {tuple(target.shape[:2])}"
            )
        return logits

    def export_config(self, *, patch_size: int = 16, train_tile: int = 128) -> dict[str, Any]:
        """Serializable architecture/deployment contract stored in checkpoints."""
        max_halo = ((int(train_tile) - int(patch_size)) // (2 * int(patch_size))) \
            * int(patch_size)
        return {
            "name": ARCH_NAME,
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "prior_prob": self.prior_prob,
            "output": "target_unique_logit",
            "truth_table": "target_only_100",
            "token_normalization": "l2",
            "reference_symmetric": True,
            "patch_size": int(patch_size),
            "train_tile": int(train_tile),
            "context_radius_tokens": 0,
            "inference_context_halo": max(0, min(2 * int(patch_size), max_halo)),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "OrderAwareTripletFusionHead":
        """Build a head after validating the checkpoint semantic contract."""
        required = {
            "name",
            "in_dim",
            "hidden_dim",
            "dropout",
            "output",
            "token_normalization",
            "reference_symmetric",
            "patch_size",
            "train_tile",
            "truth_table",
            "context_radius_tokens",
        }
        missing = sorted(required.difference(config))
        if missing:
            raise ValueError(f"fusion_head_config missing keys: {missing}")
        if config["name"] != ARCH_NAME:
            raise ValueError(f"unsupported fusion architecture: {config['name']!r}")
        if config["output"] != "target_unique_logit":
            raise ValueError(f"unsupported fusion output: {config['output']!r}")
        if config["truth_table"] != "target_only_100":
            raise ValueError(f"unsupported fusion truth table: {config['truth_table']!r}")
        if config["token_normalization"] != "l2":
            raise ValueError("fusion checkpoint must use L2-normalized patch tokens")
        if config["reference_symmetric"] is not True:
            raise ValueError("fusion checkpoint must declare reference_symmetric=true")
        if int(config.get("context_radius_tokens", 0)) != 0:
            raise ValueError("fusion v1 supports only pointwise context_radius_tokens=0")
        patch_size = int(config["patch_size"])
        train_tile = int(config["train_tile"])
        if patch_size <= 0 or train_tile <= 0 or train_tile % patch_size:
            raise ValueError("fusion patch_size/train_tile must be positive and divisible")
        halo = int(config.get("inference_context_halo", 0))
        if halo < 0 or halo % patch_size or 2 * halo >= train_tile:
            raise ValueError("invalid fusion inference_context_halo")
        head = cls(
            in_dim=int(config["in_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            prior_prob=float(config.get("prior_prob", 0.01)),
        )
        head.checkpoint_config = dict(config)
        return head


def target_unique_fusion_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    event_mask: torch.Tensor,
    background_negative_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Class-balanced BCE on positions whose synthetic truth table is known.

    Only injected event patches participate.  Unlabelled image background is
    ignored, because it may contain a real target-only defect.  Positive and
    negative terms are averaged separately so a single ``100`` event is not
    overwhelmed by the six nuisance presence combinations.
    """
    if logits.shape != positive_mask.shape or logits.shape != event_mask.shape:
        raise ValueError(
            "fusion logits/positive_mask/event_mask shapes must match: "
            f"{tuple(logits.shape)}, {tuple(positive_mask.shape)}, {tuple(event_mask.shape)}"
        )
    valid = event_mask.bool()
    if background_negative_mask is not None:
        if background_negative_mask.shape != logits.shape:
            raise ValueError("background_negative_mask shape must match fusion logits")
        valid = valid | background_negative_mask.bool()
    pos = valid & positive_mask.bool()
    neg = valid & ~positive_mask.bool()
    terms = []
    if pos.any():
        terms.append(F.softplus(-logits[pos]).mean())
    if neg.any():
        terms.append(F.softplus(logits[neg]).mean())
    loss = torch.stack(terms).mean() if terms else logits.sum() * 0.0

    with torch.no_grad():
        prob = logits.sigmoid()
        pos_score = float(prob[pos].mean()) if pos.any() else 0.0
        neg_score = float(prob[neg].mean()) if neg.any() else 0.0
    return loss, {
        "fusion_pos_score": pos_score,
        "fusion_neg_score": neg_score,
        "fusion_n_pos": int(pos.sum()),
        "fusion_n_neg": int(neg.sum()),
    }


@torch.no_grad()
def select_safe_background_negatives(
    target: torch.Tensor,
    ref1: torch.Tensor,
    ref2: torch.Tensor,
    event_mask: torch.Tensor,
    fraction: float = 0.25,
) -> torch.Tensor:
    """Select low-residual non-event patches as contamination-robust negatives.

    Synthetic event masks deliberately leave real image background unlabelled.
    Without any clean ``000`` examples, however, most production patches would
    never constrain the fusion logit.  The lowest best-reference residuals are
    safe negatives under the production rule: target already matches at least
    one reference.  High-residual background, where an unknown real defect may
    live, remains ignored.
    """
    OrderAwareTripletFusionHead._validate_tokens(target, ref1, ref2)
    if event_mask.shape != target.shape[:2]:
        raise ValueError("event_mask shape must match (B,N)")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("background negative fraction must be in [0,1]")
    selected = torch.zeros_like(event_mask, dtype=torch.bool)
    if fraction == 0.0:
        return selected

    target = F.normalize(target, dim=-1)
    ref1 = F.normalize(ref1, dim=-1)
    ref2 = F.normalize(ref2, dim=-1)
    residual = torch.minimum(
        torch.linalg.vector_norm(target - ref1, dim=-1),
        torch.linalg.vector_norm(target - ref2, dim=-1),
    )
    candidates = ~event_mask.bool()
    n_tokens = target.shape[1]
    k = max(1, int(round(n_tokens * fraction)))
    idx = residual.masked_fill(~candidates, float("inf")).topk(
        min(k, n_tokens), dim=1, largest=False).indices
    selected.scatter_(1, idx, True)
    return selected & candidates

"""Triplet DINO/iBOT loss: pairing table + robust top-k exemption + repulsion.

Prototype projection uses dinov3's `DINOHead` (constructed in the training script and
passed in via the model). Teacher sharpening uses centering (EMA), matching stock DINO;
the cross-entropy is the exact `lossfunc` recipe from `dinov3.loss.ibot_patch_loss`.

Pairing table (teacher <-> student), computed at CLS and PATCH level:

  ref-ref     (w_ref2ref)     : t(ref1)<->s(ref2) + t(ref2)<->s(ref1)  -- full, NO exemption
  target-ref  (w_target2ref)  : t(ref1)<->s(target) + t(ref2)<->s(target) -- robust top-k exempt
  traditional (w_traditional) : t(img)<->s(img), img rotating over {target,ref1,ref2}

Robust top-k exemption (target-ref patch term ONLY): per image, drop the top-k% highest
per-patch losses before averaging. Defect patches legitimately cannot match the
reference; forcing them to would collapse defect features into normality. NEVER applied
to ref-ref.

Repulsion (phase 3): L_repel = -lambda * mean(per_patch_loss[synth_mask]) on the
target-ref patch term -- actively pushes synthetic-defect patches away from the reference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ce_lastdim(student_logits, teacher_probs, temp):
    """Per-token cross-entropy along the prototype dim. Shapes broadcast on (..., K)."""
    return -torch.sum(teacher_probs * F.log_softmax(student_logits / temp, dim=-1), dim=-1)


class TripletLoss(nn.Module):
    def __init__(self, cls_out_dim, patch_out_dim, student_temp=0.1, teacher_temp=0.04,
                 center_momentum=0.9, topk_pct=0.02, weights=None, repel_lambda=0.0,
                 cls_weight=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.topk_pct = topk_pct
        self.repel_lambda = repel_lambda
        self.cls_weight = cls_weight  # relative weight of CLS vs patch term
        w = weights or dict(traditional=0.3, ref2ref=0.4, target2ref=0.3)
        self.w = w
        self.register_buffer("center_cls", torch.zeros(1, cls_out_dim))
        self.register_buffer("center_patch", torch.zeros(1, 1, patch_out_dim))

    # -- teacher sharpening (centering, EMA) -------------------------------- #
    @torch.no_grad()
    def _sharpen_cls(self, t):
        return F.softmax((t - self.center_cls) / self.teacher_temp, dim=-1)

    @torch.no_grad()
    def _sharpen_patch(self, t):
        return F.softmax((t - self.center_patch) / self.teacher_temp, dim=-1)

    @torch.no_grad()
    def _update_centers(self, cls_logits, patch_logits):
        bc = cls_logits.mean(dim=0, keepdim=True)
        self.center_cls.mul_(self.center_momentum).add_(bc, alpha=1 - self.center_momentum)
        bp = patch_logits.mean(dim=(0, 1), keepdim=True)
        self.center_patch.mul_(self.center_momentum).add_(bp, alpha=1 - self.center_momentum)

    # -- pair losses -------------------------------------------------------- #
    def _cls_pair(self, s_logits, t_logits):
        t = self._sharpen_cls(t_logits)
        return _ce_lastdim(s_logits, t, self.student_temp).mean()

    def _cls_pair_per_sample(self, s_logits, t_logits):
        t = self._sharpen_cls(t_logits)
        return _ce_lastdim(s_logits, t, self.student_temp)  # (B,)

    def _patch_per_patch(self, s_logits, t_logits):
        t = self._sharpen_patch(t_logits)
        return _ce_lastdim(s_logits, t, self.student_temp)  # (B, N)

    @staticmethod
    def _robust_mean(loss_bn, pct, exclude_mask=None):
        """Mean over patches after dropping the top-`pct` fraction per image.

        `exclude_mask` (B,N, truthy = exclude) removes patches from the mean
        entirely -- known synthetic-defect patches must never be pulled toward
        the reference. Excluded patches also don't consume top-k slots (they are
        masked to -inf before the top-k selection), so the k exemptions remain
        available for real defects."""
        B, N = loss_bn.shape
        keep = torch.ones_like(loss_bn, dtype=torch.bool)
        if exclude_mask is not None:
            keep &= ~exclude_mask.bool()
        k = int(N * pct)
        if k > 0:
            masked = loss_bn.masked_fill(~keep, float("-inf"))
            topk_idx = masked.topk(k, dim=1).indices
            keep.scatter_(1, topk_idx, False)
        return (loss_bn * keep).sum() / keep.sum().clamp(min=1)

    def forward(self, cls, patch, trad_key="target", synth_mask=None,
                cls_exempt=None):
        """cls[name] / patch[name] = dict(s=student_logits, t=teacher_logits).

        name in {ref1, ref2, target}. cls logits (B,Kc); patch logits (B,N,Kp).
        cls_exempt: optional (B,) bool -- samples whose target is KNOWN to contain
        a defect (synthetic injection or has_defect flag). Their target<->ref CLS
        term is dropped entirely: pulling a defect-bearing global feature toward a
        defect-free reference is CLS-level normality collapse. RR/TD CLS are
        unaffected (refs are clean; the same-image pair is consistent either way).
        Returns (total_loss, log_dict).
        """
        logs = {}

        # 1. ref-ref : full consistency, no exemption --------------------------
        rr_patch = 0.5 * (self._patch_per_patch(patch["ref2"]["s"], patch["ref1"]["t"]).mean()
                          + self._patch_per_patch(patch["ref1"]["s"], patch["ref2"]["t"]).mean())
        rr_cls = 0.5 * (self._cls_pair(cls["ref2"]["s"], cls["ref1"]["t"])
                        + self._cls_pair(cls["ref1"]["s"], cls["ref2"]["t"]))
        L_rr = rr_patch + self.cls_weight * rr_cls

        # 2. target-ref : robust top-k exemption on the patch term -------------
        tr_pp = 0.5 * (self._patch_per_patch(patch["target"]["s"], patch["ref1"]["t"])
                       + self._patch_per_patch(patch["target"]["s"], patch["ref2"]["t"]))  # (B,N)
        # Known synthetic-defect patches are excluded from the pull (mask-certain),
        # on top of the statistical top-k exemption for unlabeled real defects.
        tr_patch = self._robust_mean(tr_pp, self.topk_pct, exclude_mask=synth_mask)
        tr_cls_ps = 0.5 * (self._cls_pair_per_sample(cls["target"]["s"], cls["ref1"]["t"])
                           + self._cls_pair_per_sample(cls["target"]["s"], cls["ref2"]["t"]))
        if cls_exempt is not None and cls_exempt.any():
            keep = ~cls_exempt.bool()
            tr_cls = tr_cls_ps[keep].mean() if keep.any() else tr_cls_ps.sum() * 0.0
        else:
            tr_cls = tr_cls_ps.mean()
        L_tr = tr_patch + self.cls_weight * tr_cls

        # 3. traditional same-image pair (rotating) ----------------------------
        td_patch = self._patch_per_patch(patch[trad_key]["s"], patch[trad_key]["t"]).mean()
        td_cls = self._cls_pair(cls[trad_key]["s"], cls[trad_key]["t"])
        L_td = td_patch + self.cls_weight * td_cls

        total = (self.w["ref2ref"] * L_rr + self.w["target2ref"] * L_tr
                 + self.w["traditional"] * L_td)

        # Repulsion (phase 3): push synthetic-defect patches away from refs.
        if self.repel_lambda > 0 and synth_mask is not None and synth_mask.sum() > 0:
            m = synth_mask.bool()
            repel = -self.repel_lambda * tr_pp[m].mean()
            total = total + repel
            logs["repel"] = float(repel.detach())

        # Standard DINO order: losses use the OLD center; update centers afterwards.
        self._update_centers(
            torch.cat([cls[n]["t"] for n in ("ref1", "ref2", "target")], 0),
            torch.cat([patch[n]["t"] for n in ("ref1", "ref2", "target")], 0),
        )

        logs.update(L_refref=float(L_rr.detach()), L_target2ref=float(L_tr.detach()),
                    L_trad=float(L_td.detach()),
                    exempt_pct=self.topk_pct)
        if cls_exempt is not None:
            logs["n_cls_exempt"] = int(cls_exempt.sum())
        return total, logs

    @torch.no_grad()
    def exempted_patch_map(self, patch, sample_idx=0):
        """For sanity logging: which patches would be exempted on target-ref (the
        highest-loss top-k%). Returns a boolean (N,) map for one sample."""
        tr_pp = 0.5 * (self._patch_per_patch(patch["target"]["s"], patch["ref1"]["t"])
                       + self._patch_per_patch(patch["target"]["s"], patch["ref2"]["t"]))
        loss_n = tr_pp[sample_idx]
        N = loss_n.numel()
        k = int(N * self.topk_pct)
        exempt = torch.zeros(N, dtype=torch.bool, device=loss_n.device)
        if k > 0:
            exempt[loss_n.topk(k).indices] = True
        return exempt

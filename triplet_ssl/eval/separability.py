"""Patch-feature separability evaluation (GT masks read HERE ONLY).

For each triplet, per-patch residual:

    r_k = || f(target)_k - mean(f(ref1)_k, f(ref2)_k) ||_2

A patch is labelled "defect" if it overlaps the GT mask beyond a threshold (default: any
overlap). Reports:

  * patch-level AUROC (defect vs normal) -- the gate every phase must beat.
  * mean-residual(defect) / mean-residual(normal) ratio.
  * ~N residual heatmaps overlaid with GT contours.
  * a normal-only guardrail (residuals should stay low/uniform -> false-positive tendency).

This module must never be imported by the training path. `train_triplet.py` calls it only
for evaluation between phases.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from triplet_ssl import IMG_MEAN, IMG_STD

_MEAN = np.array(IMG_MEAN, np.float32).reshape(1, 1, 3)
_STD = np.array(IMG_STD, np.float32).reshape(1, 1, 3)


def _auroc(scores, labels):
    """Dependency-free AUROC via the Mann-Whitney U statistic (average ranks)."""
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _load_norm(path, img_size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) - _MEAN) / _STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


@torch.no_grad()
def _patch_tokens(backbone, x):
    out = backbone.get_intermediate_layers(x, n=[len(backbone.blocks) - 1],
                                           return_class_token=False, norm=True)
    tok = out[0].squeeze(0)                 # (N, D)
    return F.normalize(tok, dim=-1)


def _residual(tg, r1, r2, mode="mean"):
    """Per-patch residual of target vs the two refs.

    mean: ||f(t) - (f(r1)+f(r2))/2||  -- lowest noise when both refs are clean.
    min : min(||f(t)-f(r1)||, ||f(t)-f(r2)||) -- classic double detection:
          target only needs to match ONE ref, so a defect/nuisance present in a
          single ref cannot raise the score (robust to dirty refs).
    """
    if mode == "min":
        return torch.minimum(torch.norm(tg - r1, dim=-1),
                             torch.norm(tg - r2, dim=-1))
    return torch.norm(tg - 0.5 * (r1 + r2), dim=-1)


def _triplet_score(tg, r1, r2, mode="mean", fusion_head=None):
    """Use the deployed scoring path when a trained fusion head is supplied."""
    if fusion_head is None:
        return _residual(tg, r1, r2, mode)
    logits = fusion_head(tg.unsqueeze(0), r1.unsqueeze(0), r2.unsqueeze(0))
    if logits.shape != (1, tg.shape[0]) or not torch.isfinite(logits).all():
        raise RuntimeError("fusion head produced invalid evaluation logits")
    return logits[0].sigmoid()


def _mask_to_patch_labels(mask, grid, patch_size, overlap_thresh):
    m = mask[: grid * patch_size, : grid * patch_size]
    m = m.reshape(grid, patch_size, grid, patch_size).sum(axis=(1, 3))
    return (m > overlap_thresh).astype(np.int64).reshape(-1)


def _heatmap(target_path, residual, grid, img_size, mask, out_path):
    r = residual.reshape(grid, grid)
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    r = cv2.resize((r * 255).astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(r, cv2.COLORMAP_JET)
    base = cv2.resize(cv2.imread(str(target_path)), (img_size, img_size))
    overlay = cv2.addWeighted(base, 0.5, heat, 0.5, 0)
    if mask is not None and mask.max() > 0:
        mm = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours((mm > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), overlay)


@torch.no_grad()
def evaluate_separability(backbone, root, split, device, img_size=224, patch_size=16,
                          overlap_thresh=0, n_heatmaps=10, out_dir=None, tag="phase",
                          residual_mode="mean", fusion_head=None):
    """Returns a metrics dict. Writes heatmaps to out_dir if given."""
    backbone = backbone.to(device).eval()
    if fusion_head is not None:
        fusion_head = fusion_head.to(device).eval()
    root = Path(root) / split
    with open(root / "manifest.json") as f:
        items = json.load(f)
    grid = img_size // patch_size

    all_res, all_lab = [], []
    def_res, norm_res = [], []          # per-patch residual pooled by GT label
    heatmaps_written = 0
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    for m in items:
        d = root / m["id"]
        r1 = _patch_tokens(backbone, _load_norm(d / "ref1.png", img_size).to(device))
        r2 = _patch_tokens(backbone, _load_norm(d / "ref2.png", img_size).to(device))
        tg = _patch_tokens(backbone, _load_norm(d / "target.png", img_size).to(device))
        residual = _triplet_score(tg, r1, r2, residual_mode,
                                  fusion_head).cpu().numpy()                 # (N,)

        mask = cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE)         # GT: eval only
        if mask.shape != (img_size, img_size):
            mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        labels = _mask_to_patch_labels(mask, grid, patch_size, overlap_thresh)

        all_res.append(residual)
        all_lab.append(labels)
        def_res.append(residual[labels == 1])
        norm_res.append(residual[labels == 0])

        if out_dir and heatmaps_written < n_heatmaps and m["has_defect"]:
            _heatmap(d / "target.png", residual, grid, img_size, mask,
                     Path(out_dir) / f"{tag}_{m['id']}_"
                     f"{'fusion' if fusion_head is not None else 'residual'}.png")
            heatmaps_written += 1

    res = np.concatenate(all_res)
    lab = np.concatenate(all_lab)
    def_res = np.concatenate([d for d in def_res if d.size]) if any(d.size for d in def_res) else np.array([])
    norm_res = np.concatenate([d for d in norm_res if d.size])

    auroc = _auroc(res, lab)
    ratio = (def_res.mean() / (norm_res.mean() + 1e-8)) if def_res.size else float("nan")
    return dict(auroc=float(auroc), defect_normal_ratio=float(ratio),
                residual_mode=residual_mode,
                score_method="fusion" if fusion_head is not None else "residual",
                evaluation_pipeline="resized_folder_eval",
                mean_res_defect=float(def_res.mean()) if def_res.size else float("nan"),
                mean_res_normal=float(norm_res.mean()),
                n_defect_patches=int((lab == 1).sum()),
                n_normal_patches=int((lab == 0).sum()),
                n_triplets=len(items))


@torch.no_grad()
def evaluate_guardrail(backbone, root, split, device, **kw):
    """Normal-only guardrail: residual stats on triplets whose target has NO defect."""
    root_p = Path(root) / split
    with open(root_p / "manifest.json") as f:
        items = json.load(f)
    normals = [m for m in items if not m["has_defect"]]
    if not normals:
        return dict(guardrail_mean_res=float("nan"), guardrail_p99_res=float("nan"), n=0)

    backbone = backbone.to(device).eval()
    img_size = kw.get("img_size", 224)
    residual_mode = kw.get("residual_mode", "mean")
    fusion_head = kw.get("fusion_head")
    if fusion_head is not None:
        fusion_head = fusion_head.to(device).eval()
    all_res = []
    for m in normals:
        d = root_p / m["id"]
        r1 = _patch_tokens(backbone, _load_norm(d / "ref1.png", img_size).to(device))
        r2 = _patch_tokens(backbone, _load_norm(d / "ref2.png", img_size).to(device))
        tg = _patch_tokens(backbone, _load_norm(d / "target.png", img_size).to(device))
        all_res.append(_triplet_score(tg, r1, r2, residual_mode,
                                      fusion_head).cpu().numpy())
    res = np.concatenate(all_res)
    return dict(guardrail_mean_res=float(res.mean()),
                guardrail_p99_res=float(np.percentile(res, 99)), n=len(normals))

"""Inference: triplet TIFF -> per-patch anomaly map + defect detections.

Recipe (must MATCH training conditions):
  read tiff (ch1=target, ch2=ref1, ch3=ref2) -> [optional registration, same
  setting as training] -> same IMG_MEAN/STD normalization -> TEACHER backbone
  (eval, no grad) at native resolution in 128px tiles -> L2-normalized patch
  tokens -> deployed order-aware fusion head (or legacy residual/KNN) -> score map
  (x16 upsample)
  -> [threshold, calibrated on NORMAL tiffs only] -> connected components ->
  defect boxes.

Calibrate a threshold on a directory of known-normal TIFFs:
    python triplet_ssl/infer.py --checkpoint runs/exp/phase2_teacher.pth \
        --input data/normal_tiffs --calib

Run detection:
    python triplet_ssl/infer.py --checkpoint runs/exp/phase3_teacher.pth \
        --input path/to/tiffs --out-dir infer_out --method fusion \
        --threshold 0.31 [--register] [--device cuda]

Outputs per TIFF: <stem>_heatmap.png (overlay), <stem>_scores.npy (patch grid),
plus a single detections.json.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triplet_ssl import IMG_MEAN, IMG_STD
from triplet_ssl.data.triplet_dataset import read_tiff3, coarse_register
from triplet_ssl.eval.separability import _residual
from triplet_ssl.models.order_aware_fusion import OrderAwareTripletFusionHead
from triplet_ssl.models.backbone import build_canonical_backbone, validate_backbone_config

_MEAN = np.array(IMG_MEAN, np.float32).reshape(1, 1, 3)
_STD = np.array(IMG_STD, np.float32).reshape(1, 1, 3)


def _extract_backbone_state(ckpt):
    if not isinstance(ckpt, dict):
        raise TypeError("checkpoint must contain a state-dict mapping")
    state = ckpt
    for key in ("teacher_backbone", "model", "teacher", "state_dict"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    cleaned = {}
    for key, value in state.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
        cleaned[name] = value
    return cleaned


def _load_backbone_from_checkpoint(ckpt, device, backbone_factory=None):
    if isinstance(ckpt, dict):
        validate_backbone_config(ckpt.get("backbone_config"))
    m = (backbone_factory or build_canonical_backbone)()
    state = _extract_backbone_state(ckpt)
    msg = m.load_state_dict(state, strict=False)
    if msg.missing_keys:
        raise ValueError(
            "checkpoint does not fully cover the inference backbone; missing keys: "
            f"{msg.missing_keys[:8]}{'...' if len(msg.missing_keys) > 8 else ''}")
    if msg.unexpected_keys:
        raise ValueError(
            "checkpoint contains weights outside the canonical inference backbone; "
            f"unexpected keys: {msg.unexpected_keys[:8]}"
            f"{'...' if len(msg.unexpected_keys) > 8 else ''}")
    print(f"[load] backbone missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}")
    return m.to(device).eval()


def load_backbone(ckpt_path, device):
    """Legacy-compatible backbone-only loader."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return _load_backbone_from_checkpoint(ckpt, device)


def load_inference_bundle(ckpt_path, device, *, backbone_factory=None):
    """Load the EMA backbone and optional strictly validated fusion head.

    Backbone-only legacy checkpoints remain valid for residual/KNN inference.
    A partial fusion bundle (state without config or vice versa) is rejected so
    deployment can never silently use a random/mismatched relation head.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = _load_backbone_from_checkpoint(ckpt, device, backbone_factory)
    if not isinstance(ckpt, dict):
        return backbone, None, {"checkpoint_version": 0}

    have_state = "teacher_fusion_head" in ckpt
    have_config = "fusion_head_config" in ckpt
    if have_state != have_config:
        raise ValueError(
            "invalid deployment checkpoint: teacher_fusion_head and "
            "fusion_head_config must either both be present or both be absent")
    metadata = {
        "checkpoint_version": int(ckpt.get("checkpoint_version", 1)),
        "phase": ckpt.get("phase"),
        "global_step": ckpt.get("global_step"),
        "validation": ckpt.get("validation"),
        "preprocess": ckpt.get("preprocess"),
    }
    if not have_state and metadata["preprocess"] is None:
        warnings.warn(
            "legacy checkpoint has no preprocessing metadata; using source defaults "
            "for channel order, normalization, uint16 range, and registration",
            RuntimeWarning,
            stacklevel=2,
        )
        return backbone, None, metadata
    if have_state and metadata["checkpoint_version"] != 2:
        raise ValueError(
            f"unsupported fusion checkpoint_version={metadata['checkpoint_version']}; expected 2")

    preprocess = metadata["preprocess"]
    required_pre = {"mean", "std", "input_scaling", "channel_order",
                    "source_channel_indices", "register"}
    if not isinstance(preprocess, dict) or not required_pre.issubset(preprocess):
        missing = sorted(required_pre.difference(preprocess or {}))
        raise ValueError(f"checkpoint preprocess metadata missing keys: {missing}")
    mean = np.asarray(preprocess["mean"], dtype=np.float32)
    std = np.asarray(preprocess["std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() \
            or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("checkpoint normalization must contain finite mean/std triples")
    scaling = preprocess["input_scaling"]
    if scaling == "read_tiff3_uint8":
        # Backward compatibility for the first v2 bundle: uint16 used its high
        # byte, which is effectively a 0..65535 acquisition range.
        preprocess.setdefault("uint16_black_level", 0.0)
        preprocess.setdefault("uint16_white_level", 65535.0)
        preprocess["uint16_decode_mode"] = "legacy_high_byte"
        preprocess.setdefault("source_dtype", None)
    elif scaling in {"fixed_uint16_range_to_uint8_v1",
                     "fixed_uint16_range_to_0_255_float_v1"}:
        if not {"uint16_black_level", "uint16_white_level"}.issubset(preprocess):
            raise ValueError("fixed uint16 scaling requires black/white level metadata")
        preprocess["uint16_decode_mode"] = (
            "uint8_linear" if scaling == "fixed_uint16_range_to_uint8_v1"
            else "float_linear")
        preprocess.setdefault("source_dtype", None)
    elif scaling == "uint8_0_255_identity_v1":
        preprocess.setdefault("uint16_black_level", 0.0)
        preprocess.setdefault("uint16_white_level", 65535.0)
        preprocess["uint16_decode_mode"] = "float_linear"
        preprocess["source_dtype"] = "uint8"
    elif scaling == "float_0_255_identity_v1":
        preprocess.setdefault("uint16_black_level", 0.0)
        preprocess.setdefault("uint16_white_level", 65535.0)
        preprocess["uint16_decode_mode"] = "float_linear"
        source_dtype = str(preprocess.get("source_dtype", ""))
        if not source_dtype.startswith("float"):
            raise ValueError("float TIFF scaling requires a floating source_dtype")
    else:
        raise ValueError(f"unsupported input scaling: {scaling!r}")
    black = float(preprocess["uint16_black_level"])
    white = float(preprocess["uint16_white_level"])
    if not np.isfinite(black) or not np.isfinite(white) or white <= black:
        raise ValueError("invalid uint16 black/white level metadata")
    if list(preprocess["channel_order"]) != ["target", "ref1", "ref2"]:
        raise ValueError("checkpoint channel roles must be target/ref1/ref2")
    source_order = tuple(int(i) for i in preprocess["source_channel_indices"])
    if sorted(source_order) != [0, 1, 2]:
        raise ValueError("source_channel_indices must be a permutation of [0,1,2]")

    if not have_state:
        return backbone, None, metadata

    config = ckpt["fusion_head_config"]
    fusion_head = OrderAwareTripletFusionHead.from_config(config)
    backbone_dim = getattr(backbone, "embed_dim", None)
    if backbone_dim is not None and int(backbone_dim) != int(config["in_dim"]):
        raise ValueError(
            f"fusion in_dim={config['in_dim']} does not match backbone embed_dim={backbone_dim}")
    fusion_head.load_state_dict(ckpt["teacher_fusion_head"], strict=True)
    fusion_head = fusion_head.to(device).eval()
    metadata["fusion_head_config"] = dict(config)
    return backbone, fusion_head, metadata


def resolve_inference_method(requested, fusion_head):
    valid = {"residual", "knn", "fusion", "auto"}
    if requested not in valid:
        raise ValueError(f"unknown inference method {requested!r}; expected one of {sorted(valid)}")
    if requested == "fusion" and fusion_head is None:
        raise RuntimeError(
            "--method fusion requires a phase-3 checkpoint containing "
            "teacher_fusion_head; retrain/export the order-aware deployment bundle")
    if requested == "auto":
        if fusion_head is not None:
            return "fusion"
        warnings.warn(
            "checkpoint has no fusion head; --method auto falls back to residual-min",
            RuntimeWarning,
            stacklevel=2,
        )
        return "residual"
    return requested


def resolve_inference_tile(requested, method, bundle_metadata):
    """Use the trained token-context size unless the caller explicitly overrides it."""
    if requested is not None:
        return int(requested)
    if method == "fusion":
        config = bundle_metadata.get("fusion_head_config")
        if not isinstance(config, dict) or "train_tile" not in config:
            raise ValueError("fusion deployment metadata is missing train_tile")
        return int(config["train_tile"])
    return 128


def resolve_context_halo(requested, method, tile, patch=16, bundle_metadata=None):
    """Default to central stitching for fusion while keeping legacy scores unchanged."""
    if requested is not None:
        return int(requested)
    if method != "fusion":
        return 0
    config = (bundle_metadata or {}).get("fusion_head_config", {})
    if "inference_context_halo" in config:
        return int(config["inference_context_halo"])
    # Two patch tokens (32px for ViT-B/16) on every side. For unusually small
    # tiles, fall back to one quarter of the tile, aligned to the patch grid.
    max_halo = ((int(tile) - patch) // (2 * patch)) * patch
    return max(0, min(2 * patch, max_halo))


def _norm(img, mean=_MEAN, std=_STD):
    x = (img.astype(np.float32) - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1)


def knn_scores(ft, f1, f2, g, window=1, k=1):
    """Windowed cross-die KNN scoring, SYMMETRIC in direction.

    Two anomaly directions, matched to the presence semantics:
      extra   (1,0,0): target patch finds no match in either ref's window
      missing (0,1,1): a ref patch (agreeing refs) finds no match in the
                       target's window -- "target lacks what refs have" is
                       also a defect; single-ref nuisance stays suppressed
    score = elementwise max of the two directions.

    - The window keeps positional semantics: an UNRESTRICTED bank search would
      let a nuisance blob elsewhere on the ref 'explain' a defect blob (same
      morphology) and suppress it. Cross-die + near-position is the only legal
      match source for this task.
    - window tolerates residual misalignment / pattern phase jitter that strict
      per-position residual cannot.
    - window=0, k=1 reduces EXACTLY to residual_mode='min' (both directions
      coincide at w=0 since sim is symmetric per position).

    ft/f1/f2: (T, N, D) L2-normalized patch tokens, N = g*g.
    Returns (score, extra, missing), each (T, N) L2-equivalent distances.
    """
    if window < 0 or k < 1:
        raise ValueError("knn window must be >= 0 and k must be >= 1")
    N = ft.shape[1]
    pos = torch.arange(N, device=ft.device)
    ry, rx = pos // g, pos % g
    allow = (((ry[:, None] - ry[None, :]).abs() <= window)
             & ((rx[:, None] - rx[None, :]).abs() <= window))       # (N, N)
    k_eff = min(k, (window + 1) ** 2, N)  # corner windows have (w+1)^2 candidates

    def _dir(a, b):   # best window match in b for each patch of a -> sim (T, N)
        s = torch.bmm(a, b.transpose(1, 2)).masked_fill(~allow, -2.0)
        return s.topk(k_eff, dim=2).values.mean(dim=2)

    to_d = lambda s: torch.sqrt((2.0 - 2.0 * s).clamp(min=0.0))
    # EXTRA direction, catches (1,0,0): a target patch is fine if SOME ref
    # explains it -> max over refs.
    extra = to_d(torch.maximum(_dir(ft, f1), _dir(ft, f2)))
    # MISSING direction, catches (0,1,1): flag only if NEITHER ref patch can be
    # explained by the target (max over refs of the best sim) -- a single-ref
    # nuisance still finds its match through the other ref, stays suppressed.
    missing = to_d(torch.maximum(_dir(f1, ft), _dir(f2, ft)))
    return torch.maximum(extra, missing), extra, missing


@torch.no_grad()
def score_map(model, path, device, tile=128, residual_mode="min",
              register=False, patch=16, chunk=16,
              method="residual", knn_window=1, knn_k=1,
              fusion_head=None, channel_order=(0, 1, 2),
              uint16_black_level=0.0, uint16_white_level=65535.0,
              normalization_mean=IMG_MEAN, normalization_std=IMG_STD,
              context_halo=0, uint16_decode_mode="float_linear",
              expected_dtype=None):
    """Return valid-grid scores/diagnostics, original size, and target image."""
    requested_method = method
    method = resolve_inference_method(method, fusion_head)
    if requested_method == "auto" and method == "residual":
        residual_mode = "min"
    if patch <= 0:
        raise ValueError("patch must be positive")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if tile and (tile < patch or tile % patch):
        raise ValueError("tile must be 0 or a positive multiple of patch size")
    if context_halo < 0 or context_halo % patch:
        raise ValueError("context_halo must be a non-negative multiple of patch size")
    if method == "fusion":
        fc = getattr(fusion_head, "checkpoint_config", None)
        if fc is not None and int(fc["patch_size"]) != patch:
            raise ValueError(
                f"fusion checkpoint patch_size={fc['patch_size']} but inference patch={patch}")
        if fc is not None and int(fc["train_tile"]) != int(tile):
            raise ValueError(
                f"fusion checkpoint was trained with tile={fc['train_tile']}; "
                f"inference tile={tile} changes token context and score calibration")
    tgt, r1, r2 = read_tiff3(
        path, channel_order=tuple(channel_order),
        uint16_black_level=uint16_black_level,
        uint16_white_level=uint16_white_level,
        uint16_decode_mode=uint16_decode_mode,
        expected_dtype=expected_dtype)
    norm_mean = np.asarray(normalization_mean, np.float32).reshape(1, 1, 3)
    norm_std = np.asarray(normalization_std, np.float32).reshape(1, 1, 3)
    if not np.isfinite(norm_mean).all() or not np.isfinite(norm_std).all() \
            or np.any(norm_std <= 0):
        raise ValueError("normalization mean/std must be finite and std must be positive")
    if register:
        r1, _ = coarse_register(tgt, r1, patch / 2.0)
        r2, _ = coarse_register(tgt, r2, patch / 2.0)
    H, W = tgt.shape[:2]
    ts = tile if tile and tile > 0 else ((max(H, W) + patch - 1) // patch) * patch
    halo = int(context_halo)
    if 2 * halo >= ts:
        raise ValueError("context_halo must leave at least one central output patch")
    core = ts - 2 * halo
    core_h = ((H + core - 1) // core) * core
    core_w = ((W + core - 1) // core) * core
    Hp, Wp = core_h + 2 * halo, core_w + 2 * halo
    def pad(im):
        return cv2.copyMakeBorder(
            im, halo, Hp - H - halo, halo, Wp - W - halo,
            cv2.BORDER_REPLICATE)
    tgt_p, r1_p, r2_p = pad(tgt), pad(r1), pad(r2)

    coords = [(y, x) for y in range(0, core_h, core)
              for x in range(0, core_w, core)]
    g = ts // patch
    halo_g, core_g = halo // patch, core // patch
    scores = np.zeros((core_h // patch, core_w // patch), np.float32)
    aux = {k: np.zeros_like(scores) for k in ("d_t1", "d_t2", "d_12")}

    for i in range(0, len(coords), chunk):
        cs = coords[i:i + chunk]
        stack = []
        for img in (tgt_p, r1_p, r2_p):
            stack += [_norm(img[y:y + ts, x:x + ts], norm_mean, norm_std)
                      for (y, x) in cs]
        x_in = torch.stack(stack).to(device)          # (3*T, 3, ts, ts)
        tok = F.normalize(
            model.forward_features(x_in)["x_norm_patchtokens"], dim=-1)
        T = len(cs)
        ft, f1, f2 = tok[:T], tok[T:2 * T], tok[2 * T:]
        # Pairwise distances for per-detection presence diagnostics.
        pair = dict(d_t1=torch.norm(ft - f1, dim=-1),
                    d_t2=torch.norm(ft - f2, dim=-1),
                    d_12=torch.norm(f1 - f2, dim=-1))
        if method == "fusion":
            logits = fusion_head(ft, f1, f2)
            expected = (T, g * g)
            if logits.shape != expected:
                raise RuntimeError(
                    f"fusion head returned {tuple(logits.shape)}, expected {expected}")
            if not torch.isfinite(logits).all():
                raise RuntimeError("fusion head returned NaN/Inf logits")
            res = logits.sigmoid()
            pair["fusion_logit"] = logits
            pair["fusion_prob"] = res
        elif method == "knn":
            res, extra, missing = knn_scores(ft, f1, f2, g,
                                             window=knn_window, k=knn_k)
            pair["d_extra"] = extra
            pair["d_missing"] = missing
        else:
            res = _residual(ft, f1, f2, residual_mode)    # (T, g*g)
        for j, (y, x) in enumerate(cs):
            sl = (slice(y // patch, y // patch + core_g),
                  slice(x // patch, x // patch + core_g))
            local = (slice(halo_g, halo_g + core_g),
                     slice(halo_g, halo_g + core_g))
            scores[sl] = res[j].reshape(g, g)[local].cpu().numpy()
            for k_, m in pair.items():
                aux.setdefault(k_, np.zeros_like(scores))[sl] = \
                    m[j].reshape(g, g)[local].cpu().numpy()
    # Padding exists only to complete the final patch/tile.  It must never enter
    # calibration or detection (480px with patch16 is 30x30, not padded 32x32).
    valid_h = (H + patch - 1) // patch
    valid_w = (W + patch - 1) // patch
    scores = scores[:valid_h, :valid_w]
    aux = {k: v[:valid_h, :valid_w] for k, v in aux.items()}
    return scores, aux, (H, W), tgt


def detections_from(scores, threshold, aux=None, patch=16, image_hw=None):
    """Threshold -> connected components -> pixel-space boxes.

    With `aux` (pairwise distance maps) each detection carries presence
    diagnostics. A detection already has BOTH d_t1 and d_t2 high (min-logic),
    so the remaining discriminator is the ref-ref distance:
      d_12 low  -> refs agree with each other, target is the odd one out
                   => 'defect'; with knn the direction adds the subtype:
                   extra_1_0_0 (target has a blob refs lack) or
                   missing_0_1_1 (refs share a blob target lacks)
      d_12 high -> the refs disagree between themselves (one-ref nuisance at
                   the boundary, registration issue, dirty ref)
                   => 'refs_disagree', review before trusting
    """
    mask = (scores > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        sel = labels[y:y + h, x:x + w] == i
        comp = scores[y:y + h, x:x + w][sel]
        px, py = int(x * patch), int(y * patch)
        pw, ph = int(w * patch), int(h * patch)
        if image_hw is not None:
            ih, iw = image_hw
            pw, ph = min(pw, max(iw - px, 0)), min(ph, max(ih - py, 0))
        d = dict(x=px, y=py, w=pw, h=ph,
                 score=float(comp.max()), n_patches=int(sel.sum()))
        if aux is not None:
            for k_, m in aux.items():
                d[k_] = float(m[y:y + h, x:x + w][sel].mean())
            if "fusion_prob" in d:
                # The trained head already implements the directional 100 truth
                # table; do not reuse an anomaly threshold as a d_12 threshold.
                d["presence"] = "target_unique_1_0_0"
            elif d["d_12"] < threshold:    # legacy residual/KNN diagnostic
                d["presence"] = "defect"
                if "d_extra" in d:         # knn method: direction tells the kind
                    d["subtype"] = ("extra_1_0_0" if d["d_extra"] >= d["d_missing"]
                                    else "missing_0_1_1")
            else:
                d["presence"] = "refs_disagree"
        dets.append(d)
    return sorted(dets, key=lambda d: -d["score"])


def save_heatmap(tgt, scores, HW, out_path, patch=16, dets=None):
    H, W = HW
    r = scores - scores.min()
    r = (255 * r / (r.max() + 1e-8)).astype(np.uint8)
    heat = cv2.applyColorMap(
        cv2.resize(r, (scores.shape[1] * patch, scores.shape[0] * patch),
                   interpolation=cv2.INTER_NEAREST)[:H, :W], cv2.COLORMAP_JET)
    base = cv2.cvtColor(
        np.clip(tgt[:H, :W], 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    for d in dets or []:
        cv2.rectangle(overlay, (d["x"], d["y"]),
                      (d["x"] + d["w"], d["y"] + d["h"]), (255, 255, 255), 1)
    cv2.imwrite(str(out_path), overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="phaseN_teacher.pth")
    ap.add_argument("--input", required=True, help="a .tif/.tiff file or a directory")
    ap.add_argument("--out-dir", default="triplet_ssl/runs/infer")
    ap.add_argument("--tile", type=int, default=None,
                    help="tile size (= training crop); omitted = checkpoint value "
                         "for fusion, otherwise 128. 0 = whole image in one pass")
    ap.add_argument("--chunk", type=int, default=1,
                    help="tiles per backbone call; 1 is the low-VRAM safe default")
    ap.add_argument("--context-halo", type=int, default=None,
                    help="pixels of context discarded on each tile edge; omitted = "
                         "32 for fusion central stitching, 0 for legacy methods")
    ap.add_argument("--residual-mode", default="min", choices=["min", "mean"])
    ap.add_argument("--method", default="auto",
                    choices=["residual", "knn", "fusion", "auto"],
                    help="residual = per-position diff vs refs; knn = windowed "
                         "cross-die k-NN; fusion = trained target-only head; "
                         "auto = fusion when available, else residual-min")
    ap.add_argument("--knn-window", type=int, default=1,
                    help="spatial search radius in patches (0 == residual min)")
    ap.add_argument("--knn-k", type=int, default=1)
    ap.add_argument("--register", action="store_true",
                    help="sub-pixel register refs onto target (match training setting)")
    ap.add_argument("--allow-preprocess-mismatch", action="store_true",
                    help="explicitly allow a train/infer registration mismatch")
    ap.add_argument("--threshold", type=float, default=None,
                    help="patch-score threshold for detections (from --calib)")
    ap.add_argument("--calib", action="store_true",
                    help="treat input as NORMAL-ONLY tiffs; print score percentiles "
                         "to pick a threshold. No GT involved.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, fusion_head, bundle_meta = load_inference_bundle(args.checkpoint, device)
    method = resolve_inference_method(args.method, fusion_head)
    args.tile = resolve_inference_tile(args.tile, method, bundle_meta)
    args.context_halo = resolve_context_halo(
        args.context_halo, method, args.tile, patch=16,
        bundle_metadata=bundle_meta)
    if args.method == "auto" and method == "residual":
        args.residual_mode = "min"
    if fusion_head is not None and method != "fusion":
        warnings.warn(
            f"loaded fusion head is being bypassed by explicit --method {method}; "
            "this legacy score is not directional target-only detection",
            RuntimeWarning,
        )
    print(f"[method] requested={args.method} effective={method} "
          f"tile={args.tile} halo={args.context_halo} "
          f"fusion_head={'yes' if fusion_head is not None else 'no'}")
    preprocess = bundle_meta.get("preprocess") or {}
    source_order = tuple(preprocess.get("source_channel_indices", (0, 1, 2)))
    uint16_black = float(preprocess.get("uint16_black_level", 0.0))
    uint16_white = float(preprocess.get("uint16_white_level", 65535.0))
    uint16_decode_mode = preprocess.get("uint16_decode_mode", "float_linear")
    source_dtype = preprocess.get("source_dtype")
    training_source_layout = preprocess.get("source_layout")
    norm_mean = tuple(preprocess.get("mean", IMG_MEAN))
    norm_std = tuple(preprocess.get("std", IMG_STD))
    trained_register = bool(preprocess.get("register", args.register))
    if preprocess and trained_register != bool(args.register):
        message = (f"registration mismatch: fusion checkpoint trained with register="
                   f"{trained_register}, inference requested register={bool(args.register)}")
        if not args.allow_preprocess_mismatch:
            raise RuntimeError(message + "; pass --allow-preprocess-mismatch to override")
        warnings.warn(message, RuntimeWarning)

    p = Path(args.input)
    files = sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff"))) if p.is_dir() else [p]
    if not files:
        sys.exit(f"no .tif/.tiff under {p}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    sm_kw = dict(tile=args.tile, chunk=args.chunk, residual_mode=args.residual_mode,
                 register=args.register, method=method,
                 knn_window=args.knn_window, knn_k=args.knn_k,
                 fusion_head=fusion_head, channel_order=source_order,
                 uint16_black_level=uint16_black,
                 uint16_white_level=uint16_white,
                 normalization_mean=norm_mean, normalization_std=norm_std,
                 context_halo=args.context_halo,
                 uint16_decode_mode=uint16_decode_mode,
                 expected_dtype=source_dtype)
    if method == "knn":
        tag_m = f"knn w={args.knn_window} k={args.knn_k}"
    elif method == "fusion":
        tag_m = "order-aware target-unique fusion sigmoid score"
    else:
        tag_m = f"residual {args.residual_mode}"

    if args.calib:
        alls = []
        for f in files:
            s, _, _, _ = score_map(model, f, device, **sm_kw)
            alls.append(s.ravel())
        a = np.concatenate(alls)
        print(f"[calib] {len(files)} normal tiffs, {a.size} patches ({tag_m})")
        percentiles = {f"p{q:g}": float(np.percentile(a, q))
                       for q in (99.0, 99.9, 99.99)}
        for q, value in percentiles.items():
            print(f"  {q:<7}: {value:.4f}")
        print(f"  max   : {a.max():.4f}   <- threshold just above this "
              f"= zero false positives on this set")
        calibration = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_version": bundle_meta.get("checkpoint_version"),
            "checkpoint_global_step": bundle_meta.get("global_step"),
            "checkpoint_validation": bundle_meta.get("validation"),
            "effective_method": method,
            "score_semantics": ("sigmoid(target_unique_logit)" if method == "fusion"
                                else tag_m),
            "normal_only_required": True,
            "n_files": len(files),
            "n_patches": int(a.size),
            "normal_score_percentiles": percentiles,
            "normal_score_max": float(a.max()),
            "tile": args.tile,
            "context_halo": args.context_halo,
            "patch": 16,
            "register": bool(args.register),
            "source_channel_indices": list(source_order),
            "uint16_black_level": uint16_black,
            "uint16_white_level": uint16_white,
            "uint16_decode_mode": uint16_decode_mode,
            "source_dtype": source_dtype,
            "training_source_layout": training_source_layout,
            "normalization_mean": list(norm_mean),
            "normalization_std": list(norm_std),
        }
        (out / "calibration.json").write_text(json.dumps(calibration, indent=2))
        print(f"[calib] saved -> {out / 'calibration.json'}")
        return

    all_dets = {}
    for f in files:
        s, aux, HW, tgt = score_map(model, f, device, **sm_kw)
        dets = (detections_from(s, args.threshold, aux, image_hw=HW)
                if args.threshold is not None else [])
        np.save(out / f"{f.stem}_scores.npy", s)
        save_heatmap(tgt, s, HW, out / f"{f.stem}_heatmap.png", dets=dets)
        all_dets[f.stem] = dets
        tag = f"{len(dets)} detections" if args.threshold is not None else "map only"
        print(f"[infer] {f.name}: {tag}, score max={s.max():.4f}")
    (out / "detections.json").write_text(json.dumps(all_dets, indent=2))
    (out / "inference_metadata.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_version": bundle_meta.get("checkpoint_version"),
        "checkpoint_global_step": bundle_meta.get("global_step"),
        "checkpoint_validation": bundle_meta.get("validation"),
        "requested_method": args.method,
        "effective_method": method,
        "threshold": args.threshold,
        "tile": args.tile,
        "context_halo": args.context_halo,
        "register": bool(args.register),
        "source_channel_indices": list(source_order),
        "uint16_black_level": uint16_black,
        "uint16_white_level": uint16_white,
        "uint16_decode_mode": uint16_decode_mode,
        "source_dtype": source_dtype,
        "training_source_layout": training_source_layout,
        "normalization_mean": list(norm_mean),
        "normalization_std": list(norm_std),
        "score_semantics": ("sigmoid(target_unique_logit)" if method == "fusion"
                            else tag_m),
    }, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()

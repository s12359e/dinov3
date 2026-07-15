"""Inference: triplet TIFF -> per-patch anomaly map + defect detections.

Recipe (must MATCH training conditions):
  read tiff (ch1=target, ch2=ref1, ch3=ref2) -> [optional registration, same
  setting as training] -> same IMG_MEAN/STD normalization -> TEACHER backbone
  (eval, no grad) at native resolution in 128px tiles -> L2-normalized patch
  tokens -> residual = min/mean over the two refs -> score map (x16 upsample)
  -> [threshold, calibrated on NORMAL tiffs only] -> connected components ->
  defect boxes.

Calibrate a threshold on a directory of known-normal TIFFs:
    python triplet_ssl/infer.py --checkpoint runs/exp/phase2_teacher.pth \
        --input data/normal_tiffs --calib

Run detection:
    python triplet_ssl/infer.py --checkpoint runs/exp/phase2_teacher.pth \
        --input path/to/tiffs --out-dir infer_out --threshold 0.31 \
        [--register] [--residual-mode min] [--device cuda]

Outputs per TIFF: <stem>_heatmap.png (overlay), <stem>_scores.npy (patch grid),
plus a single detections.json.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.models.vision_transformer import vit_base
from triplet_ssl import IMG_MEAN, IMG_STD
from triplet_ssl.data.triplet_dataset import read_tiff3, coarse_register
from triplet_ssl.eval.separability import _residual

_MEAN = np.array(IMG_MEAN, np.float32).reshape(1, 1, 3)
_STD = np.array(IMG_STD, np.float32).reshape(1, 1, 3)


def load_backbone(ckpt_path, device):
    m = vit_base(patch_size=16, img_size=224)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("teacher_backbone", ckpt.get("model", ckpt))
    state = {k.replace("backbone.", ""): v for k, v in state.items()}
    msg = m.load_state_dict(state, strict=False)
    print(f"[load] {ckpt_path} missing={len(msg.missing_keys)} "
          f"unexpected={len(msg.unexpected_keys)}")
    return m.to(device).eval()


def _norm(img):
    x = (img.astype(np.float32) - _MEAN) / _STD
    return torch.from_numpy(x).permute(2, 0, 1)


@torch.no_grad()
def score_map(model, path, device, tile=128, residual_mode="min",
              register=False, patch=16, chunk=16):
    """Returns (scores (Hp/16, Wp/16) float32, (H, W) original size, target u8)."""
    tgt, r1, r2 = read_tiff3(path)
    if register:
        r1, _ = coarse_register(tgt, r1, patch / 2.0)
        r2, _ = coarse_register(tgt, r2, patch / 2.0)
    H, W = tgt.shape[:2]
    ts = tile if tile and tile > 0 else ((max(H, W) + patch - 1) // patch) * patch
    Hp = ((H + ts - 1) // ts) * ts
    Wp = ((W + ts - 1) // ts) * ts
    pad = lambda im: cv2.copyMakeBorder(im, 0, Hp - H, 0, Wp - W, cv2.BORDER_REPLICATE)
    tgt_p, r1_p, r2_p = pad(tgt), pad(r1), pad(r2)

    coords = [(y, x) for y in range(0, Hp, ts) for x in range(0, Wp, ts)]
    g = ts // patch
    scores = np.zeros((Hp // patch, Wp // patch), np.float32)

    for i in range(0, len(coords), chunk):
        cs = coords[i:i + chunk]
        stack = []
        for img in (tgt_p, r1_p, r2_p):
            stack += [_norm(img[y:y + ts, x:x + ts]) for (y, x) in cs]
        x_in = torch.stack(stack).to(device)          # (3*T, 3, ts, ts)
        tok = F.normalize(
            model.forward_features(x_in)["x_norm_patchtokens"], dim=-1)
        T = len(cs)
        ft, f1, f2 = tok[:T], tok[T:2 * T], tok[2 * T:]
        res = _residual(ft, f1, f2, residual_mode)    # (T, g*g)
        for j, (y, x) in enumerate(cs):
            scores[y // patch:y // patch + g,
                   x // patch:x // patch + g] = res[j].reshape(g, g).cpu().numpy()
    return scores, (H, W), tgt


def detections_from(scores, threshold, patch=16):
    """Threshold -> connected components -> pixel-space boxes."""
    mask = (scores > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        comp = scores[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i]
        dets.append(dict(x=int(x * patch), y=int(y * patch),
                         w=int(w * patch), h=int(h * patch),
                         score=float(comp.max()), n_patches=int((labels == i).sum())))
    return sorted(dets, key=lambda d: -d["score"])


def save_heatmap(tgt, scores, HW, out_path, patch=16, dets=None):
    H, W = HW
    r = scores - scores.min()
    r = (255 * r / (r.max() + 1e-8)).astype(np.uint8)
    heat = cv2.applyColorMap(
        cv2.resize(r, (scores.shape[1] * patch, scores.shape[0] * patch),
                   interpolation=cv2.INTER_NEAREST)[:H, :W], cv2.COLORMAP_JET)
    base = cv2.cvtColor(tgt[:H, :W], cv2.COLOR_RGB2BGR)
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
    ap.add_argument("--tile", type=int, default=128,
                    help="tile size (= training crop). 0 = whole image in one pass")
    ap.add_argument("--residual-mode", default="min", choices=["min", "mean"])
    ap.add_argument("--register", action="store_true",
                    help="sub-pixel register refs onto target (match training setting)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="patch-score threshold for detections (from --calib)")
    ap.add_argument("--calib", action="store_true",
                    help="treat input as NORMAL-ONLY tiffs; print score percentiles "
                         "to pick a threshold. No GT involved.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_backbone(args.checkpoint, device)

    p = Path(args.input)
    files = sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff"))) if p.is_dir() else [p]
    if not files:
        sys.exit(f"no .tif/.tiff under {p}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    if args.calib:
        alls = []
        for f in files:
            s, _, _ = score_map(model, f, device, args.tile,
                                args.residual_mode, args.register)
            alls.append(s.ravel())
        a = np.concatenate(alls)
        print(f"[calib] {len(files)} normal tiffs, {a.size} patches "
              f"(mode={args.residual_mode})")
        for q in (99.0, 99.9, 99.99):
            print(f"  p{q:<6}: {np.percentile(a, q):.4f}")
        print(f"  max   : {a.max():.4f}   <- threshold just above this "
              f"= zero false positives on this set")
        return

    all_dets = {}
    for f in files:
        s, HW, tgt = score_map(model, f, device, args.tile,
                               args.residual_mode, args.register)
        dets = detections_from(s, args.threshold) if args.threshold is not None else []
        np.save(out / f"{f.stem}_scores.npy", s)
        save_heatmap(tgt, s, HW, out / f"{f.stem}_heatmap.png", dets=dets)
        all_dets[f.stem] = dets
        tag = f"{len(dets)} detections" if args.threshold is not None else "map only"
        print(f"[infer] {f.name}: {tag}, score max={s.max():.4f}")
    (out / "detections.json").write_text(json.dumps(all_dets, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()

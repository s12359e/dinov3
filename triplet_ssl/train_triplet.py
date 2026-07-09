"""Triplet continued-SSL training + per-phase separability gate.

Phases (config `phase`):
  0 - no training; evaluate frozen DINOv3 (baseline AUROC, the bar to beat).
  1 - triplet DINO (pairing table + patch loss + shared-geometry views).
  2 - + defect oversampling + robust top-k exemption.
  3 - + synthetic-defect repulsion.

Stock DINO stabilisation is kept: EMA teacher, centering, temperature, cosine LR + warmup.
Backbone LR is small (continued pretraining) with layer-wise decay; heads train normally.

Init is ALWAYS from a DINOv3 checkpoint (mimic weights until real ones are placed);
`--checkpoint` swaps in real weights with no code change.

GT masks are read only by `triplet_ssl.eval.separability` (never in the training path).
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.models.vision_transformer import vit_base
from dinov3.layers.dino_head import DINOHead
from triplet_ssl.data.triplet_dataset import (
    TripletDataset, SharedGeomTripletAug, DefectOversampleBatchSampler, triplet_collate)
from triplet_ssl.data.synth_defect import SyntheticDefect, augmentation_survival_check
from triplet_ssl.losses.triplet_loss import TripletLoss
from triplet_ssl.eval.separability import evaluate_separability, evaluate_guardrail
from triplet_ssl.eval.plot_curves import plot_training_curves
from triplet_ssl import IMG_MEAN, IMG_STD

NAMES = ("ref1", "ref2", "target")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        with open(path.parent / cfg["base"]) as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "base"})
    return cfg


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_backbone(cfg, device):
    return vit_base(patch_size=cfg["model"]["patch_size"],
                    img_size=cfg["model"]["img_size"]).to(device)


def load_checkpoint(backbone, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt
    for k in ("model", "teacher", "teacher_backbone", "state_dict"):
        if isinstance(state, dict) and k in state:
            state = state[k]
            break
    state = {k.replace("backbone.", ""): v for k, v in state.items()}
    msg = backbone.load_state_dict(state, strict=False)
    print(f"[init] loaded {ckpt_path} (mimic={ckpt.get('mimic', False) if isinstance(ckpt, dict) else False}) "
          f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")


def build_head(cfg, out_dim):
    m = cfg["model"]
    return DINOHead(in_dim=768, out_dim=out_dim, hidden_dim=m["head_hidden"],
                    bottleneck_dim=m["head_bottleneck"], nlayers=m["head_nlayers"])


def embed(backbone, cls_head, patch_head, x):
    feat = backbone.forward_features(x)
    return cls_head(feat["x_norm_clstoken"]), patch_head(feat["x_norm_patchtokens"])


# --------------------------------------------------------------------------- #
# Layer-wise LR decay (small backbone LR, normal head LR)
# --------------------------------------------------------------------------- #
def param_groups_layerwise(backbone, heads, backbone_lr, head_lr, decay, wd, n_blocks):
    groups = []

    def block_id(name):
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1          # blocks are layers 1..n
        if any(name.startswith(p) for p in ("patch_embed", "cls_token", "storage_tokens",
                                            "mask_token", "rope_embed")):
            return 0                                     # stem = layer 0
        return n_blocks + 1                              # final norm = top

    for name, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        scale = decay ** (n_blocks + 1 - block_id(name))
        groups.append({"params": [p], "lr": backbone_lr * scale,
                       "weight_decay": 0.0 if p.ndim == 1 else wd})
    for h in heads:
        for p in h.parameters():
            groups.append({"params": [p], "lr": head_lr,
                           "weight_decay": 0.0 if p.ndim == 1 else wd})
    return groups


@torch.no_grad()
def ema_update(student, teacher, m):
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(m).add_(sp.data, alpha=1 - m)


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="override cfg checkpoint (real weights)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default="triplet_ssl/runs")
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.checkpoint: cfg["checkpoint"] = args.checkpoint
    if args.data_root: cfg["data"]["root"] = args.data_root
    if args.num_steps is not None: cfg["optim"]["num_steps"] = args.num_steps
    if args.batch_size is not None: cfg["optim"]["batch_size"] = args.batch_size

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["seed"])
    phase = cfg["phase"]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Triplet SSL — Phase {phase} | device={device} ===")

    # -- Backbones (student + EMA teacher), init from checkpoint --------------
    student_bb = build_backbone(cfg, device)
    teacher_bb = build_backbone(cfg, device)
    load_checkpoint(student_bb, cfg["checkpoint"])
    teacher_bb.load_state_dict(student_bb.state_dict())
    teacher_bb.requires_grad_(False)

    # -- Heads (CLS + patch), student + EMA teacher --------------------------
    s_cls = build_head(cfg, cfg["model"]["cls_out_dim"]).to(device)
    s_patch = build_head(cfg, cfg["model"]["patch_out_dim"]).to(device)
    s_cls.init_weights(); s_patch.init_weights()
    t_cls = copy.deepcopy(s_cls); t_patch = copy.deepcopy(s_patch)
    for h in (t_cls, t_patch): h.requires_grad_(False)

    img_size, patch_size = cfg["model"]["img_size"], cfg["model"]["patch_size"]

    # ---- Phase 0: no training, evaluate frozen teacher ----------------------
    if phase == 0:
        _evaluate_and_report(teacher_bb, cfg, device, out_dir, phase,
                             notes="frozen DINOv3 baseline")
        return

    # -- Loss ----------------------------------------------------------------
    lc = cfg["loss"]
    triplet_loss = TripletLoss(
        cls_out_dim=cfg["model"]["cls_out_dim"], patch_out_dim=cfg["model"]["patch_out_dim"],
        student_temp=lc["student_temp"], teacher_temp=lc["teacher_temp"],
        center_momentum=lc["center_momentum"],
        topk_pct=lc["topk_exempt"]["pct"] if lc["topk_exempt"]["enable"] else 0.0,
        weights=lc["weights"], cls_weight=lc["cls_weight"],
        repel_lambda=cfg["synth_defect"]["lambda"] if cfg["synth_defect"]["enable"] else 0.0,
    ).to(device)

    # -- Data ----------------------------------------------------------------
    dc = cfg["data"]
    aug = SharedGeomTripletAug(img_size=img_size, crop_scale=tuple(dc["crop_scale"]),
                               flip_prob=dc["flip_prob"], brightness_delta=dc["brightness_delta"],
                               contrast_range=tuple(dc["contrast_range"]),
                               gamma_range=tuple(dc["gamma_range"]), blur_prob=dc["blur_prob"])
    sd = cfg["synth_defect"]
    synth = SyntheticDefect(prob=sd["prob"], types=tuple(sd["types"]),
                            psf_sigma=tuple(sd.get("psf_sigma", (1.0, 1.7))),
                            psf_amplitude=tuple(sd.get("psf_amplitude", (15, 80))),
                            seed=cfg["seed"]) if sd["enable"] else None
    train_ds = TripletDataset(dc["root"], dc["train_split"], transform=aug,
                              register=dc["register"], patch_size=patch_size,
                              load_masks=False, synth_defect=synth)
    train_ds.assert_no_masks()
    print(f"[data] {len(train_ds)} triplets ({len(train_ds.defect_idx)} defective)")
    if synth is not None:
        augmentation_survival_check(train_ds, n=min(100, len(train_ds)))

    oc = cfg["oversample"]
    bs = cfg["optim"]["batch_size"]
    sampler = DefectOversampleBatchSampler(
        train_ds, batch_size=bs,
        defect_frac=oc["defect_frac"] if oc["enable"] else None,
        num_batches=cfg["optim"]["num_steps"], seed=cfg["seed"])
    loader = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler,
                                         num_workers=0, collate_fn=triplet_collate)

    # -- Optimizer -----------------------------------------------------------
    groups = param_groups_layerwise(student_bb, [s_cls, s_patch],
                                    cfg["optim"]["backbone_lr"], cfg["optim"]["head_lr"],
                                    cfg["optim"]["layerwise_decay"], cfg["optim"]["weight_decay"],
                                    n_blocks=len(student_bb.blocks))
    for g in groups: g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))

    # -- Train loop ----------------------------------------------------------
    total = cfg["optim"]["num_steps"]; warmup = cfg["optim"]["warmup_steps"]
    mom0 = cfg["optim"]["momentum_teacher"]
    mom_final = cfg["optim"].get("momentum_teacher_final", 1.0)
    clip = cfg["optim"]["grad_clip"]
    # Teacher-temperature warmup (standard DINO: linear start->end, then constant).
    tt_start = lc["teacher_temp"]
    tt_end = lc.get("teacher_temp_end", tt_start)
    tt_warm = max(1, int(lc.get("teacher_temp_warmup_frac", 0.3) * total))
    student_bb.train(); s_cls.train(); s_patch.train()
    train_log = []
    t0 = time.time()
    for step, batch in enumerate(loader):
        f = lr_factor(step, total, warmup)
        for g in opt.param_groups: g["lr"] = g["base_lr"] * f
        triplet_loss.teacher_temp = tt_start + (tt_end - tt_start) * min(step / tt_warm, 1.0)
        # Cosine EMA momentum ramp mom0 -> mom_final (stock DINO recipe).
        mom = mom_final - (mom_final - mom0) * (math.cos(math.pi * step / max(total, 1)) + 1) / 2

        s_views = {n: batch[n][:, 0].to(device) for n in NAMES}
        t_views = {n: batch[n][:, 1].to(device) for n in NAMES}
        cls, patch = {}, {}
        for n in NAMES:
            cs, ps = embed(student_bb, s_cls, s_patch, s_views[n])
            with torch.no_grad():
                ct, pt = embed(teacher_bb, t_cls, t_patch, t_views[n])
            cls[n] = dict(s=cs, t=ct.detach())
            patch[n] = dict(s=ps, t=pt.detach())

        synth_mask = batch["synth_mask"].to(device) if "synth_mask" in batch else None
        loss, logs = triplet_loss(cls, patch, trad_key=NAMES[step % 3], synth_mask=synth_mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(student_bb.parameters()) + list(s_cls.parameters()) + list(s_patch.parameters()), clip)
        opt.step()
        ema_update(student_bb, teacher_bb, mom)
        ema_update(s_cls, t_cls, mom); ema_update(s_patch, t_patch, mom)

        row = dict(step=step, loss=float(loss.detach()),
                   L_refref=logs["L_refref"], L_target2ref=logs["L_target2ref"],
                   L_trad=logs["L_trad"], lr=opt.param_groups[-1]["lr"],
                   teacher_temp=triplet_loss.teacher_temp, momentum=mom)
        if "repel" in logs:
            row["repel"] = logs["repel"]
        train_log.append(row)

        if step % max(1, total // 10) == 0 or step == total - 1:
            extra = f" repel={logs['repel']:.3f}" if "repel" in logs else ""
            print(f"step {step:04d}/{total} loss={loss.item():.4f} "
                  f"rr={logs['L_refref']:.3f} tr={logs['L_target2ref']:.3f} "
                  f"td={logs['L_trad']:.3f} lr={opt.param_groups[-1]['lr']:.2e}{extra}")

    print(f"[train] {total} steps in {time.time() - t0:.1f}s")

    # -- Training curves ------------------------------------------------------
    (out_dir / f"phase{phase}_train_log.json").write_text(json.dumps(train_log))
    plot_training_curves(train_log, out_dir / f"phase{phase}_curves.png")
    print(f"[curves] -> {out_dir / f'phase{phase}_curves.png'}")

    # -- Exempted-patch sanity overlays (phase>=2) ----------------------------
    if lc["topk_exempt"]["enable"]:
        _log_exempt_overlays(triplet_loss, student_bb, s_cls, s_patch,
                             teacher_bb, t_cls, t_patch, train_ds, device,
                             out_dir, phase)

    torch.save({"teacher_backbone": teacher_bb.state_dict(), "phase": phase},
               out_dir / f"phase{phase}_teacher.pth")
    _evaluate_and_report(teacher_bb, cfg, device, out_dir, phase,
                         notes=f"phase{phase} adapted")


@torch.no_grad()
def _log_exempt_overlays(loss_mod, student_bb, s_cls, s_patch, teacher_bb, t_cls,
                         t_patch, train_ds, device, out_dir, phase, n_samples=4):
    """Render which patches the robust top-k would exempt, overlaid on the target
    view, for a few known-defective samples. Sanity signal: exempted patches should
    visibly correlate with defect regions. Not a training input."""
    idxs = train_ds.defect_idx[:n_samples]
    if not idxs:
        print("[exempt] no known-defective triplets to visualise")
        return
    batch = triplet_collate([train_ds[i] for i in idxs])
    s_views = {n: batch[n][:, 0].to(device) for n in NAMES}
    t_views = {n: batch[n][:, 1].to(device) for n in NAMES}
    patch = {}
    for n in NAMES:
        _, ps_ = embed(student_bb, s_cls, s_patch, s_views[n])
        _, pt_ = embed(teacher_bb, t_cls, t_patch, t_views[n])
        patch[n] = dict(s=ps_, t=pt_)

    pp = student_bb.patch_size
    g = s_views["target"].shape[-1] // pp
    mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
    std = torch.tensor(IMG_STD).view(3, 1, 1)
    for bi, di in enumerate(idxs):
        exempt = loss_mod.exempted_patch_map(patch, sample_idx=bi).cpu().numpy()
        img = (s_views["target"][bi].cpu() * std + mean).clamp(0, 255)
        img = img.byte().permute(1, 2, 0).numpy()
        overlay = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        for flat in np.flatnonzero(exempt):
            r, c = divmod(int(flat), g)
            cv2.rectangle(overlay, (c * pp, r * pp), ((c + 1) * pp - 1, (r + 1) * pp - 1),
                          (0, 0, 255), 1)
        tid = train_ds.items[di]["id"]
        cv2.imwrite(str(out_dir / f"phase{phase}_exempt_{tid}.png"), overlay)
        print(f"[exempt] {tid}: {int(exempt.sum())} patches exempted "
              f"-> phase{phase}_exempt_{tid}.png")


def _evaluate_and_report(backbone, cfg, device, out_dir, phase, notes=""):
    dc, ec, mc = cfg["data"], cfg["eval"], cfg["model"]
    metrics = evaluate_separability(
        backbone, dc["root"], dc["eval_split"], device,
        img_size=mc["img_size"], patch_size=mc["patch_size"],
        overlap_thresh=ec["overlap_thresh"], n_heatmaps=ec["n_heatmaps"],
        out_dir=str(out_dir / f"phase{phase}_heatmaps"), tag=f"phase{phase}")
    guard = evaluate_guardrail(backbone, dc["root"], dc["eval_split"], device,
                               img_size=mc["img_size"])
    metrics.update(guard)

    baseline_file = out_dir / "phase0_auroc.txt"
    have_baseline = baseline_file.exists()
    if phase == 0:
        baseline_file.write_text(str(metrics["auroc"]))
        delta = 0.0
    elif have_baseline:
        base = float(baseline_file.read_text())
        delta = metrics["auroc"] - base
        gate = cfg["separability_gate"]["min_auroc_delta"]
        metrics["auroc_delta_vs_phase0"] = delta
        metrics["gate_pass"] = bool(delta >= gate)
    else:
        delta = None  # phase 0 not run in this out-dir

    (out_dir / f"phase{phase}_metrics.json").write_text(json.dumps(metrics, indent=2))
    summary = out_dir / "summary.tsv"
    if not summary.exists():
        summary.write_text("phase\tauroc\tdefect_normal_ratio\tguardrail_mean_res\tnotes\n")
    with open(summary, "a") as f:
        f.write(f"{phase}\t{metrics['auroc']:.4f}\t{metrics['defect_normal_ratio']:.3f}\t"
                f"{metrics['guardrail_mean_res']:.4f}\t{notes}\n")

    if phase == 0:
        gate_str = "  (baseline)"
    elif delta is None:
        gate_str = "  (no phase-0 baseline in this --out-dir; run phase0 first)"
    else:
        gate_str = f"  (delta vs phase0: {delta:+.4f}, gate={'PASS' if metrics['gate_pass'] else 'FAIL'})"
    print("---")
    print(f"phase:                 {phase}")
    print(f"AUROC:                 {metrics['auroc']:.4f}" + gate_str)
    print(f"defect/normal ratio:   {metrics['defect_normal_ratio']:.3f}")
    print(f"guardrail mean/p99:    {metrics['guardrail_mean_res']:.4f} / {metrics['guardrail_p99_res']:.4f}")
    print(f"heatmaps -> {out_dir / f'phase{phase}_heatmaps'}")


if __name__ == "__main__":
    main()

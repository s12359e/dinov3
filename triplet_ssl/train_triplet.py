"""Triplet continued-SSL training + per-phase separability gate.

Phases (config `phase`):
  0 - no training; evaluate frozen DINOv3 (baseline AUROC, the bar to beat).
  1 - triplet DINO (pairing table + patch loss + shared-geometry views).
  2 - + defect oversampling + robust top-k exemption.
  3 - + bounded synthetic-defect margin and order-aware target-only fusion head.

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
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.layers.dino_head import DINOHead
from triplet_ssl.data.triplet_dataset import (
    TripletDataset, TiffTripletDataset, SharedGeomTripletAug,
    DefectOversampleBatchSampler, triplet_collate)
from triplet_ssl.data.synth_defect import TripletSyntheticPSF, augmentation_survival_check
from triplet_ssl.losses.triplet_loss import TripletLoss
from triplet_ssl.models.order_aware_fusion import (
    OrderAwareTripletFusionHead, select_safe_background_negatives,
    target_unique_fusion_loss)
from triplet_ssl.models.backbone import (
    build_canonical_backbone, export_backbone_config, validate_backbone_config)
from triplet_ssl.eval.separability import evaluate_separability, evaluate_guardrail
from triplet_ssl.eval.plot_curves import plot_training_curves
from triplet_ssl import IMG_MEAN, IMG_STD

NAMES = ("ref1", "ref2", "target")


# --------------------------------------------------------------------------- #
# Distributed (torchrun --nproc_per_node=N)
# --------------------------------------------------------------------------- #
def setup_distributed():
    """Init from torchrun env. Returns local_rank, or None for single-process.

    nccl on CUDA (H200s), gloo on CPU. DDP_INIT_FILE escape hatch: Windows
    torch builds lack libuv so torchrun's TCPStore fails -- set DDP_INIT_FILE
    to a shared temp path and launch the ranks manually to smoke-test DDP."""
    if "RANK" not in os.environ:
        return None
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_file = os.environ.get("DDP_INIT_FILE")
    if init_file:
        dist.init_process_group(backend=backend,
                                init_method=f"file:///{Path(init_file).as_posix()}",
                                rank=int(os.environ["RANK"]),
                                world_size=int(os.environ["WORLD_SIZE"]))
    else:
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


class TripletStudent(nn.Module):
    """Everything the student computes per step behind ONE forward().

    DDP registers its gradient reducer on the wrapped module's forward; calling
    submodules (backbone/heads) directly around a DDP wrapper breaks gradient
    bucketing. This container is what gets DDP-wrapped."""

    def __init__(self, backbone, cls_head, patch_head, fusion_head=None):
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head
        self.patch_head = patch_head
        self.fusion_head = fusion_head

    def _append_fusion(self, outputs, patch_tokens):
        if self.fusion_head is None:
            return outputs
        if patch_tokens.shape[0] % 3:
            raise ValueError("stacked triplet batch must contain exactly 3*B images")
        # s_nat is stacked in NAMES order: ref1, ref2, target.
        f1, f2, ft = patch_tokens.chunk(3, dim=0)
        return outputs + (self.fusion_head(ft, f1, f2),)

    def forward(self, s_nat, s_loc=None):
        if s_loc is not None:
            f_nat, f_loc = self.backbone.forward_features([s_nat, s_loc],
                                                          masks=[None, None])
            tok = f_nat["x_norm_patchtokens"]
            outputs = (self.cls_head(f_loc["x_norm_clstoken"]),
                       self.patch_head(tok), tok)
            return self._append_fusion(outputs, tok)
        f = self.backbone.forward_features(s_nat)
        tok = f["x_norm_patchtokens"]
        outputs = (self.cls_head(f["x_norm_clstoken"]),
                   self.patch_head(tok), tok)
        return self._append_fusion(outputs, tok)


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
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        with open(path.parent / cfg["base"], encoding="utf-8") as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "base"})
    return cfg


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_backbone(cfg, device):
    if int(cfg["model"]["patch_size"]) != 16:
        raise ValueError("canonical dinov3_vitb16 requires model.patch_size=16")
    return build_canonical_backbone().to(device)


def _extract_backbone_state(ckpt):
    """Unwrap common DINO/DDP checkpoints into a backbone-only state dict."""
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


def load_checkpoint(backbone, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        validate_backbone_config(ckpt.get("backbone_config"))
    state = _extract_backbone_state(ckpt)
    msg = backbone.load_state_dict(state, strict=False)
    if msg.missing_keys:
        preview = msg.missing_keys[:8]
        suffix = "..." if len(msg.missing_keys) > len(preview) else ""
        raise ValueError(
            "checkpoint does not fully initialize the training backbone; "
            f"missing keys: {preview}{suffix}")
    if msg.unexpected_keys:
        preview = msg.unexpected_keys[:8]
        suffix = "..." if len(msg.unexpected_keys) > len(preview) else ""
        raise ValueError(
            "checkpoint contains weights outside the canonical training backbone; "
            f"unexpected keys: {preview}{suffix}")
    print(f"[init] loaded {ckpt_path} "
          f"(mimic={ckpt.get('mimic', False) if isinstance(ckpt, dict) else False}) "
          f"missing=0 unexpected={len(msg.unexpected_keys)}")


def build_head(cfg, out_dim):
    m = cfg["model"]
    return DINOHead(in_dim=768, out_dim=out_dim, hidden_dim=m["head_hidden"],
                    bottleneck_dim=m["head_bottleneck"], nlayers=m["head_nlayers"])


def build_fusion_head(cfg):
    fc = cfg.get("fusion_head", {})
    if not fc.get("enable", False):
        return None
    return OrderAwareTripletFusionHead(
        in_dim=768,
        hidden_dim=int(fc.get("hidden_dim", 128)),
        dropout=float(fc.get("dropout", 0.1)),
        prior_prob=float(fc.get("prior_prob", 0.01)),
    )


def embed(backbone, cls_head, patch_head, x):
    feat = backbone.forward_features(x)
    return (cls_head(feat["x_norm_clstoken"]),
            patch_head(feat["x_norm_patchtokens"]),
            feat["x_norm_patchtokens"])


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

    local_rank = setup_distributed()
    ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1
    is_main = rank == 0
    if ddp:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Per-rank seeds: each rank must draw DIFFERENT crops/batches.
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    phase = cfg["phase"]
    fusion_enabled = bool(cfg.get("fusion_head", {}).get("enable", False))
    if fusion_enabled and not cfg["synth_defect"].get("enable", False):
        raise ValueError("fusion_head.enable requires synth_defect.enable truth-table supervision")
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Triplet SSL — Phase {phase} | device={device} | "
              f"world={world} ===")

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

    # The deployed fusion head follows the same EMA student->teacher path as the
    # backbone, so checkpointed tokens and relation head stay distribution-aligned.
    s_fusion = build_fusion_head(cfg)
    if s_fusion is not None:
        s_fusion = s_fusion.to(device)
        t_fusion = copy.deepcopy(s_fusion).requires_grad_(False).eval()
    else:
        t_fusion = None

    # Student container behind one forward() (required for DDP), then wrap.
    student = TripletStudent(student_bb, s_cls, s_patch, s_fusion).to(device)
    if ddp:
        student = nn.parallel.DistributedDataParallel(
            student, device_ids=[local_rank] if torch.cuda.is_available() else None)
    core = student.module if ddp else student
    # DDP broadcasts the wrapped student from rank 0.  Re-copy the EMA heads
    # afterwards; per-rank augmentation seeds were intentionally different and
    # would otherwise leave unwrapped teacher heads with different initial state.
    t_cls.load_state_dict(core.cls_head.state_dict())
    t_patch.load_state_dict(core.patch_head.state_dict())
    if t_fusion is not None:
        t_fusion.load_state_dict(core.fusion_head.state_dict())

    img_size, patch_size = cfg["model"]["img_size"], cfg["model"]["patch_size"]

    # ---- Phase 0: no training, evaluate frozen teacher ----------------------
    if phase == 0:
        if is_main:
            _evaluate_and_report(teacher_bb, cfg, device, out_dir, phase,
                                 notes="frozen DINOv3 baseline")
        if ddp:
            dist.barrier(); dist.destroy_process_group()
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
        repel_margin=cfg["synth_defect"].get("margin", 0.5),
    ).to(device)

    # -- Data ----------------------------------------------------------------
    dc = cfg["data"]
    aug = SharedGeomTripletAug(img_size=img_size, crop_scale=tuple(dc["crop_scale"]),
                               flip_prob=dc["flip_prob"], brightness_delta=dc["brightness_delta"],
                               contrast_range=tuple(dc["contrast_range"]),
                               gamma_range=tuple(dc["gamma_range"]), blur_prob=dc["blur_prob"])
    sd = cfg["synth_defect"]
    synth = TripletSyntheticPSF(
        n_events=tuple(sd.get("n_events", (3, 8))),
        defect_prob=sd.get("defect_prob", 0.5),
        missing_frac=sd.get("missing_frac", 0.0),
        psf_sigma=tuple(sd.get("psf_sigma", (1.1, 1.7))),
        psf_amplitude=tuple(sd.get("psf_amplitude", (15, 80))),
        defect_center_jitter=sd.get("defect_center_jitter"),
        seed=cfg["seed"] + rank) if sd["enable"] else None
    if dc.get("format", "folder") == "tiff3":
        train_ds = TiffTripletDataset(dc["root"], crop_size=dc.get("crop_size", 128),
                                      transform=aug, register=dc["register"],
                                      patch_size=patch_size, synth_defect=synth,
                                      channel_order=tuple(dc.get("channel_order", (0, 1, 2))),
                                      cls_local_size=dc.get("cls_local_size", 64),
                                      uint16_black_level=dc.get("uint16_black_level", 0),
                                      uint16_white_level=dc.get("uint16_white_level", 65535))
        print(f"[data] tiff3: {len(train_ds)} TIFFs, native {dc.get('crop_size', 128)}px "
              f"window crop (no resize)")
    else:
        train_ds = TripletDataset(dc["root"], dc["train_split"], transform=aug,
                                  register=dc["register"], patch_size=patch_size,
                                  load_masks=False, synth_defect=synth)
        print(f"[data] {len(train_ds)} triplets ({len(train_ds.defect_idx)} defective)")
    train_ds.assert_no_masks()
    if synth is not None and is_main:
        augmentation_survival_check(train_ds, n=min(100, len(train_ds)))

    oc = cfg["oversample"]
    bs = cfg["optim"]["batch_size"]        # per-GPU; global = bs * world
    if is_main and world > 1:
        print(f"[dist] world={world}  per-GPU batch={bs}  global batch={bs * world}")
    sampler = DefectOversampleBatchSampler(
        train_ds, batch_size=bs,
        defect_frac=oc["defect_frac"] if oc["enable"] else None,
        num_batches=cfg["optim"]["num_steps"], seed=cfg["seed"] + rank)
    nw = dc.get("num_workers", 0)
    loader = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler,
                                         num_workers=nw, collate_fn=triplet_collate,
                                         pin_memory=(device.type == "cuda"),
                                         persistent_workers=(nw > 0))

    # -- Optimizer -----------------------------------------------------------
    train_heads = [s_cls, s_patch] + ([s_fusion] if s_fusion is not None else [])
    groups = param_groups_layerwise(student_bb, train_heads,
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
    amp_on = bool(cfg["optim"].get("amp", False)) and device.type == "cuda"
    student.train()
    train_log = []
    t0 = time.time()
    for step, batch in enumerate(loader):
        f = lr_factor(step, total, warmup)
        for g in opt.param_groups: g["lr"] = g["base_lr"] * f
        triplet_loss.teacher_temp = tt_start + (tt_end - tt_start) * min(step / tt_warm, 1.0)
        # Cosine EMA momentum ramp mom0 -> mom_final (stock DINO recipe).
        mom = mom_final - (mom_final - mom0) * (math.cos(math.pi * step / max(total, 1)) + 1) / 2

        # Batched forwards (GPU-friendly): the three same-size views are stacked
        # into one (3B, ...) pass -- 9 forwards/step become 3. ViT uses LayerNorm
        # (per-sample stats), so this is numerically identical to per-view calls.
        # iBOT-style split with DINO's asymmetry direction: patch loss on the
        # aligned native views; STUDENT's CLS side comes from the small local
        # window (local-to-global, with grad) while the TEACHER only ever sees
        # the full native crop (stable, complete targets).
        B = batch["ref1"].shape[0]
        has_cls_views = "target_cls" in batch
        s_nat = torch.cat([batch[n][:, 0] for n in NAMES]).to(device, non_blocking=True)
        t_nat = torch.cat([batch[n][:, 1] for n in NAMES]).to(device, non_blocking=True)

        s_loc = (torch.cat([batch[n + "_cls"] for n in NAMES]).to(device, non_blocking=True)
                 if has_cls_views else None)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            # One student forward through the DDP-wrapped container (both
            # resolutions via forward_features_list; heads applied selectively).
            student_out = student(s_nat, s_loc)
            if fusion_enabled:
                cs_all, ps_nat, fs_nat, fusion_logits = student_out
            else:
                cs_all, ps_nat, fs_nat = student_out
                fusion_logits = None
            with torch.no_grad():
                ct_nat, pt_nat, ft_nat = embed(teacher_bb, t_cls, t_patch, t_nat)

            cls, patch = {}, {}
            for i, n in enumerate(NAMES):
                sl = slice(i * B, (i + 1) * B)
                cls[n] = dict(s=cs_all[sl], t=ct_nat[sl].detach())
                patch[n] = dict(s=ps_nat[sl], t=pt_nat[sl].detach(),
                                s_feat=fs_nat[sl], t_feat=ft_nat[sl].detach())

            synth_mask = batch["synth_mask"].to(device) if "synth_mask" in batch else None
            synth_event_mask = (batch["synth_event_mask"].to(device)
                                if "synth_event_mask" in batch else None)
            synth_unmatched_mask = (batch["synth_unmatched_mask"].to(device)
                                    if "synth_unmatched_mask" in batch else synth_mask)
            # TR-CLS exemption: target KNOWN to contain a defect (image-level flag
            # or synthetic injection) -> drop that sample's target<->ref CLS pull.
            cls_exempt = batch["is_defect"].to(device).bool()
            if synth_unmatched_mask is not None:
                cls_exempt = cls_exempt | (synth_unmatched_mask.sum(dim=1) > 0)
            loss, logs = triplet_loss(cls, patch, trad_key=NAMES[step % 3],
                                      synth_mask=synth_mask, cls_exempt=cls_exempt,
                                      pull_exclude_mask=synth_unmatched_mask)
            if fusion_enabled:
                if synth_mask is None or synth_event_mask is None:
                    raise RuntimeError(
                        "fusion training requires synth_mask and synth_event_mask from the dataset")
                fc = cfg["fusion_head"]
                background_neg = select_safe_background_negatives(
                    patch["target"]["s_feat"], patch["ref1"]["s_feat"],
                    patch["ref2"]["s_feat"], synth_event_mask,
                    fraction=float(fc.get("background_negative_frac", 0.25)))
                fusion_raw, fusion_logs = target_unique_fusion_loss(
                    fusion_logits, synth_mask, synth_event_mask, background_neg)
                fwarm = int(fc.get("warmup_steps", 0))
                warm_scale = min((step + 1) / max(fwarm, 1), 1.0) if fwarm > 0 else 1.0
                fusion_weight = float(fc.get("loss_weight", 0.5)) * warm_scale
                fusion_term = fusion_weight * fusion_raw
                loss = loss + fusion_term
                logs.update(fusion_logs)
                logs["fusion_raw"] = float(fusion_raw.detach())
                logs["fusion_loss"] = float(fusion_term.detach())
                logs["fusion_weight"] = fusion_weight

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_params = (list(student_bb.parameters()) + list(s_cls.parameters())
                       + list(s_patch.parameters()))
        if s_fusion is not None:
            clip_params += list(s_fusion.parameters())
        nn.utils.clip_grad_norm_(clip_params, clip)
        opt.step()
        ema_update(student_bb, teacher_bb, mom)
        ema_update(s_cls, t_cls, mom); ema_update(s_patch, t_patch, mom)
        if s_fusion is not None:
            # The fusion head starts random (unlike the pretrained backbone), so
            # the backbone's near-1 EMA would leave a short run mostly random.
            fusion_mom = float(cfg["fusion_head"].get("ema_momentum", 0.9))
            ema_update(s_fusion, t_fusion, fusion_mom)

        row = dict(step=step, loss=float(loss.detach()),
                   L_refref=logs["L_refref"], L_target2ref=logs["L_target2ref"],
                   L_trad=logs["L_trad"], lr=opt.param_groups[-1]["lr"],
                   teacher_temp=triplet_loss.teacher_temp, momentum=mom)
        if "repel" in logs:
            row["repel"] = logs["repel"]
            row["synth_score"] = logs["synth_score"]
        if "fusion_loss" in logs:
            row.update(fusion_loss=logs["fusion_loss"],
                       fusion_raw=logs["fusion_raw"],
                       fusion_weight=logs["fusion_weight"],
                       fusion_pos_score=logs["fusion_pos_score"],
                       fusion_neg_score=logs["fusion_neg_score"])
        train_log.append(row)

        if is_main and (step % max(1, total // 10) == 0 or step == total - 1):
            extra = (f" margin={logs['repel']:.3f} synth_score={logs['synth_score']:.3f}"
                     if "repel" in logs else "")
            if "fusion_loss" in logs:
                extra += (f" fusion={logs['fusion_loss']:.3f}"
                          f" p+={logs['fusion_pos_score']:.3f}"
                          f" p-={logs['fusion_neg_score']:.3f}")
            print(f"step {step:04d}/{total} loss={loss.item():.4f} "
                  f"rr={logs['L_refref']:.3f} tr={logs['L_target2ref']:.3f} "
                  f"td={logs['L_trad']:.3f} lr={opt.param_groups[-1]['lr']:.2e}{extra}")

    if is_main:
        print(f"[train] {total} steps in {time.time() - t0:.1f}s")

        # -- Training curves --------------------------------------------------
        (out_dir / f"phase{phase}_train_log.json").write_text(json.dumps(train_log))
        plot_training_curves(train_log, out_dir / f"phase{phase}_curves.png")
        print(f"[curves] -> {out_dir / f'phase{phase}_curves.png'}")

        # -- Exempted-patch sanity overlays (phase>=2) ------------------------
        if lc["topk_exempt"]["enable"]:
            _log_exempt_overlays(triplet_loss, core.backbone, core.cls_head,
                                 core.patch_head, teacher_bb, t_cls, t_patch,
                                 train_ds, device, out_dir, phase)

        checkpoint = {
            "checkpoint_version": 2 if t_fusion is not None else 1,
            "backbone_config": export_backbone_config(),
            "teacher_backbone": teacher_bb.state_dict(),
            "phase": phase,
            "preprocess": {
                "mean": list(IMG_MEAN),
                "std": list(IMG_STD),
                "input_scaling": "fixed_uint16_range_to_0_255_float_v1",
                "uint16_black_level": float(dc.get("uint16_black_level", 0)),
                "uint16_white_level": float(dc.get("uint16_white_level", 65535)),
                "channel_order": ["target", "ref1", "ref2"],
                "source_channel_indices": list(dc.get("channel_order", (0, 1, 2))),
                "register": bool(dc.get("register", False)),
            },
        }
        if t_fusion is not None:
            train_tile = int(dc.get("crop_size", img_size))
            checkpoint.update(
                teacher_fusion_head=t_fusion.state_dict(),
                student_fusion_head=s_fusion.state_dict(),
                fusion_head_config=t_fusion.export_config(
                    patch_size=patch_size, train_tile=train_tile),
            )
        torch.save(checkpoint, out_dir / f"phase{phase}_teacher.pth")
        _evaluate_and_report(teacher_bb, cfg, device, out_dir, phase,
                             notes=f"phase{phase} adapted",
                             fusion_head=t_fusion)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


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
        _, ps_, fs_ = embed(student_bb, s_cls, s_patch, s_views[n])
        _, pt_, ft_ = embed(teacher_bb, t_cls, t_patch, t_views[n])
        patch[n] = dict(s=ps_, t=pt_, s_feat=fs_, t_feat=ft_)

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


def _evaluate_and_report(backbone, cfg, device, out_dir, phase, notes="",
                         fusion_head=None):
    dc, ec, mc = cfg["data"], cfg["eval"], cfg["model"]
    eval_root = dc.get("eval_root") or dc["root"]
    tiff_mode = dc.get("format", "folder") == "tiff3"
    proxy_eval = tiff_mode and bool(dc.get("eval_root"))
    if tiff_mode and not proxy_eval:
        print("[eval] skipped: unlabelled TIFF data has no AUROC target. Use "
              "infer.py --calib on held-out known-normal optical TIFFs.")
        return
    if proxy_eval and not ec.get("allow_resized_folder_proxy", False):
        print("[eval] skipped: a 224px resized folder eval does not match the native "
              "TIFF tile context. Set eval.allow_resized_folder_proxy=true only "
              "for an explicitly non-deployment proxy; use infer.py --calib for "
              "the production path.")
        return
    if proxy_eval:
        print("[eval] PROXY ONLY: resized folder/whole-image tokens do not match "
              "native TIFF tiled deployment; AUROC will not be used as a gate.")
    rmode = ec.get("residual_mode", "mean")
    metrics = evaluate_separability(
        backbone, eval_root, dc["eval_split"], device,
        img_size=mc["img_size"], patch_size=mc["patch_size"],
        overlap_thresh=ec["overlap_thresh"], n_heatmaps=ec["n_heatmaps"],
        out_dir=str(out_dir / f"phase{phase}_heatmaps"), tag=f"phase{phase}",
        residual_mode=rmode, fusion_head=fusion_head)
    guard = evaluate_guardrail(backbone, eval_root, dc["eval_split"], device,
                               img_size=mc["img_size"], residual_mode=rmode,
                               fusion_head=fusion_head)
    metrics.update(guard)

    baseline_file = out_dir / "phase0_auroc.txt"
    have_baseline = baseline_file.exists()
    metrics["deployment_gate_eligible"] = not proxy_eval
    if proxy_eval:
        delta = None
    elif phase == 0:
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

    if proxy_eval:
        gate_str = "  (proxy only; not a deployment gate)"
    elif phase == 0:
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

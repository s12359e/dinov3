"""
DINOv3 SSL autoresearch training script. Single-GPU, single-file.
The AI agent modifies this file to find the best SSL hyperparameters.

Usage: python train.py --config config.yaml
       python train.py  (uses defaults)

The metric is: best_anomaly_score (higher is better).
"""

import argparse
import gc
import math
import os
import sys
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent dir to path for dinov3 imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dinov3.models.vision_transformer import DinoVisionTransformer
from prepare import TIME_BUDGET, IMG_SIZE, PATCH_SIZE, evaluate_ssl_knn

# ---------------------------------------------------------------------------
# Hyperparameters (the AI agent tunes these)
# ---------------------------------------------------------------------------

# ViT Architecture
EMBED_DIM = 384           # embedding dimension (384=ViT-S, 768=ViT-B)
DEPTH = 12                # number of transformer blocks
NUM_HEADS = 6             # attention heads (EMBED_DIM must be divisible)
PATCH_SIZE_CFG = 16       # patch size (14 or 16)
DROP_PATH_RATE = 0.1      # stochastic depth

# SSL Training
LR = 1e-4                 # base learning rate
WEIGHT_DECAY = 0.04       # weight decay
WARMUP_RATIO = 0.1        # fraction of time budget for warmup
BATCH_SIZE = 32           # images per step
MOMENTUM_TEACHER = 0.996  # EMA momentum for teacher
TEACHER_TEMP = 0.04       # teacher softmax temperature
STUDENT_TEMP = 0.1        # student softmax temperature
PROJ_DIM = 256            # projection head output dimension
PROJ_HIDDEN = 2048        # projection head hidden dimension

# Data Augmentation
CROP_SCALE_MIN = 0.4      # random crop min scale
CROP_SCALE_MAX = 1.0      # random crop max scale
FLIP_PROB = 0.5           # horizontal flip probability
BRIGHTNESS_DELTA = 20     # brightness jitter
CONTRAST_RANGE = (0.8, 1.2)

# Eval
EVAL_DIR = ""             # set via --eval-dir or config

# ---------------------------------------------------------------------------
# DINO SSL Components
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """Projection head for DINO: MLP -> L2-norm -> prototypes."""

    def __init__(self, in_dim, hidden_dim=2048, out_dim=256, nlayers=3):
        super().__init__()
        layers = []
        for i in range(nlayers - 1):
            d_in = in_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(d_in, hidden_dim),
                nn.GELU(),
            ])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(out_dim, out_dim, bias=False)
        self.last_layer.weight.data.copy_(
            torch.eye(out_dim) + 0.01 * torch.randn(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    """Cross-entropy loss between student and teacher outputs with centering."""

    def __init__(self, out_dim, center_momentum=0.9):
        super().__init__()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        # Teacher: sharpen with centering
        teacher_probs = F.softmax(
            (teacher_out - self.center) / TEACHER_TEMP, dim=-1
        ).detach()

        # Student
        student_log_probs = F.log_softmax(
            student_out / STUDENT_TEMP, dim=-1
        )

        # Cross entropy
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()

        # Update center
        with torch.no_grad():
            batch_center = teacher_out.mean(dim=0, keepdim=True)
            self.center = (self.center * self.center_momentum
                          + batch_center * (1 - self.center_momentum))

        return loss


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

class SSLAugmentation:
    """Two-crop augmentation for DINO SSL."""

    def __init__(self, img_size=512, mean=(109.65, 104.81, 75.48),
                 std=(54.32, 39.78, 36.47)):
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def _random_crop(self, image, scale_min, scale_max):
        h, w = image.shape[:2]
        scale = np.random.uniform(scale_min, scale_max)
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        top = np.random.randint(0, max(h - crop_h, 1) + 1)
        left = np.random.randint(0, max(w - crop_w, 1) + 1)
        crop = image[top:top + crop_h, left:left + crop_w]
        return cv2.resize(crop, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_LINEAR)

    def _augment(self, image, is_global=True):
        """Apply augmentation to one view."""
        if is_global:
            img = self._random_crop(image, CROP_SCALE_MIN, CROP_SCALE_MAX)
        else:
            img = self._random_crop(image, 0.05, CROP_SCALE_MIN)

        if np.random.random() < FLIP_PROB:
            img = np.ascontiguousarray(np.fliplr(img))

        img_f = img.astype(np.float32)
        if np.random.random() < 0.5:
            img_f += np.random.uniform(-BRIGHTNESS_DELTA, BRIGHTNESS_DELTA)
        if np.random.random() < 0.5:
            img_f *= np.random.uniform(*CONTRAST_RANGE)

        img = np.clip(img_f, 0, 255).astype(np.uint8)
        img = (img.astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __call__(self, image):
        """Returns two global views."""
        view1 = self._augment(image, is_global=True)
        view2 = self._augment(image, is_global=True)
        return view1, view2


import cv2
from torch.utils.data import Dataset, DataLoader


class SSLImageDataset(Dataset):
    """Loads images for SSL training (no labels needed)."""

    def __init__(self, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')
        ])
        if not self.image_files:
            raise FileNotFoundError(f"No images in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_files[idx]), cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to a black image
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            v1, v2 = self.transform(img)
            return v1, v2
        return img


def ssl_collate(batch):
    """Collate two-view batches."""
    v1s, v2s = zip(*batch)
    return torch.stack(v1s), torch.stack(v2s)


# ---------------------------------------------------------------------------
# EMA Teacher
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(student, teacher, momentum):
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_lr(progress):
    """Warmup -> constant -> cosine cooldown."""
    if progress < WARMUP_RATIO:
        return progress / max(WARMUP_RATIO, 1e-8)
    cooldown_start = 0.7
    if progress < cooldown_start:
        return 1.0
    frac = (progress - cooldown_start) / (1.0 - cooldown_start)
    return 0.5 * (1.0 + math.cos(math.pi * frac))


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load config YAML and override globals."""
    if not config_path or not os.path.exists(config_path):
        return
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        return

    g = globals()
    for key, val in cfg.items():
        key_upper = key.upper()
        if key_upper in g:
            g[key_upper] = val
            print(f"  Config override: {key_upper} = {val}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="YAML config file to override hyperparameters")
    parser.add_argument("--eval-dir", type=str, default="",
                        help="Directory with eval images (_x{}_y{} filenames)")
    parser.add_argument("--train-dir", type=str, default="",
                        help="Directory with training images")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Pretrained ViT checkpoint to initialize from")
    parser.add_argument("--output-dir", type=str, default="autoresearch_ssl/runs")
    parser.add_argument("--eval-size", type=int, default=None,
                        help="Resize eval images to this resolution (auto-aligned to patch_size multiple). "
                             "Default: use IMG_SIZE (512). Example: --eval-size 745")
    args = parser.parse_args()

    # Load config overrides
    if args.config:
        print(f"Loading config: {args.config}")
        load_config(args.config)

    global EVAL_DIR
    if args.eval_dir:
        EVAL_DIR = args.eval_dir

    if not args.train_dir:
        print("ERROR: --train-dir is required")
        sys.exit(1)
    if not EVAL_DIR:
        print("ERROR: --eval-dir is required")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print(f"Device: {device}")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Architecture: embed_dim={EMBED_DIM}, depth={DEPTH}, "
          f"heads={NUM_HEADS}, patch={PATCH_SIZE_CFG}")
    print(f"SSL: lr={LR}, wd={WEIGHT_DECAY}, bs={BATCH_SIZE}, "
          f"mom_teacher={MOMENTUM_TEACHER}")

    # ── Build student + teacher ViT ──────────────────────────────────────
    student_backbone = DinoVisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE_CFG,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        drop_path_rate=DROP_PATH_RATE,
    ).to(device)

    teacher_backbone = DinoVisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE_CFG,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        drop_path_rate=0.0,  # no drop path for teacher
    ).to(device)

    # Load pretrained weights if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt
        for key in ("model", "teacher", "state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        msg = student_backbone.load_state_dict(state, strict=False)
        print(f"  missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    # Copy student -> teacher (EMA starts from student)
    teacher_backbone.load_state_dict(student_backbone.state_dict())
    teacher_backbone.requires_grad_(False)

    # Projection heads
    student_head = DINOHead(EMBED_DIM, PROJ_HIDDEN, PROJ_DIM).to(device)
    teacher_head = DINOHead(EMBED_DIM, PROJ_HIDDEN, PROJ_DIM).to(device)
    teacher_head.load_state_dict(student_head.state_dict())
    teacher_head.requires_grad_(False)

    # Loss
    dino_loss = DINOLoss(PROJ_DIM).to(device)

    total_params = sum(p.numel() for p in student_backbone.parameters())
    trainable = sum(p.numel() for p in student_backbone.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in student_head.parameters())
    print(f"Parameters: backbone={total_params:,} head={head_params:,} "
          f"trainable={trainable + head_params:,}")

    # ── Optimizer ────────────────────────────────────────────────────────
    param_groups = [
        {"params": student_backbone.parameters(), "lr": LR},
        {"params": student_head.parameters(), "lr": LR * 10},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # ── Data ─────────────────────────────────────────────────────────────
    augmentation = SSLAugmentation(img_size=IMG_SIZE)
    train_ds = SSLImageDataset(args.train_dir, transform=augmentation)
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        drop_last=True,
        collate_fn=ssl_collate,
    )

    # ── Training loop (time-budgeted) ────────────────────────────────────
    print(f"\nStarting SSL training (budget={TIME_BUDGET}s)...")
    student_backbone.train()
    student_head.train()
    teacher_backbone.eval()
    teacher_head.eval()

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    t_start = time.time()
    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0
    train_iter = iter(train_loader)

    while True:
        t0 = time.time()

        # Get batch
        try:
            view1, view2 = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            view1, view2 = next(train_iter)

        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        # Forward student
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            # Student: both views
            s1_feat = student_backbone(view1)
            s2_feat = student_backbone(view2)
            if isinstance(s1_feat, dict):
                s1_feat = s1_feat.get("x_norm_clstoken", s1_feat.get("x_clstoken"))
                s2_feat = s2_feat.get("x_norm_clstoken", s2_feat.get("x_clstoken"))

            s1_proj = student_head(s1_feat)
            s2_proj = student_head(s2_feat)

            # Teacher: both views (no grad)
            with torch.no_grad():
                t1_feat = teacher_backbone(view1)
                t2_feat = teacher_backbone(view2)
                if isinstance(t1_feat, dict):
                    t1_feat = t1_feat.get("x_norm_clstoken", t1_feat.get("x_clstoken"))
                    t2_feat = t2_feat.get("x_norm_clstoken", t2_feat.get("x_clstoken"))
                t1_proj = teacher_head(t1_feat)
                t2_proj = teacher_head(t2_feat)

            # Symmetric loss: student1 vs teacher2 + student2 vs teacher1
            loss = (dino_loss(s1_proj, t2_proj) + dino_loss(s2_proj, t1_proj)) / 2

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(student_backbone.parameters()) + list(student_head.parameters()),
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student_backbone.parameters()) + list(student_head.parameters()),
                1.0,
            )
            optimizer.step()

        # EMA update teacher
        with torch.no_grad():
            # Momentum schedule: ramp from 0.996 to 1.0
            progress = min(total_training_time / TIME_BUDGET, 1.0)
            mom = MOMENTUM_TEACHER + (1.0 - MOMENTUM_TEACHER) * progress
            update_ema(student_backbone, teacher_backbone, mom)
            update_ema(student_head, teacher_head, mom)

        # LR schedule
        lr_mult = get_lr(progress)
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", LR) * lr_mult
        # Store initial LR on first step
        if step == 0:
            for pg in optimizer.param_groups:
                pg["initial_lr"] = pg["lr"] / max(lr_mult, 1e-8)

        # Timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 5:
            total_training_time += dt

        # Logging
        loss_val = loss.item()
        if math.isnan(loss_val) or loss_val > 100:
            print("\nFAIL: NaN or exploding loss")
            sys.exit(1)

        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
        debiased = smooth_loss / (1 - ema_beta ** (step + 1))
        pct = 100 * progress
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | "
              f"lr: {lr_mult * LR:.2e} | dt: {dt*1000:.0f}ms | "
              f"remaining: {remaining:.0f}s    ", end="", flush=True)

        if step == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        step += 1

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r

    # ── Evaluation ───────────────────────────────────────────────────────
    eval_size_str = f" (eval_size={args.eval_size})" if args.eval_size else ""
    print(f"\nEvaluating teacher backbone with defect KNN...{eval_size_str}")
    best_score, results = evaluate_ssl_knn(
        teacher_backbone, EVAL_DIR, device, eval_size=args.eval_size
    )

    # ── Save best checkpoint ─────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "last_teacher.pth"
    torch.save({
        "teacher_backbone": teacher_backbone.state_dict(),
        "teacher_head": teacher_head.state_dict(),
        "student_backbone": student_backbone.state_dict(),
        "student_head": student_head.state_dict(),
        "step": step,
        "best_anomaly_score": best_score,
    }, ckpt_path)

    # ── Summary ──────────────────────────────────────────────────────────
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print("---")
    print(f"best_anomaly_score: {best_score:.6f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"num_steps:          {step}")
    print(f"embed_dim:          {EMBED_DIM}")
    print(f"depth:              {DEPTH}")
    print(f"num_heads:          {NUM_HEADS}")
    print(f"lr:                 {LR}")
    print(f"batch_size:         {BATCH_SIZE}")
    print(f"momentum_teacher:   {MOMENTUM_TEACHER}")

    # Per-K results
    for k_key, val in sorted(results.items()):
        print(f"{k_key}: {val:.6f}")


if __name__ == "__main__":
    main()

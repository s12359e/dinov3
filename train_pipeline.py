#!/usr/bin/env python3
"""
Standalone training pipeline for DINOv3 Adapter v2 + UNet Decoder.
No MMSegmentation dependency.  Reads images + YOLO segmentation .txt labels.

Usage:
    python train_pipeline.py \
        --data_root data/sem_defect \
        --checkpoint checkpoints/dinov3_vitb16_pretrain.pth \
        --msda_mode original \
        --img_size 512 \
        --batch_size 4 \
        --max_iters 20000 \
        --lr 1e-4

Expected directory layout under data_root:
    images/train/   *.png
    images/val/     *.png
    labels/train/   *.txt   (YOLO segmentation format)
    labels/val/     *.txt
"""

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.eval.segmentation.models.backbone.dinov3_adapter_v2 import (
    DINOv3_Adapter,
    DynamicUNetDecoder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pipeline")


# ====================================================================
#  1. Dataset -- images + YOLO segmentation labels
# ====================================================================

class YOLOSegDataset(Dataset):
    """Reads images and YOLO-format polygon segmentation labels.

    YOLO .txt format (one line per polygon instance)::

        class_id  x1 y1 x2 y2 ... xn yn

    Coordinates are normalised to [0, 1].  When ``binary=True`` every
    YOLO class is mapped to label 1 (defect); background is 0.

    Args:
        images_dir: folder with image files.
        labels_dir: folder with matching .txt files.
        transform:  callable ``(image_np, mask_np) -> (image_tensor, mask_tensor)``.
        img_suffix: image file extension filter.
        binary:     map all YOLO classes to 1.
    """

    def __init__(self, images_dir, labels_dir, transform=None,
                 img_suffix=".png", binary=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.binary = binary

        self.image_files = sorted(
            [f for f in self.images_dir.iterdir()
             if f.suffix.lower() in (img_suffix, ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")]
        )
        if not self.image_files:
            raise FileNotFoundError(
                f"No images found in {images_dir} with suffix {img_suffix}"
            )
        logger.info(f"Dataset: {len(self.image_files)} images from {images_dir}")

    # ------------------------------------------------------------------
    @staticmethod
    def parse_yolo_label(label_path, H, W, binary=True):
        """Convert a YOLO .txt file to a (H, W) uint8 mask."""
        mask = np.zeros((H, W), dtype=np.uint8)
        if not os.path.exists(label_path):
            return mask  # no label = pure background
        with open(label_path) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:          # class_id + at least 3 points
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            points = np.array(
                [[coords[i] * W, coords[i + 1] * H]
                 for i in range(0, len(coords), 2)],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            label = 1 if binary else class_id + 1
            cv2.fillPoly(mask, [points], color=int(label))
        return mask

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Read as 3-channel (handles grayscale → RGB automatically)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = self.labels_dir / (img_path.stem + ".txt")
        H, W = image.shape[:2]
        mask = self.parse_yolo_label(str(label_path), H, W, self.binary)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


# ====================================================================
#  2. Augmentations (joint image + mask transforms)
# ====================================================================

class TrainTransform:
    """Training augmentations matching the SEM defect pipeline.

    Pipeline: RandomResize -> RandomCrop -> HFlip -> VFlip ->
              PhotoMetric (brightness/contrast) -> Normalize -> Tensor.
    """

    def __init__(self, img_size=512, scale_range=(0.75, 1.25),
                 flip_prob=0.5, brightness_delta=20,
                 contrast_range=(0.8, 1.2),
                 mean=(109.65, 104.81, 75.48),
                 std=(54.32, 39.78, 36.47)):
        self.img_size = img_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image, mask):
        # -- random resize (keep aspect ratio) -------------------------
        scale = np.random.uniform(*self.scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # -- random crop to (img_size, img_size) -----------------------
        h, w = image.shape[:2]
        cs = self.img_size
        if h < cs or w < cs:
            pad_h, pad_w = max(cs - h, 0), max(cs - w, 0)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w,
                                      cv2.BORDER_CONSTANT, value=255)
            h, w = image.shape[:2]
        top = np.random.randint(0, h - cs + 1)
        left = np.random.randint(0, w - cs + 1)
        image = image[top:top + cs, left:left + cs]
        mask = mask[top:top + cs, left:left + cs]

        # -- random flips ---------------------------------------------
        if np.random.random() < self.flip_prob:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))
        if np.random.random() < self.flip_prob:
            image = np.ascontiguousarray(np.flipud(image))
            mask = np.ascontiguousarray(np.flipud(mask))

        # -- photometric distortion (brightness + contrast) -----------
        img_f = image.astype(np.float32)
        if np.random.random() < 0.5:
            img_f += np.random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
        if np.random.random() < 0.5:
            img_f *= np.random.uniform(*self.contrast_range)
        image = np.clip(img_f, 0, 255).astype(np.uint8)

        # -- normalize & to tensor -------------------------------------
        image = (image.astype(np.float32) - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1).float()   # (3, H, W)
        mask = torch.from_numpy(mask.copy()).long()                 # (H, W)
        return image, mask


class ValTransform:
    """Validation transform: resize + normalize."""

    def __init__(self, img_size=512,
                 mean=(109.65, 104.81, 75.48),
                 std=(54.32, 39.78, 36.47)):
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image, mask):
        image = cv2.resize(image, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)
        image = (image.astype(np.float32) - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.copy()).long()
        return image, mask


# ====================================================================
#  3. Loss Functions -- Dice + Focal
# ====================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Computes per-class Dice and returns ``1 - mean(Dice)``.
    """

    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, C, H, W) raw logits.
            target: (B, H, W) class indices.
        """
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)

        valid = (target != self.ignore_index)
        target_clean = target.clone()
        target_clean[~valid] = 0

        target_oh = F.one_hot(target_clean, num_classes)    # (B, H, W, C)
        target_oh = target_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        valid_mask = valid.unsqueeze(1).float()             # (B, 1, H, W)

        intersection = (pred_soft * target_oh * valid_mask).sum(dim=(2, 3))
        cardinality = ((pred_soft + target_oh) * valid_mask).sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced segmentation.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: per-class weight list (e.g. [0.25, 0.75] for [bg, defect]),
               or a single float for binary (applied to the positive class).
        gamma: focusing parameter. Default 2.0.
    """

    def __init__(self, alpha=0.75, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, C, H, W) raw logits.
            target: (B, H, W) class indices.
        """
        num_classes = pred.shape[1]
        ce = F.cross_entropy(pred, target, reduction="none",
                             ignore_index=self.ignore_index)

        # p_t : probability of the correct class
        target_clean = target.clone()
        target_clean[target == self.ignore_index] = 0
        p_t = F.softmax(pred, dim=1).gather(
            1, target_clean.unsqueeze(1)
        ).squeeze(1)

        # alpha weighting
        if isinstance(self.alpha, (list, tuple)):
            alpha_tensor = torch.tensor(self.alpha, device=pred.device,
                                        dtype=pred.dtype)
            alpha_t = alpha_tensor[target_clean]
        else:
            alpha_t = torch.where(target_clean > 0,
                                  self.alpha, 1.0 - self.alpha)

        focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)
        valid = (target != self.ignore_index).float()
        loss = (focal_weight * ce * valid).sum() / (valid.sum() + 1e-8)
        return loss


class CombinedLoss(nn.Module):
    """Weighted sum of Dice loss and Focal loss."""

    def __init__(self, dice_weight=3.0, focal_weight=1.0,
                 focal_alpha=0.75, focal_gamma=2.0, ignore_index=255):
        super().__init__()
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                               ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        return (self.dice_weight * self.dice(pred, target)
                + self.focal_weight * self.focal(pred, target))


# ====================================================================
#  4. Metric -- Dice Score
# ====================================================================

class DiceScoreMetric:
    """Accumulates intersection / area counts across batches then computes Dice."""

    def __init__(self, num_classes=2, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self.pred_area = torch.zeros(self.num_classes, dtype=torch.float64)
        self.label_area = torch.zeros(self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred, target):
        """
        Args:
            pred:   (B, C, H, W) logits.
            target: (B, H, W) class indices.
        """
        pred_label = pred.argmax(dim=1)         # (B, H, W)
        valid = target != self.ignore_index
        pred_flat = pred_label[valid].cpu()
        target_flat = target[valid].cpu()

        for c in range(self.num_classes):
            pc = pred_flat == c
            tc = target_flat == c
            self.intersection[c] += (pc & tc).sum().item()
            self.pred_area[c] += pc.sum().item()
            self.label_area[c] += tc.sum().item()

    def compute(self):
        dice = (2.0 * self.intersection) / (
            self.pred_area + self.label_area + 1e-8
        )
        result = {
            "mean_dice": dice.mean().item(),
        }
        for c in range(self.num_classes):
            name = "background" if c == 0 else f"class_{c}"
            result[f"dice_{name}"] = dice[c].item()
        return result


# ====================================================================
#  5. Model Pipeline
# ====================================================================

class SegmentationPipeline(nn.Module):
    """Wraps Adapter + Decoder into a single forward call."""

    def __init__(self, adapter, decoder):
        super().__init__()
        self.adapter = adapter
        self.decoder = decoder

    def forward(self, x):
        feats = self.adapter(x)
        return self.decoder(feats, original_size=x.shape[2:])


def build_model(args):
    """Build the full segmentation model from CLI args."""
    # -- ViT backbone ---------------------------------------------------
    vit = DinoVisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    )
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading ViT weights from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        # Handle common checkpoint wrappers
        state = ckpt
        for key in ("model", "teacher", "state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
        # Strip 'backbone.' prefix if present
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        msg = vit.load_state_dict(state, strict=False)
        logger.info(f"  missing={len(msg.missing_keys)}  "
                    f"unexpected={len(msg.unexpected_keys)}")

    # -- interaction indexes for depth -----------------------------------
    interaction_indexes = _interaction_indexes(args.depth)
    logger.info(f"interaction_indexes={interaction_indexes}")

    # -- Adapter --------------------------------------------------------
    adapter = DINOv3_Adapter(
        backbone=vit,
        msda_mode=args.msda_mode,
        interaction_indexes=interaction_indexes,
        pretrain_size=args.img_size,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=args.grad_checkpoint,
    )

    # -- Decoder --------------------------------------------------------
    n_out = {"original": 4, "4-level": 4, "5-level": 5}[args.msda_mode]
    finest_stride = {"original": 4, "4-level": 4, "5-level": 2}[args.msda_mode]

    decoder = DynamicUNetDecoder(
        adapter_channels=[args.embed_dim] * n_out,
        num_classes=args.num_classes,
        use_pixel_shuffle=args.use_pixel_shuffle,
        use_gated_attention=args.use_gated_attention,
        use_novel_enhancement=args.use_pfg,
        periodic_pitch=args.periodic_pitch,
        finest_stride=finest_stride,
    )

    model = SegmentationPipeline(adapter, decoder)

    # Convert SyncBatchNorm -> BatchNorm for single-GPU
    if not args.distributed:
        model = _convert_sync_bn(model)

    return model


def _interaction_indexes(depth):
    """Evenly-spaced ViT block indices for the adapter."""
    n = 4
    step = depth // n
    return [step * (i + 1) - 1 for i in range(n)]


def _convert_sync_bn(module):
    """Replace all SyncBatchNorm with BatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            if child.affine:
                bn.weight = child.weight
                bn.bias = child.bias
            bn.running_mean = child.running_mean
            bn.running_var = child.running_var
            bn.num_batches_tracked = child.num_batches_tracked
            setattr(module, name, bn)
        else:
            _convert_sync_bn(child)
    return module


# ====================================================================
#  6. LR Scheduler -- Warmup + Cosine
# ====================================================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine annealing to ``eta_min``."""

    def __init__(self, optimizer, warmup_iters, total_iters, eta_min=1e-7,
                 last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_iters:
            alpha = step / max(self.warmup_iters, 1)
        else:
            progress = (step - self.warmup_iters) / max(
                self.total_iters - self.warmup_iters, 1
            )
            alpha = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * alpha
            for base_lr in self.base_lrs
        ]


# ====================================================================
#  7. Training / Validation Loops
# ====================================================================

@torch.no_grad()
def validate(model, val_loader, metric, device, amp_enabled=True):
    model.eval()
    metric.reset()
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            pred = model(images)
        if pred.shape[-2:] != masks.shape[-2:]:
            pred = F.interpolate(pred, size=masks.shape[-2:],
                                 mode="bilinear", align_corners=False)
        metric.update(pred, masks)
    return metric.compute()


def train(args):
    # -- device ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # -- seed -----------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # -- data -----------------------------------------------------------
    data_root = Path(args.data_root)
    mean = tuple(args.img_mean)
    std = tuple(args.img_std)

    train_ds = YOLOSegDataset(
        images_dir=data_root / "images" / "train",
        labels_dir=data_root / "labels" / "train",
        transform=TrainTransform(
            img_size=args.img_size,
            scale_range=tuple(args.scale_range),
            mean=mean, std=std,
        ),
        binary=args.binary,
    )
    val_ds = YOLOSegDataset(
        images_dir=data_root / "images" / "val",
        labels_dir=data_root / "labels" / "val",
        transform=ValTransform(img_size=args.img_size, mean=mean, std=std),
        binary=args.binary,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -- model ----------------------------------------------------------
    model = build_model(args).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

    # -- optimizer & scheduler ------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_iters=args.warmup_iters,
        total_iters=args.max_iters,
        eta_min=1e-7,
    )

    # -- loss & metric --------------------------------------------------
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    ).to(device)
    metric = DiceScoreMetric(num_classes=args.num_classes)

    # -- AMP scaler -----------------------------------------------------
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    # -- output dir -----------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- resume ---------------------------------------------------------
    start_step = 0
    best_dice = 0.0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        best_dice = ckpt.get("best_dice", 0.0)

    # -- training loop (iteration-based) --------------------------------
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    t0 = time.time()

    for step in range(start_step + 1, args.max_iters + 1):
        # fetch batch (infinite iteration)
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # forward + backward
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            pred = model(images)
            if pred.shape[-2:] != masks.shape[-2:]:
                pred = F.interpolate(pred, size=masks.shape[-2:],
                                     mode="bilinear", align_corners=False)
            loss = criterion(pred, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            optimizer.step()

        scheduler.step()
        running_loss += loss.item()

        # -- logging ---------------------------------------------------
        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            it_per_sec = args.log_interval / elapsed
            logger.info(
                f"Step {step}/{args.max_iters} | "
                f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                f"{it_per_sec:.1f} it/s"
            )
            running_loss = 0.0
            t0 = time.time()

        # -- validation ------------------------------------------------
        if step % args.val_interval == 0:
            results = validate(model, val_loader, metric, device, amp_enabled)
            logger.info(
                f"[Val] Step {step} | "
                + " | ".join(f"{k}={v:.4f}" for k, v in results.items())
            )
            model.train()

            # save best
            current_dice = results.get("dice_class_1", results["mean_dice"])
            is_best = current_dice > best_dice
            if is_best:
                best_dice = current_dice
                _save_checkpoint(model, optimizer, scheduler, step, best_dice,
                                 output_dir / "best.pth")
                logger.info(f"  -> New best dice: {best_dice:.4f}")

        # -- periodic checkpoint ---------------------------------------
        if step % args.save_interval == 0:
            _save_checkpoint(model, optimizer, scheduler, step, best_dice,
                             output_dir / f"step_{step}.pth")

    # -- final validation -----------------------------------------------
    results = validate(model, val_loader, metric, device, amp_enabled)
    logger.info(
        f"[Final] "
        + " | ".join(f"{k}={v:.4f}" for k, v in results.items())
    )
    _save_checkpoint(model, optimizer, scheduler, args.max_iters, best_dice,
                     output_dir / "final.pth")
    logger.info(f"Training complete. Best dice: {best_dice:.4f}")
    return results


def _save_checkpoint(model, optimizer, scheduler, step, best_dice, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_dice": best_dice,
    }, path)
    logger.info(f"Saved checkpoint: {path}")


# ====================================================================
#  8. CLI
# ====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DINOv3 Segmentation Training")

    # data
    p.add_argument("--data_root", type=str, required=True,
                   help="Root with images/{train,val} and labels/{train,val}")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--scale_range", type=float, nargs=2, default=[0.75, 1.25])
    p.add_argument("--img_mean", type=float, nargs=3,
                   default=[109.65, 104.81, 75.48])
    p.add_argument("--img_std", type=float, nargs=3,
                   default=[54.32, 39.78, 36.47])
    p.add_argument("--binary", action="store_true", default=True,
                   help="Binary segmentation (all YOLO classes -> 1)")
    p.add_argument("--num_classes", type=int, default=2)

    # model
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to DINOv3 pretrained ViT weights")
    p.add_argument("--msda_mode", type=str, default="original",
                   choices=["original", "4-level", "5-level"])
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--use_pixel_shuffle", action="store_true")
    p.add_argument("--use_gated_attention", action="store_true")
    p.add_argument("--use_pfg", action="store_true",
                   help="Enable Periodic Frequency Gating")
    p.add_argument("--periodic_pitch", type=int, default=22)
    p.add_argument("--grad_checkpoint", action="store_true", default=True)

    # training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_iters", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--distributed", action="store_true", default=False)

    # loss
    p.add_argument("--dice_weight", type=float, default=3.0)
    p.add_argument("--focal_weight", type=float, default=1.0)
    p.add_argument("--focal_alpha", type=float, default=0.75)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # logging & saving
    p.add_argument("--output_dir", type=str, default="work_dirs/train")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--val_interval", type=int, default=1000)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--resume", type=str, default="",
                   help="Path to checkpoint to resume training from")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

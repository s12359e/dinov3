# SEM Micro-Defect Segmentation Pipeline

Binary segmentation of 4x4-pixel metal-gate extrusion defects in 512x512 synthetic SEM images.

Two training paths:
- **Standalone** (`train_pipeline.py`) -- no MMSeg dependency, recommended
- **MMSegmentation** (`sem_defect_pipeline/configs/`) -- if you already use MMSeg

---

## Quick Start (End-to-End)

```bash
# 1. Generate synthetic dataset (200 SRAM + 200 Logic images)
python sem_defect_pipeline/data_gen/generate_sem_dataset.py \
    --output_dir data/sem_defect

# 2. Train (standalone, no MMSeg)
python train_pipeline.py \
    --data_root data/sem_defect \
    --checkpoint checkpoints/dinov3_vitb16_pretrain.pth \
    --msda_mode original \
    --max_iters 20000 \
    --batch_size 4 \
    --lr 1e-4
```

---

## Project Structure

```
dinov3/
├── train_pipeline.py                              # Standalone training (no MMSeg)
├── dinov3/eval/segmentation/models/backbone/
│   ├── dinov3_adapter.py                          # Original ViT-Adapter (3-level MSDA)
│   └── dinov3_adapter_v2.py                       # Configurable Adapter + UNet Decoder + PFG
├── sem_defect_pipeline/
│   ├── configs/
│   │   ├── dinov3_vitb16_fpn_sem512.py            # MMSeg: Plain ViT + PNG masks
│   │   ├── dinov3_vitb16_fpn_yolo_sem512.py       # MMSeg: Plain ViT + YOLO labels
│   │   ├── dinov3_adapter_vitb16_fpn_sem512.py    # MMSeg: ViT-Adapter + PNG masks
│   │   └── dinov3_adapter_vitb16_fpn_yolo_sem512.py  # MMSeg: ViT-Adapter + YOLO labels
│   ├── data_gen/
│   │   └── generate_sem_dataset.py                # Synthetic SEM image generator
│   ├── dinov3_backbone.py                         # MMSeg backbone wrappers
│   └── transforms.py                             # LoadYOLOAnnotations transform
└── data/sem_defect/                               # Generated dataset (not tracked)
    ├── images/{train,val,test}/*.png
    ├── masks/{train,val,test}/*.png
    └── labels/{train,val,test}/*.txt
```

---

## 1. Synthetic Data Generation

### Region Types

| Region | Description | Gates |
|--------|-------------|-------|
| **SRAM** | Standard memory array | Continuous vertical stripes |
| **Logic** | Logic area with CPODE | Gates have **cuts** (horizontal gaps at band boundaries) |

### SEM Image Topology

| Component | Description | Brightness |
|-----------|-------------|------------|
| Metal Gate | Vertical stripes, period=22px, width=7px | ~230-250 |
| PEPI band | Horizontal regions (height=64px) | ~160-190 |
| NEPI band | Between PEPI bands | ~60-90 |
| Gate Cut (Logic only) | 3-6px gap at band boundary, exposes underlying band | Same as band |
| Defect | 4x4 square extruding from gate right edge into NEPI | Same as gate |

Noise model: Gaussian blur (PSF, sigma=0.7) + additive Gaussian noise (sigma=12).

### Commands

```bash
# Default: 200 SRAM + 200 Logic, split 70/15/15 into train/val/test
python sem_defect_pipeline/data_gen/generate_sem_dataset.py

# SRAM only, 500 images
python sem_defect_pipeline/data_gen/generate_sem_dataset.py \
    --region sram --n_per_region 500

# Logic only with custom noise
python sem_defect_pipeline/data_gen/generate_sem_dataset.py \
    --region logic --noise_std 8 --blur_sigma 0.5

# Preview a Logic sample (saves to file)
python sem_defect_pipeline/data_gen/generate_sem_dataset.py \
    --preview_save preview.png --preview_region logic
```

### Output Format

Three parallel directories are generated:

| Directory | Format | Description |
|-----------|--------|-------------|
| `images/` | 3-channel uint8 PNG (R=G=B grayscale) | Input image |
| `masks/` | Single-channel uint8 PNG (0=bg, 1=defect) | Pixel mask |
| `labels/` | YOLO segmentation .txt | Normalized polygon coords |

Filenames: `{region}_{split}_{index:05d}.png` (e.g., `logic_train_00042.png`)

YOLO label format (each line): `class_id x1 y1 x2 y2 ... xn yn` with coordinates in [0, 1].

---

## 2. Model Architecture (`dinov3_adapter_v2.py`)

### DINOv3_Adapter -- Configurable ViT-Adapter Backbone

Three MSDA (Multi-Scale Deformable Attention) modes:

| Mode | Levels | Strides | Description |
|------|--------|---------|-------------|
| `original` | 3 | 8, 16, 32 | Same as original adapter; c1 (stride 4) bypasses MSDA |
| `4-level` | 4 | 4, 8, 16, 32 | All 4 scales participate in deformable attention |
| `5-level` | 5 | 2, 4, 8, 16, 32 | Extra stride-2 level from SPM stem for maximum resolution |

```python
from dinov3.eval.segmentation.models.backbone.dinov3_adapter_v2 import (
    DINOv3_Adapter, DynamicUNetDecoder, PeriodicFrequencyGating
)

# Backbone
adapter = DINOv3_Adapter(
    pretrain_size=512,
    conv_inplane=64,
    n_points=4,
    deform_num_heads=12,
    with_cffn=True,
    interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
    msda_mode='4-level',        # '4-level' recommended for defect detection
)
```

### DynamicUNetDecoder -- Segmentation Decoder

Accepts multi-scale features from the adapter and produces a segmentation mask.

| Toggle | Effect |
|--------|--------|
| `use_pixel_shuffle=True` | PixelShuffle upsampling (sub-pixel convolution) instead of bilinear |
| `use_gated_attention=True` | Semantic Spatial Gate on skip connections (channel attention + spatial gate) |

```python
decoder = DynamicUNetDecoder(
    encoder_channels=[768, 768, 768, 768],  # 4 outputs from adapter
    num_classes=2,
    use_pixel_shuffle=True,
    use_gated_attention=True,
)
```

### PeriodicFrequencyGating (PFG) -- Novel Enhancement

FFT-based spectral notch filter that suppresses the periodic gate pattern (22-pixel pitch), letting the model focus on defect signals.

```python
pfg = PeriodicFrequencyGating(
    in_channels=768,
    periodic_pitch=22,       # matches Metal Gate X-axis period
    notch_width=2,
    learnable_alpha=True,
)
```

### build_pipeline -- Quick Assembly

```python
from dinov3.eval.segmentation.models.backbone.dinov3_adapter_v2 import build_pipeline

model = build_pipeline(
    pretrain_size=512,
    msda_mode='4-level',
    num_classes=2,
    use_pixel_shuffle=True,
    use_gated_attention=True,
    use_pfg=True,
    periodic_pitch=22,
)
# model.adapter, model.decoder, model.pfg
```

---

## 3. Standalone Training (`train_pipeline.py`)

No MMSegmentation required. Uses PyTorch directly with:
- YOLO .txt label loading (polygon -> dense mask via cv2.fillPoly)
- Dice Loss + Focal Loss (configurable weights)
- Dice Score as validation metric
- AMP (mixed precision), gradient clipping, warmup + cosine LR schedule

### Full Command Reference

```bash
python train_pipeline.py \
    --data_root data/sem_defect \
    --checkpoint checkpoints/dinov3_vitb16_pretrain.pth \
    \
    # Architecture
    --msda_mode original \           # original | 4-level | 5-level
    --embed_dim 768 \
    --patch_size 16 \
    --depth 12 \
    --num_heads 12 \
    --use_pixel_shuffle \            # enable PixelShuffle in decoder
    --use_gated_attention \          # enable Semantic Spatial Gate
    --use_pfg \                      # enable Periodic Frequency Gating
    --periodic_pitch 22 \
    \
    # Data
    --img_size 512 \
    --batch_size 4 \
    --num_workers 4 \
    --num_classes 2 \
    \
    # Training
    --max_iters 20000 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --warmup_iters 1000 \
    --grad_clip 1.0 \
    \
    # Loss
    --dice_weight 3.0 \
    --focal_weight 1.0 \
    --focal_alpha 0.75 \
    --focal_gamma 2.0 \
    \
    # Output
    --output_dir work_dirs/train \
    --log_interval 50 \
    --val_interval 1000 \
    --save_interval 2000 \
    --resume ""                      # path to resume checkpoint
```

### Minimal Example

```bash
# Simplest invocation (all defaults)
python train_pipeline.py \
    --data_root data/sem_defect \
    --checkpoint checkpoints/dinov3_vitb16_pretrain.pth
```

### Expected Data Layout

```
data/sem_defect/
├── images/
│   ├── train/   *.png    # 3-channel uint8
│   └── val/     *.png
└── labels/
    ├── train/   *.txt    # YOLO segmentation format
    └── val/     *.txt
```

### Training Features

| Feature | Details |
|---------|---------|
| Mixed Precision | AMP with bfloat16 (auto-disabled on CPU) |
| Gradient Clipping | max_norm=1.0 by default |
| LR Schedule | Linear warmup (1000 iters) -> cosine annealing to 1e-7 |
| Loss | 3.0 * DiceLoss + 1.0 * FocalLoss (alpha=0.75, gamma=2.0) |
| Validation Metric | Per-class Dice Score |
| Checkpointing | Every 2000 iters + best Dice checkpoint |
| Augmentation | RandomResize, RandomCrop, HFlip, VFlip, Brightness/Contrast jitter |
| Normalization | mean=[109.65, 104.81, 75.48], std=[54.32, 39.78, 36.47] |

---

## 4. MMSegmentation Training (Alternative)

If you prefer MMSeg, four config files are provided.

### Choose Your Config

| Label Format | Backbone | Config |
|-------------|----------|--------|
| PNG masks | Plain ViT | `dinov3_vitb16_fpn_sem512.py` |
| PNG masks | ViT-Adapter | `dinov3_adapter_vitb16_fpn_sem512.py` |
| YOLO .txt | Plain ViT | `dinov3_vitb16_fpn_yolo_sem512.py` |
| YOLO .txt | ViT-Adapter | `dinov3_adapter_vitb16_fpn_yolo_sem512.py` |

### Train

```bash
# Edit checkpoint path in config first!
# Default: checkpoints/dinov3_vitb16_pretrain.pth

# Example: ViT-Adapter + YOLO labels
python -m mmseg.tools.train \
    sem_defect_pipeline/configs/dinov3_adapter_vitb16_fpn_yolo_sem512.py
```

### Requirements (MMSeg path only)

```
mmsegmentation >= 1.2.0
mmengine >= 0.9.0
mmcv >= 2.0.0rc4
```

---

## 5. Key Design Decisions

### Why Dice + Focal Loss?

Defect area is ~16 pixels out of 262,144 (0.006%). Standard CrossEntropy ignores the defect entirely.
- **Focal Loss** (alpha=0.75, gamma=2.0): down-weights easy background pixels, focuses on hard examples
- **Dice Loss** (weight=3.0): directly optimizes set-overlap (IoU proxy), unaffected by class frequency

### Why gate cuts in Logic?

Real semiconductor logic regions have CPODE (Continuous Poly On Diffusion Edge) cuts that interrupt metal gates. Training on both SRAM (continuous) and Logic (cut gates) ensures the model learns that:
- Continuous gates = normal for SRAM
- Cut gates = normal for Logic (not a defect)
- Extrusion into NEPI = defect (regardless of region)

### Why Periodic Frequency Gating?

The 22-pixel gate period creates strong spectral peaks. PFG applies a learnable notch filter in the frequency domain to suppress these periodic patterns, making the defect signal more prominent for the decoder.

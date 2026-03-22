# SEM Micro-Defect Segmentation Pipeline

Binary segmentation of 4×4-pixel metal-gate extrusion defects in 512×512 synthetic SEM images, built on DINOv3 + MMSegmentation.

## Overview

```
sem_defect_pipeline/
├── configs/
│   ├── dinov3_vitb16_fpn_sem512.py               # Plain ViT + PNG masks
│   ├── dinov3_vitb16_fpn_yolo_sem512.py           # Plain ViT + YOLO labels
│   ├── dinov3_adapter_vitb16_fpn_sem512.py        # ViT-Adapter + PNG masks
│   └── dinov3_adapter_vitb16_fpn_yolo_sem512.py   # ViT-Adapter + YOLO labels
├── data_gen/
│   └── generate_sem_dataset.py               # Synthetic SEM image generator
├── dinov3_backbone.py                        # MMSeg backbone wrappers
└── transforms.py                            # Custom transforms (YOLO label loading)
```

## Synthetic SEM Images

The generator produces grayscale images (3-channel, R=G=B) simulating SEM scans of semiconductor structures:

| Component | Description | Brightness |
|-----------|-------------|------------|
| Metal Gate | Vertical stripes, period=22px | ~230-250 |
| PEPI band | Horizontal regions | ~160-190 |
| NEPI band | Between PEPI bands | ~60-90 |
| Defect | 4×4 square extruding from gate right edge into NEPI | Same as gate |

Noise model: Gaussian blur (PSF, σ=0.7) + additive Gaussian noise (σ=12).

### Generate dataset

```bash
# Default: 800 train / 100 val / 100 test, 70% with defects
python sem_defect_pipeline/data_gen/generate_sem_dataset.py

# Custom
python sem_defect_pipeline/data_gen/generate_sem_dataset.py \
    --n_train 2000 --n_val 200 --n_test 200 \
    --defect_ratio 0.7 --seed 42 \
    --output_dir data/sem_defect
```

Output structure:
```
data/sem_defect/
├── images/{train,val,test}/*.png   # 3-channel uint8
├── masks/{train,val,test}/*.png    # single-channel: 0=bg, 1=defect
└── metadata.json                   # generation params + defect bboxes
```

### Grayscale input

Both single-channel and 3-channel grayscale images are supported. The data pipeline uses `LoadImageFromFile(color_type='color')`, which automatically converts single-channel grayscale to 3-channel (R=G=B) via `cv2.IMREAD_COLOR`. No preprocessing or manual conversion is needed.

## Label Formats

### PNG masks (default)

Single-channel uint8 PNG where pixel values are class labels (0=background, 1=defect).

```
data/sem_defect/
├── images/train/*.png
└── masks/train/*.png       # same filename as image
```

Config: `dinov3_vitb16_fpn_sem512.py` or `dinov3_adapter_vitb16_fpn_sem512.py`

### YOLO segmentation labels (.txt)

Each line: `class_id x1 y1 x2 y2 ... xn yn` with normalized [0,1] polygon coordinates. Multiple lines per file for multiple instances. Empty or missing files are treated as pure background (negative samples).

```
data/sem_defect/
├── images/train/*.png
└── labels/train/*.txt      # same stem as image
```

Config: `dinov3_vitb16_fpn_yolo_sem512.py` or `dinov3_adapter_vitb16_fpn_yolo_sem512.py`

The `LoadYOLOAnnotations` transform converts polygons to dense pixel masks at runtime. By default all YOLO classes map to label 1 (binary mode). Set `binary=False` for multi-class.

## Model Architecture

### Option A: Plain ViT (`DINOv3ViTBackbone`)

```
DINOv3 ViT-B/16 → FPN → FPNHead → 2-class segmentation
```

- Extracts features at blocks [3, 5, 7, 11] — all at stride 16
- FPN generates multi-scale from single-stride inputs
- 90.4M total params

### Option B: ViT-Adapter (`DINOv3AdapterBackbone`)

```
DINOv3 ViT-B/16 (frozen) + Adapter → FPN → FPNHead → 2-class segmentation
```

- Adapter adds SPM + deformable-attention interaction blocks
- True multi-scale output at strides 4, 8, 16, 32
- ViT frozen (85.6M); only adapter (10.2M) + neck + head trained (15.0M trainable)
- Requires GPU (MSDeformAttn backward needs CUDA)

### Loss

CrossEntropyLoss (class_weight=[1.0, 50.0]) + DiceLoss (weight=3.0) to handle extreme class imbalance (~0.006% defect area).

## Requirements

```
torch >= 2.0
mmsegmentation >= 1.2.0
mmengine >= 0.9.0
mmcv >= 2.0.0rc4
opencv-python
numpy
```

For the adapter backbone, you also need CUDA and the compiled `MultiScaleDeformableAttention` op.

## Training

```bash
# Plain ViT + PNG masks
python -m mmseg.tools.train sem_defect_pipeline/configs/dinov3_vitb16_fpn_sem512.py

# Plain ViT + YOLO labels
python -m mmseg.tools.train sem_defect_pipeline/configs/dinov3_vitb16_fpn_yolo_sem512.py

# ViT-Adapter + PNG masks
python -m mmseg.tools.train sem_defect_pipeline/configs/dinov3_adapter_vitb16_fpn_sem512.py

# ViT-Adapter + YOLO labels
python -m mmseg.tools.train sem_defect_pipeline/configs/dinov3_adapter_vitb16_fpn_yolo_sem512.py
```

Before training, update `checkpoint` in the config to point to your DINOv3 ViT-B/16 pretrained weights.

## Validation Status

Tested on CPU (PyTorch 2.10, mmseg 1.2.2):

| Component | Status |
|-----------|--------|
| Data generation (shape, dtype, defect placement, reproducibility) | Passed |
| Plain ViT backbone forward | Passed |
| Adapter backbone forward (stride 4/8/16/32 outputs) | Passed |
| Full model build (EncoderDecoder via MMSeg registry) | Passed |
| Loss computation (CrossEntropy + Dice) | Passed |
| Data pipeline — PNG masks (LoadImage → Resize → PackSegInputs) | Passed |
| Data pipeline — YOLO labels (LoadYOLOAnnotations → polygon→mask) | Passed |
| YOLO: single polygon, multi-polygon, empty file, missing file | Passed |
| YOLO: multi-class mode (binary=False) | Passed |
| Gradient flow + optimizer step (plain ViT) | Passed |
| Inference / predict mode | Passed |

Not yet validated: full training loop via Runner, checkpoint loading, multi-GPU DDP, convergence.

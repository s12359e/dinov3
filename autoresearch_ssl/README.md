# DINOv3 SSL Autoresearch

Autonomous hyperparameter search for DINOv3 self-supervised learning on semiconductor SEM defect images.

An AI agent (Claude Code) iteratively modifies `train.py` hyperparameters, runs 5-minute training experiments, and keeps the configuration that maximizes the **defect KNN anomaly score**.

## Architecture

```
autoresearch_ssl/
  prepare.py     # FIXED evaluation metric (DO NOT MODIFY)
  train.py       # SSL training script (AI agent modifies this)
  config.yaml    # Default hyperparameters
  program.md     # AI agent instructions
  results.tsv    # Experiment log (auto-generated)
```

| File | Role | Modifiable? |
|------|------|-------------|
| `prepare.py` | Evaluation function (`evaluate_ssl_knn`) + data utilities | NO |
| `train.py` | DINO student/teacher training with ViT backbone | YES |
| `config.yaml` | Default hyperparameters, overrides globals in `train.py` | YES |
| `program.md` | Instructions for the AI agent | NO (during experiments) |

## Metric

**`best_anomaly_score`** — single-image patch-level KNN cosine similarity anomaly score.

```
For each defect image:
  1. Extract patch tokens from ViT teacher backbone
  2. Locate defect patch(es) by pixel coordinates in filename (_x{X}_y{Y})
  3. Compute cosine similarity to K nearest NORMAL patches
  4. anomaly = 1 - mean_cosine_similarity
Final score = mean anomaly across all images
```

Higher score = defect features are more distinguishable from normal = better SSL model.

## Setup

### 1. Generate data (if not already done)

```bash
cd dinov3
python sem_defect_pipeline/data_gen/generate_sem_dataset.py \
    --output_dir data/sem_defect \
    --region sram logic \
    --n_per_region 200
```

### 2. Prepare eval images (rename with defect coordinates)

```bash
cd autoresearch_ssl
python prepare.py \
    --eval-dir ../data/sem_defect/images/val \
    --metadata ../data/sem_defect/metadata.json
```

This renames val images from `sram_val_00001.png` to `sram_val_00001_x342_y474.png`.

Verify:
```bash
python prepare.py --eval-dir ../data/sem_defect/images/val
# Should show: Found 47/63 images with _x{}_y{} coordinates
```

### 3. (Optional) Download pretrained checkpoint

If you have a DINOv3 pretrained ViT checkpoint:
```bash
mkdir -p ../checkpoints
# Place checkpoint at: ../checkpoints/dinov3_vitb16_pretrain.pth
```

## Manual Usage (Single Experiment)

### With GPU

```bash
python train.py \
    --config config.yaml \
    --train-dir ../data/sem_defect/images/train \
    --eval-dir ../data/sem_defect/images/val
```

### With pretrained checkpoint

```bash
python train.py \
    --config config.yaml \
    --train-dir ../data/sem_defect/images/train \
    --eval-dir ../data/sem_defect/images/val \
    --checkpoint ../checkpoints/dinov3_vitb16_pretrain.pth
```

### Custom output directory

```bash
python train.py \
    --config config.yaml \
    --train-dir ../data/sem_defect/images/train \
    --eval-dir ../data/sem_defect/images/val \
    --output-dir ./my_runs
```

Each run trains for **5 minutes** (wall clock), then evaluates and prints:

```
---
best_anomaly_score: 0.850000
training_seconds:   300.1
total_seconds:      315.2
peak_vram_mb:       8500.0
num_steps:          500
embed_dim:          384
depth:              12
```

## Autonomous AI Agent Mode

This is the primary use case. An AI agent (Claude Code) runs experiments in a loop.

### Start the agent

Open a Claude Code session and say:

```
Read autoresearch_ssl/program.md. Tag: apr6. Start the experiment loop. Do not stop.
```

The agent will:
1. Create branch `autoresearch/apr6`
2. Initialize `results.tsv`
3. Loop: modify `train.py` -> commit -> run 5 min -> evaluate -> keep or revert
4. Never stop until manually interrupted

### What the agent modifies

The agent tunes hyperparameters in `train.py`:

| Category | Parameters |
|----------|-----------|
| Learning rate | `LR`: 1e-4, 3e-4, 5e-4, 1e-3 |
| Teacher EMA | `MOMENTUM_TEACHER`: 0.99, 0.996, 0.999 |
| Temperature | `TEACHER_TEMP`: 0.02, 0.04, 0.07, 0.1 |
| Batch size | `BATCH_SIZE`: 16, 32, 64 |
| Architecture | `EMBED_DIM`: 384, 768 / `DEPTH`: 6, 12, 24 |
| Augmentation | `CROP_SCALE_MIN`, blur, jitter |
| Advanced | iBOT patch loss, KoLeo regularizer |

### Results tracking

The agent logs to `results.tsv`:

```
commit	best_anomaly_score	memory_gb	status	description
a1b2c3d	0.103001	4.2	keep	baseline depth=12 lr=1e-4
e4f5g6h	0.095000	4.2	discard	lr=1e-3 too high
```

### Review results

```bash
# See all experiments
cat results.tsv

# See best score
sort -t$'\t' -k2 -rn results.tsv | head -5

# See git history of train.py changes
git log --oneline autoresearch_ssl/train.py
```

## config.yaml Reference

```yaml
# ViT Architecture
embed_dim: 384            # 384=ViT-S, 768=ViT-B
depth: 12                 # transformer blocks
num_heads: 6              # attention heads
patch_size_cfg: 16        # patch size
drop_path_rate: 0.1       # stochastic depth

# SSL Training
lr: 1e-4                  # base learning rate
weight_decay: 0.04        # AdamW weight decay
warmup_ratio: 0.1         # warmup fraction
batch_size: 32            # images per step
momentum_teacher: 0.996   # EMA momentum
teacher_temp: 0.04        # teacher temperature (lower = sharper)
student_temp: 0.1         # student temperature
proj_dim: 256             # projection head output dim
proj_hidden: 2048         # projection head hidden dim

# Data Augmentation
crop_scale_min: 0.4       # global crop min scale
crop_scale_max: 1.0       # global crop max scale
flip_prob: 0.5            # horizontal flip probability
brightness_delta: 20      # brightness jitter range
```

## Constraints

- **Time**: Each experiment = 5 minutes training (wall clock)
- **VRAM**: Soft constraint. ViT-S/16 bs=32 ~ 8GB. Reduce `batch_size` if OOM
- **Eval data**: Val images must have `_x{X}_y{Y}` in filenames (run `prepare.py` first)

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `No images with _x{}_y{} found` | Run `prepare.py --eval-dir ... --metadata ...` to rename images |
| OOM | Reduce `batch_size` in config.yaml (try 16 or 8) |
| NaN loss | Lower `lr`, check `teacher_temp` not too low |
| Low anomaly score | Try pretrained checkpoint, increase `depth`, tune augmentation |
| Segfault on CPU | Expected on older PyTorch; use GPU or Python 3.12+ |

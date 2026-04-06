# DINOv3 SSL Autoresearch

Autonomous SSL hyperparameter search for semiconductor defect detection.

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). Create branch `autoresearch/<tag>`.
2. **Read the in-scope files**:
   - `prepare.py` — FIXED. Contains evaluation metric (`evaluate_ssl_knn`), data utilities, constants. DO NOT MODIFY.
   - `train.py` — the file you modify. ViT architecture, SSL training, hyperparameters.
   - `config.yaml` — default config. You can also modify this to change parameters.
3. **Verify eval data exists**: Check that the eval directory has images with `_x{X}_y{Y}` in filenames. If not, run: `python prepare.py --eval-dir ../data/sem_defect/images/val --metadata ../data/sem_defect/metadata.json`
4. **Initialize results.tsv** with just the header row.
5. **Confirm and go**.

## Goal

**Maximize `best_anomaly_score`.** This is the single-image patch-level KNN anomaly score:

```
For each defect image:
  1. Extract patch tokens from ViT teacher backbone
  2. Find defect patch(es) by pixel coordinates
  3. Compute cosine similarity to K nearest NORMAL patches
  4. anomaly = 1 - mean_cosine (higher = defect more distinguishable)
Final score = mean anomaly across all images
```

Higher score = SSL learned features where defects stand out from normal structures = better model.

## Experimentation

Each experiment trains for a **fixed time budget of 5 minutes** (wall clock). Launch:

```bash
python train.py --config config.yaml \
    --train-dir ../data/sem_defect/images/train \
    --eval-dir ../data/sem_defect/images/val \
    --checkpoint ../checkpoints/dinov3_vitb16_pretrain.pth
```

Or without a pretrained checkpoint (train from scratch):

```bash
python train.py --config config.yaml \
    --train-dir ../data/sem_defect/images/train \
    --eval-dir ../data/sem_defect/images/val
```

**What you CAN modify:**
- `train.py` — everything: ViT config, SSL loss, optimizer, augmentation, batch size, LR, etc.
- `config.yaml` — change default hyperparameters.

**What you CANNOT modify:**
- `prepare.py` — it contains the fixed evaluation metric. DO NOT TOUCH.

## Output format

The script prints a summary:

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

Extract the key metric:
```bash
grep "^best_anomaly_score:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	best_anomaly_score	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. best_anomaly_score (e.g. 0.850000) — use 0.000000 for crashes
3. peak memory in GB (.1f) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description

## The experiment loop

LOOP FOREVER:

1. Look at git state
2. Modify `train.py` (or `config.yaml`) with an experimental idea
3. git commit
4. Run: `python train.py --config config.yaml --train-dir ../data/sem_defect/images/train --eval-dir ../data/sem_defect/images/val > run.log 2>&1`
5. Read results: `grep "^best_anomaly_score:\|^peak_vram_mb:" run.log`
6. If grep empty → crash. `tail -n 50 run.log` to debug.
7. Record in results.tsv
8. If best_anomaly_score improved (HIGHER) → keep commit
9. If equal or worse → git reset back

## What to try

Suggested experiments (in rough priority order):

### Quick wins
- **Learning rate**: try 3e-4, 5e-4, 1e-3
- **Teacher momentum**: try 0.99, 0.998, 0.999
- **Teacher temperature**: try 0.02, 0.07, 0.1
- **Batch size**: try 16, 64 (affects effective learning)

### Architecture
- **Embed dim**: 384 vs 768 (bigger may be better with pretrained weights)
- **Depth**: 6 vs 12 vs 24
- **Drop path rate**: 0.0, 0.05, 0.2
- **Projection head**: try nlayers=2 vs 4, hidden_dim=1024 vs 4096

### Augmentation (critical for SSL)
- **Crop scale**: try (0.08, 1.0) more aggressive, or (0.6, 1.0) more conservative
- **Add Gaussian blur**: sigma=(0.1, 2.0)
- **Add color jitter/solarization**
- **Multi-crop**: add local crops (smaller resolution)

### Advanced
- **iBOT patch loss**: add masked image modeling alongside DINO
- **KoLeo regularizer**: uniform feature distribution
- **Asymmetric augmentation**: stronger augmentation for student
- **Gradient accumulation**: simulate larger batch without more VRAM

## Constraints

- **Time**: fixed 5 minutes. Everything must complete in this window.
- **VRAM**: soft constraint. Some increase OK for meaningful score gains.
- **Simplicity**: simpler is better when scores are close. Removing complexity for equal score = win.

## NEVER STOP

Once the loop begins, do NOT pause to ask. The human may be asleep. Run experiments autonomously until manually stopped. If stuck, think harder — try combining ideas, read the code for angles, try radical changes.

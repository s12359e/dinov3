# Triplet-based continued SSL pretraining on DINOv3

Additive module for triplet (target, ref1, ref2) continued SSL on DINOv3 ViT-B/16 for
industrial inspection. The triplet structure supplies the positive pairs instead of only
augmented views of one image. Nothing outside `triplet_ssl/` is modified (it reuses
`dinov3.*` components and the SEM data generator read-only).

## Layout
```
triplet_ssl/
  data_gen/generate_triplets.py   # synthetic triplet generator (ref1/ref2/target + mask)
  make_mimic_checkpoint.py        # random-weight ViT-B/16 .pth (until real weights placed)
  data/triplet_dataset.py         # shared-geom/independent-photo aug, registration, oversampler
  data/synth_defect.py            # phase-3 synthetic defects + survival check
  losses/triplet_loss.py          # pairing table, robust top-k exemption, repulsion
  eval/separability.py            # per-patch residual AUROC (GT masks read HERE ONLY)
  train_triplet.py                # training loop + per-phase separability gate
  configs/{base,phase0,phase1,phase2,phase3}.yaml
```

## One-time setup
```bash
PY="C:/Users/Peter Peng/AppData/Local/Programs/Python/Python312/python.exe"   # has torch+cv2

# Mimic DINOv3 checkpoint (swap for official ViT-B/16 later — same load hook)
"$PY" triplet_ssl/make_mimic_checkpoint.py --out checkpoints/dinov3_vitb16_mimic.pth

# Synthetic triplet dataset (add --misalign_px N to exercise registration)
"$PY" triplet_ssl/data_gen/generate_triplets.py --out data/sem_triplet \
    --n_train 200 --n_val 60 --n_test 60 --defect_frac 0.5
```

## Run the phases (into ONE --out-dir so the gate reads the phase-0 baseline)
```bash
OUT=triplet_ssl/runs/exp1
"$PY" triplet_ssl/train_triplet.py --config triplet_ssl/configs/phase0.yaml --out-dir $OUT  # baseline
"$PY" triplet_ssl/train_triplet.py --config triplet_ssl/configs/phase1.yaml --out-dir $OUT  # triplet DINO
"$PY" triplet_ssl/train_triplet.py --config triplet_ssl/configs/phase2.yaml --out-dir $OUT  # + oversample + top-k
"$PY" triplet_ssl/train_triplet.py --config triplet_ssl/configs/phase3.yaml --out-dir $OUT  # + repulsion (optional)
```
Each phase writes `phaseN_metrics.json`, `phaseN_heatmaps/`, and appends a row to
`summary.tsv` (`Phase | AUROC | defect/normal ratio | guardrail | notes`). Phase >0 prints
the AUROC delta vs the phase-0 baseline and PASS/FAIL against `separability_gate`.

Useful overrides: `--checkpoint <real_dinov3.pth>`, `--num-steps`, `--batch-size`,
`--device cuda`, `--data-root`.

## Design notes / spec compliance
- **Shared geometry, independent photometrics.** One crop+flip per triplet applied to all
  three images (patch k aligns across views); brightness/contrast/gamma sampled per image.
  Vertical flip is disabled; blur/jitter kept mild for defect survival.
- **Pairing table** (`losses/triplet_loss.py`): ref↔ref full consistency (no exemption);
  target↔ref with robust top-k exemption; rotating traditional same-image pair. Weights
  `w_traditional=0.3, w_ref2ref=0.4, w_target2ref=0.3`. Computed at CLS **and** patch level.
- **Robust top-k exemption** applies to target↔ref patch loss ONLY (drops the top-k%
  highest-loss patches per image). Never on ref↔ref. `exempted_patch_map` logs which
  patches are exempted for a defective sample (sanity signal).
- **Defect oversampling** (phase 2): `DefectOversampleBatchSampler` holds defect triplets
  to `oversample.defect_frac` of each batch (5–20%).
- **Synthetic-defect repulsion** (phase 3): `L_repel = -λ·mean(patch_loss[paste_mask])`,
  plus an augmentation-survival check.
- **Stock DINO stabilisation** kept: EMA teacher, centering, temperature, cosine LR +
  warmup. Backbone LR small (`~1e-5`) with layer-wise decay; heads normal LR.
- **GT masks touch eval only.** Training uses `TripletDataset(load_masks=False)` and
  `assert_no_masks()`; `eval/separability.py` is the sole reader of `mask.png`.

## Reusing dinov3 vs self-contained
Reused read-only: `dinov3.models.vision_transformer.vit_base` (backbone),
`dinov3.layers.dino_head.DINOHead` (prototype heads), and the exact `lossfunc`/centering
recipe from `dinov3.loss.ibot_patch_loss`. The cross-entropy is re-implemented at
per-patch granularity (needed for top-k exemption + repulsion) and uses **centering**
(not Sinkhorn) so it runs single-process/CPU — dinov3's Sinkhorn path calls
`dist.all_reduce` unconditionally and `torch.compile`, which require a distributed launch.

## Status
Wired and smoke-tested end-to-end on CPU with the **mimic** (random-weight) checkpoint:
Phase 0 (frozen baseline eval) and Phase 1 (train step + re-eval) both run and produce the
AUROC/ratio/guardrail/heatmaps/summary outputs. Numbers are placeholders until (a) official
DINOv3 ViT-B/16 weights replace the mimic and (b) real triplet data replaces the synthetic
set. Both swap in without code changes (`--checkpoint`, `--data-root`).

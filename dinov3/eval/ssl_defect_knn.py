#!/usr/bin/env python3
"""
Single-image patch-level KNN evaluation for SSL defect detection.

Given an SSL-pretrained ViT, extract patch tokens from each image,
then measure how well the defect patch stands out from its neighbors
via cosine similarity.  The intuition: a good SSL model learns features
where defect patches are *dissimilar* to normal patches in the same image.

Metric
------
For each image with a defect at (x, y):
  1. Extract all patch tokens from the ViT (each patch covers patch_size x patch_size pixels).
  2. Identify the defect patch token(s) that overlap the defect bounding box.
  3. Compute cosine similarity between the defect patch and all other patches.
  4. Take the mean cosine similarity of the defect patch to its K nearest neighbors.
  5. Anomaly score = 1 - mean_knn_cosine  (higher = more anomalous = better SSL).

The final SSL index is the **mean anomaly score** across all defect images.
A higher score means the model produces features where defects are more
distinguishable from their surroundings.

Filename convention
-------------------
Images must encode defect coordinates in the filename:
    {prefix}_x{X}_y{Y}.png     e.g. sram_001_x342_y474.png
    {prefix}_x{X}_y{Y}.jpg

Where (X, Y) is the top-left corner of the defect in pixel coordinates.
Images without _x{}_y{} in the filename are treated as defect-free (skipped).

Usage
-----
    # Single GPU
    python -m dinov3.eval.ssl_defect_knn \
        --checkpoint checkpoints/dinov3_vitb16_pretrain.pth \
        --data_dir data/sem_defect/images/val \
        --patch_size 16 --img_size 512 --defect_size 4

    # With custom K
    python -m dinov3.eval.ssl_defect_knn \
        --checkpoint checkpoints/dinov3_vitb16_pretrain.pth \
        --data_dir data/sem_defect/images/val \
        --k 5 10 20
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ssl_defect_knn")


# ====================================================================
#  Filename parsing
# ====================================================================

# Matches: anything_x{digits}_y{digits}.ext
_COORD_PATTERN = re.compile(r'_x(\d+)_y(\d+)')


def parse_defect_coords(filename: str) -> Optional[Tuple[int, int]]:
    """Extract (x, y) defect coordinates from filename.

    Returns None if the filename does not contain _x{}_y{} pattern.
    """
    match = _COORD_PATTERN.search(filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


# ====================================================================
#  Feature extraction
# ====================================================================

def extract_patch_tokens(model, image_tensor, use_cls=False):
    """Extract patch-level feature tokens from a ViT model.

    Args:
        model: DINOv3 ViT model with get_intermediate_layers or forward_features.
        image_tensor: (1, 3, H, W) normalized image tensor.
        use_cls: if True, also return the CLS token.

    Returns:
        patch_tokens: (N_patches, D) tensor of patch features.
        cls_token: (1, D) CLS token if use_cls=True, else None.
    """
    with torch.no_grad():
        # Try get_intermediate_layers first (DINOv3 API)
        if hasattr(model, 'get_intermediate_layers'):
            outputs = model.get_intermediate_layers(
                image_tensor, n=[len(model.blocks) - 1],
                return_class_token=True,
            )
            patch_tokens, cls_token = outputs[0]  # last layer
        elif hasattr(model, 'forward_features'):
            out = model.forward_features(image_tensor)
            if isinstance(out, dict):
                patch_tokens = out.get('x_norm_patchtokens', out.get('x_patchtokens'))
                cls_token = out.get('x_norm_clstoken', out.get('x_clstoken'))
            else:
                # Assume (B, N+1, D) with CLS at position 0
                cls_token = out[:, :1, :]
                patch_tokens = out[:, 1:, :]
        else:
            raise ValueError("Model must have get_intermediate_layers or forward_features")

    # Squeeze batch dim
    patch_tokens = patch_tokens.squeeze(0)  # (N_patches, D)
    if cls_token is not None:
        cls_token = cls_token.squeeze(0)    # (1, D) or (D,)
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)

    # L2 normalize
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    if use_cls and cls_token is not None:
        cls_token = F.normalize(cls_token, dim=-1)

    return patch_tokens, cls_token if use_cls else None


def pixel_to_patch_indices(x, y, defect_w, defect_h, patch_size, grid_w, grid_h):
    """Convert pixel-level defect bbox to patch-level indices.

    Args:
        x, y: top-left pixel coordinates of defect.
        defect_w, defect_h: defect size in pixels.
        patch_size: ViT patch size.
        grid_w, grid_h: number of patches in W and H dimensions.

    Returns:
        List of patch indices (flattened row-major) that overlap the defect.
    """
    # Patch grid coordinates that overlap with the defect bbox
    p_x_start = max(0, x // patch_size)
    p_x_end = min(grid_w, (x + defect_w - 1) // patch_size + 1)
    p_y_start = max(0, y // patch_size)
    p_y_end = min(grid_h, (y + defect_h - 1) // patch_size + 1)

    indices = []
    for py in range(p_y_start, p_y_end):
        for px in range(p_x_start, p_x_end):
            indices.append(py * grid_w + px)
    return indices


# ====================================================================
#  Single-image KNN anomaly score
# ====================================================================

def compute_single_image_knn_score(
    patch_tokens: torch.Tensor,
    defect_patch_indices: List[int],
    ks: List[int],
) -> Dict[str, float]:
    """Compute KNN-based anomaly score for defect patches within one image.

    Args:
        patch_tokens: (N, D) L2-normalized patch features.
        defect_patch_indices: indices of patches overlapping the defect.
        ks: list of K values for KNN.

    Returns:
        Dict with scores for each K:
            knn_{k}_cosine: mean cosine similarity to K nearest neighbors
            knn_{k}_anomaly: 1 - cosine (higher = more anomalous = better)
    """
    N = patch_tokens.shape[0]
    if not defect_patch_indices:
        return {}

    # Cosine similarity matrix: defect patches vs all patches
    defect_feats = patch_tokens[defect_patch_indices]  # (M, D)
    sim_matrix = torch.mm(defect_feats, patch_tokens.T)  # (M, N)

    # Mask out self-similarity (set to -inf so they're not selected as neighbors)
    for i, idx in enumerate(defect_patch_indices):
        sim_matrix[i, idx] = -1.0
        # Also mask other defect patches to measure distance to NORMAL patches
        for other_idx in defect_patch_indices:
            if other_idx != idx:
                sim_matrix[i, other_idx] = -1.0

    results = {}
    for k in ks:
        actual_k = min(k, N - len(defect_patch_indices))
        if actual_k <= 0:
            continue
        topk_sims, _ = sim_matrix.topk(actual_k, dim=1, largest=True)
        mean_cosine = topk_sims.mean().item()
        results[f"knn_{k}_cosine"] = mean_cosine
        results[f"knn_{k}_anomaly"] = 1.0 - mean_cosine

    return results


# ====================================================================
#  Dataset evaluation
# ====================================================================

def evaluate_ssl_defect_knn(
    model: nn.Module,
    data_dir: str,
    img_size: int = 512,
    patch_size: int = 16,
    defect_size: int = 4,
    ks: List[int] = None,
    mean: Tuple[float, ...] = (109.65, 104.81, 75.48),
    std: Tuple[float, ...] = (54.32, 39.78, 36.47),
    device: torch.device = None,
    output_path: str = None,
) -> Dict[str, float]:
    """Evaluate SSL model quality using single-image patch KNN on defect images.

    Args:
        model: ViT backbone (e.g. DinoVisionTransformer).
        data_dir: directory with images. Filenames must contain _x{}_y{}.
        img_size: resize images to this size.
        patch_size: ViT patch size.
        defect_size: defect bounding box side length in pixels.
        ks: list of K values for KNN.
        mean, std: normalization constants (0-255 scale).
        device: torch device.
        output_path: if set, save per-image results to this JSON file.

    Returns:
        Dict with aggregated scores (mean across all defect images).
    """
    if ks is None:
        ks = [5, 10, 20]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    grid_h = img_size // patch_size
    grid_w = img_size // patch_size

    # Collect image files with defect coordinates
    data_path = Path(data_dir)
    image_files = sorted([
        f for f in data_path.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    ])

    defect_images = []
    for f in image_files:
        coords = parse_defect_coords(f.stem)
        if coords is not None:
            defect_images.append((f, coords))

    if not defect_images:
        logger.warning(f"No images with _x{{}}_y{{}} coordinates found in {data_dir}")
        logger.info("Filename format: prefix_x123_y456.png")
        return {}

    logger.info(f"Found {len(defect_images)} defect images (out of {len(image_files)} total)")

    # Accumulate scores
    all_scores = {f"knn_{k}_cosine": [] for k in ks}
    all_scores.update({f"knn_{k}_anomaly": [] for k in ks})
    per_image_results = []

    t0 = time.time()
    for i, (img_path, (def_x, def_y)) in enumerate(defect_images):
        # Load and preprocess
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning(f"Cannot read {img_path}, skipping")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Scale defect coordinates if image is resized
        orig_h, orig_w = image.shape[:2]
        scale_x = img_size / orig_w
        scale_y = img_size / orig_h
        scaled_def_x = int(def_x * scale_x)
        scaled_def_y = int(def_y * scale_y)
        scaled_def_w = max(1, int(defect_size * scale_x))
        scaled_def_h = max(1, int(defect_size * scale_y))

        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # Normalize
        image_f = (image.astype(np.float32) - mean_np) / std_np
        image_t = torch.from_numpy(image_f).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Extract features
        patch_tokens, _ = extract_patch_tokens(model, image_t)

        # Find defect patch indices
        defect_indices = pixel_to_patch_indices(
            scaled_def_x, scaled_def_y,
            scaled_def_w, scaled_def_h,
            patch_size, grid_w, grid_h,
        )

        if not defect_indices:
            logger.warning(f"Defect at ({def_x},{def_y}) maps to no patches, skipping {img_path.name}")
            continue

        # Compute KNN score
        scores = compute_single_image_knn_score(patch_tokens, defect_indices, ks)

        img_result = {
            "filename": img_path.name,
            "defect_x": def_x,
            "defect_y": def_y,
            "defect_patches": defect_indices,
            "num_patches": patch_tokens.shape[0],
        }
        img_result.update(scores)
        per_image_results.append(img_result)

        for key, val in scores.items():
            if key in all_scores:
                all_scores[key].append(val)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(defect_images)} images")

    elapsed = time.time() - t0
    logger.info(f"Evaluated {len(per_image_results)} images in {elapsed:.1f}s")

    # Aggregate
    aggregated = {}
    best_anomaly_score = 0.0
    best_k = ks[0]

    for key, values in all_scores.items():
        if values:
            mean_val = sum(values) / len(values)
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            aggregated[f"mean_{key}"] = mean_val
            aggregated[f"std_{key}"] = std_val

            # Track best anomaly score
            if "anomaly" in key and mean_val > best_anomaly_score:
                best_anomaly_score = mean_val
                best_k = int(key.split("_")[1])

    aggregated["best_anomaly_score"] = best_anomaly_score
    aggregated["best_k"] = best_k
    aggregated["num_images"] = len(per_image_results)

    # Log results
    logger.info("=" * 60)
    logger.info("SSL Defect KNN Evaluation Results")
    logger.info("=" * 60)
    for k in ks:
        cos_key = f"mean_knn_{k}_cosine"
        ano_key = f"mean_knn_{k}_anomaly"
        if cos_key in aggregated:
            logger.info(
                f"  K={k:3d} | cosine={aggregated[cos_key]:.4f} | "
                f"anomaly={aggregated[ano_key]:.4f} (+/- {aggregated[f'std_knn_{k}_anomaly']:.4f})"
            )
    logger.info(f"  Best anomaly score: {best_anomaly_score:.4f} (K={best_k})")
    logger.info("=" * 60)

    # Save results
    if output_path:
        output = {
            "aggregated": aggregated,
            "per_image": per_image_results,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return aggregated


# ====================================================================
#  Convenience: rename images with defect coords from metadata
# ====================================================================

def rename_images_with_coords(
    images_dir: str,
    metadata_path: str,
    split: str = "val",
    region: str = None,
    dry_run: bool = True,
):
    """Rename images to include defect coordinates from metadata.json.

    Converts: sram_val_00001.png -> sram_val_00001_x342_y474.png

    Args:
        images_dir: directory with images.
        metadata_path: path to metadata.json from generate_sem_dataset.py.
        split: which split to rename (train/val/test).
        region: filter by region (sram/logic), or None for all.
        dry_run: if True, only print what would be renamed.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    images_path = Path(images_dir)
    renamed = 0

    for meta_key, entries in metadata.items():
        if split and split not in meta_key:
            continue
        if region and region not in meta_key:
            continue

        for entry in entries:
            if not entry.get("has_defect") or not entry.get("defect_bbox"):
                continue
            bbox = entry["defect_bbox"]
            old_name = entry["filename"]
            new_name = f"{old_name}_x{bbox['x']}_y{bbox['y']}"

            old_path = images_path / f"{old_name}.png"
            new_path = images_path / f"{new_name}.png"

            if not old_path.exists():
                # Maybe already renamed
                if new_path.exists():
                    continue
                logger.warning(f"  Not found: {old_path}")
                continue

            if dry_run:
                print(f"  {old_path.name} -> {new_path.name}")
            else:
                old_path.rename(new_path)
            renamed += 1

    action = "Would rename" if dry_run else "Renamed"
    logger.info(f"{action} {renamed} files")
    if dry_run:
        logger.info("Run with --no_dry_run to actually rename")


# ====================================================================
#  CLI
# ====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SSL Defect KNN Evaluation — single-image patch cosine similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate
  python -m dinov3.eval.ssl_defect_knn \\
      --checkpoint checkpoints/dinov3_vitb16_pretrain.pth \\
      --data_dir data/sem_defect/images/val

  # Rename files to include defect coordinates
  python -m dinov3.eval.ssl_defect_knn --rename \\
      --data_dir data/sem_defect/images/val \\
      --metadata data/sem_defect/metadata.json \\
      --no_dry_run
""",
    )

    # Mode
    p.add_argument("--rename", action="store_true",
                   help="Rename images to include defect coords from metadata.json")

    # Eval args
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to DINOv3 pretrained ViT weights")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with images (filenames must contain _x{}_y{})")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--defect_size", type=int, default=4,
                   help="Defect bounding box side length in pixels")
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20],
                   help="K values for KNN")
    p.add_argument("--img_mean", type=float, nargs=3,
                   default=[109.65, 104.81, 75.48])
    p.add_argument("--img_std", type=float, nargs=3,
                   default=[54.32, 39.78, 36.47])
    p.add_argument("--output", type=str, default="work_dirs/ssl_knn_results.json",
                   help="Path to save detailed results JSON")

    # Rename args
    p.add_argument("--metadata", type=str, default="",
                   help="Path to metadata.json (for --rename mode)")
    p.add_argument("--split", type=str, default="val",
                   help="Split to rename (for --rename mode)")
    p.add_argument("--region", type=str, default=None,
                   choices=["sram", "logic"],
                   help="Filter by region (for --rename mode)")
    p.add_argument("--no_dry_run", action="store_true",
                   help="Actually rename files (default: dry run)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.rename:
        if not args.metadata:
            logger.error("--metadata is required for --rename mode")
            return
        rename_images_with_coords(
            images_dir=args.data_dir,
            metadata_path=args.metadata,
            split=args.split,
            region=args.region,
            dry_run=not args.no_dry_run,
        )
        return

    # -- Build model --------------------------------------------------------
    from dinov3.models.vision_transformer import DinoVisionTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DinoVisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading weights from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt
        for key in ("model", "teacher", "state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        msg = model.load_state_dict(state, strict=False)
        logger.info(f"  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    else:
        logger.warning("No checkpoint loaded — using random weights!")

    # -- Evaluate -----------------------------------------------------------
    results = evaluate_ssl_defect_knn(
        model=model,
        data_dir=args.data_dir,
        img_size=args.img_size,
        patch_size=args.patch_size,
        defect_size=args.defect_size,
        ks=args.k,
        mean=tuple(args.img_mean),
        std=tuple(args.img_std),
        device=device,
        output_path=args.output,
    )

    # Print final SSL index
    if results:
        print(f"\n*** SSL Index (best anomaly score): {results['best_anomaly_score']:.4f} ***")


if __name__ == "__main__":
    main()

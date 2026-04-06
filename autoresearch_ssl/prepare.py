"""
Fixed evaluation and data utilities for DINOv3 SSL autoresearch.
DO NOT MODIFY. The AI agent only modifies train.py.

This provides:
  - evaluate_ssl_knn(): the ground truth metric (defect KNN anomaly score)
  - Data loading utilities
  - Constants
"""

import json
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

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
IMG_SIZE = 512             # image size
PATCH_SIZE = 16            # ViT patch size
DEFECT_SIZE = 4            # defect bounding box side length
KNN_KS = [5, 10, 20]      # K values for KNN evaluation
IMG_MEAN = (109.65, 104.81, 75.48)  # normalization (0-255 scale)
IMG_STD = (54.32, 39.78, 36.47)

# ---------------------------------------------------------------------------
# Filename coordinate parsing
# ---------------------------------------------------------------------------

_COORD_PATTERN = re.compile(r'_x(\d+)_y(\d+)')


def parse_defect_coords(filename: str) -> Optional[Tuple[int, int]]:
    """Extract (x, y) defect coordinates from filename."""
    match = _COORD_PATTERN.search(filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


# ---------------------------------------------------------------------------
# Rename utility (run once to prepare eval images)
# ---------------------------------------------------------------------------

def rename_images_with_coords(images_dir, metadata_path, split="val"):
    """Rename images to include defect coordinates from metadata.json.

    Converts: sram_val_00001.png -> sram_val_00001_x342_y474.png
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    images_path = Path(images_dir)
    renamed = 0

    for meta_key, entries in metadata.items():
        if split not in meta_key:
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
                continue
            old_path.rename(new_path)
            renamed += 1

    print(f"Renamed {renamed} files in {images_dir}")


# ---------------------------------------------------------------------------
# Feature extraction from ViT
# ---------------------------------------------------------------------------

def extract_patch_tokens(model, image_tensor):
    """Extract patch-level feature tokens from a ViT model.

    Args:
        model: DINOv3 ViT with get_intermediate_layers or forward_features.
        image_tensor: (1, 3, H, W) normalized image tensor.

    Returns:
        patch_tokens: (N_patches, D) L2-normalized patch features.
    """
    with torch.no_grad():
        if hasattr(model, 'get_intermediate_layers'):
            outputs = model.get_intermediate_layers(
                image_tensor, n=[len(model.blocks) - 1],
                return_class_token=True,
            )
            patch_tokens, _ = outputs[0]
        elif hasattr(model, 'forward_features'):
            out = model.forward_features(image_tensor)
            if isinstance(out, dict):
                patch_tokens = out.get('x_norm_patchtokens',
                                       out.get('x_patchtokens'))
            else:
                patch_tokens = out[:, 1:, :]
        else:
            raise ValueError("Model needs get_intermediate_layers or forward_features")

    patch_tokens = patch_tokens.squeeze(0)
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    return patch_tokens


def pixel_to_patch_indices(x, y, defect_w, defect_h, patch_size, grid_w, grid_h):
    """Convert pixel bbox to patch indices."""
    p_x_start = max(0, x // patch_size)
    p_x_end = min(grid_w, (x + defect_w - 1) // patch_size + 1)
    p_y_start = max(0, y // patch_size)
    p_y_end = min(grid_h, (y + defect_h - 1) // patch_size + 1)

    indices = []
    for py in range(p_y_start, p_y_end):
        for px in range(p_x_start, p_x_end):
            indices.append(py * grid_w + px)
    return indices


def compute_single_image_knn_score(patch_tokens, defect_patch_indices, ks):
    """Compute KNN anomaly score for defect patches within one image.

    anomaly = 1 - mean_cosine_to_K_nearest_normal_neighbors
    Higher = defect more distinguishable = better SSL features.
    """
    N = patch_tokens.shape[0]
    if not defect_patch_indices:
        return {}

    defect_feats = patch_tokens[defect_patch_indices]
    sim_matrix = torch.mm(defect_feats, patch_tokens.T)

    # Mask self and other defect patches
    for i, idx in enumerate(defect_patch_indices):
        sim_matrix[i, idx] = -1.0
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


# ---------------------------------------------------------------------------
# Main evaluation function (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ssl_knn(model, eval_dir, device=None):
    """Evaluate SSL model quality using single-image patch KNN.

    This is the ground truth metric. DO NOT MODIFY.

    Args:
        model: ViT backbone (must support get_intermediate_layers or forward_features).
        eval_dir: directory with images containing _x{}_y{} in filenames.
        device: torch device.

    Returns:
        best_anomaly_score (float): the SSL index. Higher is better.
        results_dict (dict): detailed results.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    mean_np = np.array(IMG_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.array(IMG_STD, dtype=np.float32).reshape(1, 1, 3)
    grid_h = IMG_SIZE // PATCH_SIZE
    grid_w = IMG_SIZE // PATCH_SIZE

    data_path = Path(eval_dir)
    image_files = sorted([
        f for f in data_path.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')
    ])

    defect_images = []
    for f in image_files:
        coords = parse_defect_coords(f.stem)
        if coords is not None:
            defect_images.append((f, coords))

    if not defect_images:
        print(f"[EVAL] No images with _x{{}}_y{{}} found in {eval_dir}")
        return 0.0, {}

    all_anomaly = {f"knn_{k}_anomaly": [] for k in KNN_KS}

    for img_path, (def_x, def_y) in defect_images:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        scale_x = IMG_SIZE / orig_w
        scale_y = IMG_SIZE / orig_h
        scaled_x = int(def_x * scale_x)
        scaled_y = int(def_y * scale_y)
        scaled_w = max(1, int(DEFECT_SIZE * scale_x))
        scaled_h = max(1, int(DEFECT_SIZE * scale_y))

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        image_f = (image.astype(np.float32) - mean_np) / std_np
        image_t = torch.from_numpy(image_f).permute(2, 0, 1).unsqueeze(0).float().to(device)

        patch_tokens = extract_patch_tokens(model, image_t)
        defect_indices = pixel_to_patch_indices(
            scaled_x, scaled_y, scaled_w, scaled_h,
            PATCH_SIZE, grid_w, grid_h,
        )

        if not defect_indices:
            continue

        scores = compute_single_image_knn_score(patch_tokens, defect_indices, KNN_KS)
        for key, val in scores.items():
            if key in all_anomaly:
                all_anomaly[key].append(val)

    # Aggregate
    results = {}
    best_score = 0.0
    for key, values in all_anomaly.items():
        if values:
            mean_val = sum(values) / len(values)
            results[f"mean_{key}"] = mean_val
            if mean_val > best_score:
                best_score = mean_val

    results["best_anomaly_score"] = best_score
    results["num_images"] = len(defect_images)

    return best_score, results


# ---------------------------------------------------------------------------
# Main (data preparation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare eval data for SSL autoresearch")
    parser.add_argument("--eval-dir", type=str, required=True,
                        help="Directory with eval images")
    parser.add_argument("--metadata", type=str, default="",
                        help="metadata.json path — rename images to include defect coords")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    if args.metadata:
        print("Renaming images to include defect coordinates...")
        rename_images_with_coords(args.eval_dir, args.metadata, args.split)
    else:
        # Verify images have coordinates
        data_path = Path(args.eval_dir)
        count = sum(1 for f in data_path.iterdir()
                    if parse_defect_coords(f.stem) is not None)
        total = sum(1 for f in data_path.iterdir()
                    if f.suffix.lower() in ('.png', '.jpg'))
        print(f"Found {count}/{total} images with _x{{}}_y{{}} coordinates")
        if count == 0:
            print("Run with --metadata to rename images first")

    print("Done! Ready for autoresearch.")

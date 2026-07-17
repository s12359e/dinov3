"""Point-labeled TIFF validation using the production inference score map.

Labels are encoded in the final part of each filename stem as ``#x,y``.  The
public API converts labels to zero-based pixel coordinates, ranks individual
score-grid cells (without NMS), and matches their clipped patch centres to the
single ground-truth point in each image.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from triplet_ssl.infer import score_map


_POINT_LABEL_RE = re.compile(r"^[^#]+#([0-9]+),([0-9]+)$")
_TIFF_SUFFIXES = {".tif", ".tiff"}


def _validate_coordinate_base(coordinate_base: int) -> None:
    if isinstance(coordinate_base, bool) or coordinate_base not in (0, 1):
        raise ValueError("coordinate_base must be either 0 or 1")


def parse_point_label(path: str | Path, coordinate_base: int = 0) -> tuple[int, int]:
    """Parse the one ``#x,y`` suffix and return zero-based ``(x, y)``.

    The complete stem must contain exactly one ``#`` and end immediately after
    the two unsigned integer coordinates.  Bounds are image-dependent and are
    therefore checked by :func:`evaluate_labeled_tiffs`.
    """

    _validate_coordinate_base(coordinate_base)
    path = Path(path)
    if path.suffix.lower() not in _TIFF_SUFFIXES:
        raise ValueError(f"point label must belong to a .tif/.tiff file: {path.name!r}")
    match = _POINT_LABEL_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"TIFF stem must end in exactly one strict #x,y label: {path.stem!r}"
        )
    x, y = (int(value) - coordinate_base for value in match.groups())
    return x, y


def _labeled_tiff_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        files = [root] if root.suffix.lower() in _TIFF_SUFFIXES else []
    elif root.is_dir():
        files = [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in _TIFF_SUFFIXES
        ]
    else:
        raise ValueError(f"validation path does not exist: {root}")
    files.sort(key=lambda path: (path.name.casefold(), path.name))
    if not files:
        raise ValueError(f"validation requires at least one .tif/.tiff image under {root}")
    return files


def _module_eval_states(model: Any, fusion_head: Any) -> list[tuple[Any, bool]]:
    """Put distinct modules in eval mode and retain their original modes."""

    states: list[tuple[Any, bool]] = []
    seen: set[int] = set()
    try:
        for name, module in (("model", model), ("fusion_head", fusion_head)):
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            if not hasattr(module, "training") or not callable(getattr(module, "eval", None)):
                raise TypeError(f"{name} must be a module with training state and eval()")
            states.append((module, bool(module.training)))
            module.eval()
    except Exception:
        for module, was_training in reversed(states):
            module.train(was_training)
        raise
    return states


def _as_numpy_scores(scores: Any, path: Path) -> np.ndarray:
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError(f"score map for {path.name!r} must be a non-empty 2-D array")
    if not np.isfinite(scores).all():
        raise ValueError(f"score map for {path.name!r} contains NaN or Inf")
    return scores


def _unpack_score_result(result: Any, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise ValueError(
            f"score function for {path.name!r} must return (scores, aux, (H, W), target)"
        )
    scores = _as_numpy_scores(result[0], path)
    image_hw = result[2]
    if not isinstance(image_hw, (tuple, list)) or len(image_hw) != 2:
        raise ValueError(f"score function for {path.name!r} returned invalid image size")
    height, width = image_hw
    if isinstance(height, bool) or isinstance(width, bool):
        raise ValueError(f"score function for {path.name!r} returned invalid image size")
    try:
        height, width = int(height), int(width)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"score function for {path.name!r} returned invalid image size"
        ) from error
    if height <= 0 or width <= 0:
        raise ValueError(f"score function for {path.name!r} returned invalid image size")
    return scores, (height, width)


def _ranked_points(
    scores: np.ndarray,
    image_hw: tuple[int, int],
    patch: int,
    count: int,
    ground_truth: tuple[int, int],
    match_radius_px: float,
) -> list[dict[str, Any]]:
    height, width = image_hw
    # Stable sorting gives row-major tie-breaking because flatten() is row-major.
    order = np.argsort(-scores.reshape(-1), kind="stable")[: min(count, scores.size)]
    points: list[dict[str, Any]] = []
    gt_x, gt_y = ground_truth
    for rank, flat_index in enumerate(order, start=1):
        grid_y, grid_x = np.unravel_index(int(flat_index), scores.shape)
        x = min((int(grid_x) + 0.5) * patch, float(width - 1))
        y = min((int(grid_y) + 0.5) * patch, float(height - 1))
        distance = math.hypot(x - gt_x, y - gt_y)
        points.append(
            {
                "rank": int(rank),
                "score": float(scores[grid_y, grid_x]),
                "grid_x": int(grid_x),
                "grid_y": int(grid_y),
                "x": float(x),
                "y": float(y),
                "distance_px": float(distance),
                "matched": bool(distance < match_radius_px),
            }
        )
    return points


def evaluate_labeled_tiffs(
    model: Any,
    fusion_head: Any,
    root: str | Path,
    device: Any,
    *,
    score_kwargs: Mapping[str, Any],
    top_k: int = 5,
    match_radius_px: float = 15.0,
    coordinate_base: int = 0,
    score_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one point-labeled defect per TIFF through ``infer.score_map``.

    ``score_fn`` is an injectable function with the same call contract as
    :func:`triplet_ssl.infer.score_map`; it exists so ranking and matching can be
    unit-tested without decoding images or running a backbone.
    """

    _validate_coordinate_base(coordinate_base)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    try:
        match_radius_px = float(match_radius_px)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("match_radius_px must be a finite positive number") from error
    if not math.isfinite(match_radius_px) or match_radius_px <= 0:
        raise ValueError("match_radius_px must be a finite positive number")
    if score_kwargs is None:
        raise TypeError("score_kwargs must be a mapping")
    call_kwargs = dict(score_kwargs)
    if "fusion_head" in call_kwargs and call_kwargs["fusion_head"] is not fusion_head:
        raise ValueError("fusion_head must be supplied through the dedicated argument")
    call_kwargs["fusion_head"] = fusion_head
    patch = call_kwargs.get("patch", 16)
    if isinstance(patch, bool) or not isinstance(patch, int) or patch <= 0:
        raise ValueError("score_kwargs['patch'] must be a positive integer")
    files = _labeled_tiff_files(root)
    scoring_function = score_map if score_fn is None else score_fn

    predictions: list[dict[str, Any]] = []
    states: list[tuple[Any, bool]] = []
    try:
        states = _module_eval_states(model, fusion_head)
        for path in files:
            gt_x, gt_y = parse_point_label(path, coordinate_base=coordinate_base)
            result = scoring_function(model, path, device, **call_kwargs)
            scores, image_hw = _unpack_score_result(result, path)
            height, width = image_hw
            if not (0 <= gt_x < width and 0 <= gt_y < height):
                raise ValueError(
                    f"ground-truth point ({gt_x}, {gt_y}) for {path.name!r} is "
                    f"outside zero-based image bounds width={width}, height={height}"
                )
            expected_shape = (
                (height + patch - 1) // patch,
                (width + patch - 1) // patch,
            )
            if scores.shape != expected_shape:
                raise ValueError(
                    f"score map for {path.name!r} has shape {scores.shape}, "
                    f"expected {expected_shape} for image {image_hw} and patch={patch}"
                )

            ranked = _ranked_points(
                scores,
                image_hw,
                patch,
                max(top_k, 5),
                (gt_x, gt_y),
                match_radius_px,
            )
            top_points = ranked[:top_k]
            top_five = ranked[:5]
            matched_rank = next(
                (point["rank"] for point in top_points if point["matched"]), None
            )
            predictions.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "ground_truth": {"x": int(gt_x), "y": int(gt_y)},
                    "image_size": {"height": int(height), "width": int(width)},
                    "score_grid_shape": [int(scores.shape[0]), int(scores.shape[1])],
                    "top_points": top_points,
                    "top1_hit": bool(top_points[0]["matched"]),
                    "top_k_hit": bool(matched_rank is not None),
                    "top5_hit": bool(any(point["matched"] for point in top_five)),
                    "matched_rank": int(matched_rank) if matched_rank is not None else None,
                    "min_distance_px": float(
                        min(point["distance_px"] for point in top_points)
                    ),
                }
            )
    finally:
        for module, was_training in reversed(states):
            module.train(was_training)

    n_images = len(predictions)
    top_k_hits = sum(item["top_k_hit"] for item in predictions)
    top5_hits = sum(item["top5_hit"] for item in predictions)
    top1_hits = sum(item["top1_hit"] for item in predictions)
    matched_ranks = [
        item["matched_rank"] for item in predictions if item["matched_rank"] is not None
    ]
    mean_min_distance = sum(item["min_distance_px"] for item in predictions) / n_images
    summary = {
        "n_images": int(n_images),
        "top_k": int(top_k),
        "match_radius_px": float(match_radius_px),
        "coordinate_base": int(coordinate_base),
        "top_k_hit_count": int(top_k_hits),
        "top_k_hit_rate": float(top_k_hits / n_images),
        "topk_hit_rate": float(top_k_hits / n_images),
        "top5_hit_count": int(top5_hits),
        "top5_hit_rate": float(top5_hits / n_images),
        "top1_hit_count": int(top1_hits),
        "top1_hit_rate": float(top1_hits / n_images),
        "mean_min_distance": float(mean_min_distance),
        "mean_matched_rank": (
            float(sum(matched_ranks) / len(matched_ranks)) if matched_ranks else None
        ),
        "matched_images": int(len(matched_ranks)),
    }
    return {"summary": summary, "predictions": predictions}


def validation_selection_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return the deterministic lexicographic key used to select checkpoints.

    Higher is better: top-k hit rate, then top-1 hit rate, then lower mean
    minimum distance.  Either the full evaluator result or its summary is
    accepted.
    """

    summary = metrics.get("summary", metrics)
    if not isinstance(summary, Mapping):
        raise TypeError("metrics must be a summary mapping or contain one")
    if "top_k_hit_rate" in summary:
        top_k_hit_rate = summary["top_k_hit_rate"]
    elif "topk_hit_rate" in summary:
        top_k_hit_rate = summary["topk_hit_rate"]
    else:
        top_k_hit_rate = summary["top5_hit_rate"]
    values = (
        float(top_k_hit_rate),
        float(summary["top1_hit_rate"]),
        float(summary["mean_min_distance"]),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("selection metrics must be finite")
    return values[0], values[1], -values[2]

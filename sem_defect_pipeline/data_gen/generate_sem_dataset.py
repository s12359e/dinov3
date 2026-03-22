"""
Synthetic SEM (Scanning Electron Microscope) image generator.

Topology rules
--------------
Metal Gate  : Vertical stripes, X-axis period = 22 px.
              Brightest component  → pixel value ~ 230–250.

PEPI band   : Horizontal regions alternating with NEPI.
              Second brightest     → pixel value ~  160–190.

NEPI band   : Between PEPI bands.
              Third brightest (darkest) → pixel value ~  60–90.

Defect      : 4×4 square extruding from the RIGHT edge of a Metal Gate
              into a NEPI region.  Same brightness as the Metal Gate.
              The defect LEFT column shares the gate's right edge column,
              making it physically attached.

Noise model : (1) slight Gaussian blur (PSF simulation), then
              (2) additive Gaussian noise (shot + readout noise).

Output
------
images/  — 3-channel uint8 PNG (grayscale replicated to R=G=B)
masks/   — single-channel uint8 PNG  (0=background, 1=defect)

Usage
-----
    python generate_sem_dataset.py                   # default: 800 train / 100 val / 100 test
    python generate_sem_dataset.py --n_train 2000 --seed 0
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core image generator
# ─────────────────────────────────────────────────────────────────────────────

class SEMImageGenerator:
    """
    Generates a single synthetic SEM image with an optional metal-extrusion defect.

    Parameters
    ----------
    width, height     : image dimensions (default 512×512).
    gate_period       : X-axis pitch of metal gates in pixels (default 22).
    gate_width        : width of each metal gate stripe in pixels (default 7).
                        Should satisfy gate_width < gate_period.
    band_height       : height of each PEPI or NEPI horizontal band (default 64).
                        PEPI and NEPI alternate, so one full period = 2×band_height.
    gate_brightness   : mean pixel value for Metal Gate and defect (default 240).
    pepi_brightness   : mean pixel value for PEPI regions (default 175).
    nepi_brightness   : mean pixel value for NEPI regions (default 75).
    defect_size       : side length of the square defect in pixels (default 4 → 4×4=16 px).
    noise_std         : std-dev of additive Gaussian noise (default 12).
    blur_sigma        : sigma for Gaussian blur PSF (default 0.7).
    brightness_jitter : ±jitter applied to region base brightnesses per image (default 8).
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        gate_period: int = 22,
        gate_width: int = 7,
        band_height: int = 64,
        gate_brightness: int = 240,
        pepi_brightness: int = 175,
        nepi_brightness: int = 75,
        defect_size: int = 4,
        noise_std: float = 12.0,
        blur_sigma: float = 0.7,
        brightness_jitter: int = 8,
    ):
        assert gate_width < gate_period, "gate_width must be less than gate_period"
        assert defect_size >= 1

        self.width = width
        self.height = height
        self.gate_period = gate_period
        self.gate_width = gate_width
        self.band_height = band_height
        self.gate_brightness = gate_brightness
        self.pepi_brightness = pepi_brightness
        self.nepi_brightness = nepi_brightness
        self.defect_size = defect_size
        self.noise_std = noise_std
        self.blur_sigma = blur_sigma
        self.brightness_jitter = brightness_jitter

    # ── helpers ───────────────────────────────────────────────────────────────

    def _jitter(self, base: int, rng: np.random.Generator) -> int:
        """Per-image brightness jitter to simulate SEM beam-current variation."""
        delta = int(rng.integers(-self.brightness_jitter, self.brightness_jitter + 1))
        return int(np.clip(base + delta, 0, 255))

    def _gate_columns(self, phase_offset: int):
        """
        Return a list of (x_start, x_end) column ranges for all gates.

        phase_offset: random X shift so gates don't always start at column 0.
        """
        gates = []
        half = self.gate_width // 2
        x = phase_offset
        while x < self.width + self.gate_period:
            x_start = max(0, x - half)
            x_end   = min(self.width, x + self.gate_width - half)
            if x_start < self.width:
                gates.append((x_start, x_end))
            x += self.gate_period
        return gates

    def _band_regions(self):
        """
        Return (y_start, y_end, kind) for each horizontal band.
        kind ∈ {'PEPI', 'NEPI'}.
        """
        bands = []
        y = 0
        kind_cycle = ['PEPI', 'NEPI']
        i = 0
        while y < self.height:
            y_end = min(y + self.band_height, self.height)
            bands.append((y, y_end, kind_cycle[i % 2]))
            y = y_end
            i += 1
        return bands

    def _find_defect_candidates(self, gates, bands):
        """
        Find all valid (defect_x, defect_y) positions where:
          - The defect LEFT edge is immediately to the right of a gate.
          - The defect lies fully within a NEPI band (with a 3-px margin).
          - The defect fits within image boundaries.
        """
        ds = self.defect_size
        margin = 3  # keep defect away from band edges for clean labels

        nepi_bands = [(ys, ye) for ys, ye, kind in bands if kind == 'NEPI']
        candidates = []

        for (x_start, x_end) in gates:
            defect_x = x_end          # physically attached to gate right edge
            if defect_x + ds > self.width:
                continue
            for (ys, ye) in nepi_bands:
                y_lo = ys + margin
                y_hi = ye - ds - margin
                if y_hi > y_lo:
                    candidates.append((defect_x, y_lo, y_hi))

        return candidates

    # ── main generation method ────────────────────────────────────────────────

    def generate(
        self,
        with_defect: bool = True,
        rng: np.random.Generator = None,
    ):
        """
        Generate one SEM image + binary mask.

        Parameters
        ----------
        with_defect : whether to place a defect (default True).
        rng         : numpy Generator for reproducibility.

        Returns
        -------
        image_rgb : np.ndarray, shape (H, W, 3), dtype uint8
                    Grayscale values replicated to 3 channels.
        mask      : np.ndarray, shape (H, W),    dtype uint8
                    0 = background, 1 = defect.
        meta      : dict with generation metadata (gate positions, defect bbox, etc.)
        """
        if rng is None:
            rng = np.random.default_rng()

        W, H = self.width, self.height
        ds = self.defect_size

        # Per-image brightness jitter
        gate_val = self._jitter(self.gate_brightness, rng)
        pepi_val = self._jitter(self.pepi_brightness, rng)
        nepi_val = self._jitter(self.nepi_brightness, rng)

        # ── 1. Horizontal bands (PEPI / NEPI) ────────────────────────────────
        canvas = np.zeros((H, W), dtype=np.float32)
        bands  = self._band_regions()
        for (ys, ye, kind) in bands:
            canvas[ys:ye, :] = pepi_val if kind == 'PEPI' else nepi_val

        # ── 2. Metal Gates (vertical stripes) ────────────────────────────────
        phase_offset = int(rng.integers(0, self.gate_period))
        gates = self._gate_columns(phase_offset)
        for (xs, xe) in gates:
            canvas[:, xs:xe] = gate_val

        # ── 3. Defect placement ───────────────────────────────────────────────
        mask = np.zeros((H, W), dtype=np.uint8)
        defect_bbox = None

        if with_defect:
            candidates = self._find_defect_candidates(gates, bands)
            if candidates:
                dx, y_lo, y_hi = candidates[int(rng.integers(0, len(candidates)))]
                dy = int(rng.integers(y_lo, y_hi + 1))
                canvas[dy:dy + ds, dx:dx + ds] = gate_val   # same brightness as gate
                mask[dy:dy + ds, dx:dx + ds]   = 1
                defect_bbox = dict(x=int(dx), y=int(dy), w=int(ds), h=int(ds))
            else:
                # Fallback: no valid NEPI slot found (very rare for default params)
                with_defect = False

        # ── 4. SEM noise model ────────────────────────────────────────────────
        # Step A: Gaussian blur — simulates electron beam point-spread function.
        #         Uses a small kernel; sigma=0.7 keeps structural edges sharp.
        blurred = cv2.GaussianBlur(
            canvas,
            ksize=(0, 0),
            sigmaX=self.blur_sigma,
            sigmaY=self.blur_sigma,
        )

        # Step B: Additive Gaussian noise — simulates shot noise + readout noise.
        noise   = rng.normal(0.0, self.noise_std, blurred.shape).astype(np.float32)
        noisy   = np.clip(blurred + noise, 0, 255).astype(np.uint8)

        # ── 5. Convert to 3-channel (ViT expects 3 channels) ─────────────────
        image_rgb = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)

        meta = dict(
            gate_phase_offset=phase_offset,
            gate_brightness=gate_val,
            pepi_brightness=pepi_val,
            nepi_brightness=nepi_val,
            num_gates=len(gates),
            has_defect=with_defect,
            defect_bbox=defect_bbox,
        )

        return image_rgb, mask, meta


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    output_dir: str = 'data/sem_defect',
    n_train: int = 800,
    n_val: int = 100,
    n_test: int = 100,
    defect_ratio: float = 0.7,    # fraction of images that contain a defect
    seed: int = 42,
    generator_kwargs: dict = None,
):
    """
    Generate a full train/val/test dataset and save to disk.

    Parameters
    ----------
    output_dir      : root output directory.
    n_train/val/test: number of images per split.
    defect_ratio    : fraction of images with a defect (rest are defect-free negatives).
    seed            : global random seed for reproducibility.
    generator_kwargs: extra kwargs forwarded to SEMImageGenerator.__init__.
    """
    output_dir = Path(output_dir)
    gen = SEMImageGenerator(**(generator_kwargs or {}))

    rng = np.random.default_rng(seed)
    random.seed(seed)

    splits = {
        'train': n_train,
        'val':   n_val,
        'test':  n_test,
    }

    all_metadata = {}

    for split, n in splits.items():
        img_dir  = output_dir / 'images' / split
        mask_dir = output_dir / 'masks'  / split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {n} images for split='{split}' ...")
        split_meta = []

        for i in range(n):
            with_defect = (rng.random() < defect_ratio)
            img, mask, meta = gen.generate(with_defect=with_defect, rng=rng)

            stem = f"{split}_{i:05d}"
            cv2.imwrite(str(img_dir  / f"{stem}.png"), img)
            cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)

            meta['filename'] = stem
            split_meta.append(meta)

        all_metadata[split] = split_meta
        n_defect = sum(1 for m in split_meta if m['has_defect'])
        print(f"  {n_defect}/{n} images have defects ({100*n_defect/n:.1f}%)")

    # Save metadata for auditing / visualisation
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nDataset saved to: {output_dir.resolve()}")
    return all_metadata


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation utility
# ─────────────────────────────────────────────────────────────────────────────

def visualise_sample(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    meta: dict,
    zoom: int = 3,
    save_path: str = None,
):
    """
    Display (or save) an image+mask side-by-side with defect bbox overlay.

    zoom: integer upscale factor for visibility of tiny structures.
    """
    H, W = image_rgb.shape[:2]
    zH, zW = H * zoom, W * zoom

    img_show  = cv2.resize(image_rgb, (zW, zH), interpolation=cv2.INTER_NEAREST)
    mask_show = cv2.resize(mask * 255, (zW, zH), interpolation=cv2.INTER_NEAREST)
    mask_bgr  = cv2.cvtColor(mask_show, cv2.COLOR_GRAY2BGR)

    # Draw defect bbox on the image panel
    if meta.get('defect_bbox'):
        b = meta['defect_bbox']
        x1 = b['x'] * zoom
        y1 = b['y'] * zoom
        x2 = (b['x'] + b['w']) * zoom
        y2 = (b['y'] + b['h']) * zoom
        cv2.rectangle(img_show, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 1)
        cv2.putText(
            img_show, 'DEFECT',
            (max(0, x1 - 5), max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
        )

    combined = np.hstack([img_show, mask_bgr])

    if save_path:
        cv2.imwrite(save_path, combined)
        print(f"Saved visualisation → {save_path}")
    else:
        cv2.imshow('SEM sample  |  left: image  right: mask', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Generate synthetic SEM defect dataset')
    p.add_argument('--output_dir',    default='data/sem_defect')
    p.add_argument('--n_train',       type=int,   default=800)
    p.add_argument('--n_val',         type=int,   default=100)
    p.add_argument('--n_test',        type=int,   default=100)
    p.add_argument('--defect_ratio',  type=float, default=0.7)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--gate_period',   type=int,   default=22,
                   help='X-axis pitch of Metal Gates in pixels')
    p.add_argument('--gate_width',    type=int,   default=7)
    p.add_argument('--band_height',   type=int,   default=64,
                   help='Height of each PEPI or NEPI band')
    p.add_argument('--noise_std',     type=float, default=12.0)
    p.add_argument('--blur_sigma',    type=float, default=0.7)
    p.add_argument('--preview',       action='store_true',
                   help='Generate and show one sample before writing dataset')
    p.add_argument('--preview_save',  default=None,
                   help='If set, save the preview image to this path instead of displaying')
    return p.parse_args()


def main():
    args = parse_args()

    generator_kwargs = dict(
        gate_period=args.gate_period,
        gate_width=args.gate_width,
        band_height=args.band_height,
        noise_std=args.noise_std,
        blur_sigma=args.blur_sigma,
    )
    gen = SEMImageGenerator(**generator_kwargs)

    if args.preview or args.preview_save:
        rng = np.random.default_rng(args.seed)
        img, mask, meta = gen.generate(with_defect=True, rng=rng)
        print("Preview metadata:", json.dumps(meta, indent=2))
        visualise_sample(img, mask, meta,
                         zoom=3,
                         save_path=args.preview_save)
        if not args.preview_save:
            return  # don't write dataset if interactive preview requested

    generate_dataset(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        defect_ratio=args.defect_ratio,
        seed=args.seed,
        generator_kwargs=generator_kwargs,
    )


if __name__ == '__main__':
    main()

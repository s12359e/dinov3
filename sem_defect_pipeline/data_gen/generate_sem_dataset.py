"""
Synthetic SEM (Scanning Electron Microscope) image generator.

Supports two region types:
  - SRAM  : Continuous vertical metal gate stripes (no cuts).
  - Logic : Metal gates with CPODE cuts — random horizontal gaps where
            the gate is interrupted, exposing the underlying PEPI/NEPI band.

Topology rules
--------------
Metal Gate  : Vertical stripes, X-axis period = 22 px.
              Brightest component  → pixel value ~ 230–250.

PEPI band   : Horizontal regions alternating with NEPI.
              Second brightest     → pixel value ~  160–190.

NEPI band   : Between PEPI bands.
              Third brightest (darkest) → pixel value ~  60–90.

Cut (Logic) : A horizontal gap in a gate stripe where the gate metal is
              removed (CPODE cut).  The gap exposes the underlying band
              brightness.  Cut height is 3–6 px, placed at PEPI/NEPI
              band boundaries (typical for CPODE placement).

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
labels/  — YOLO segmentation .txt (normalized polygon coordinates)

Usage
-----
    python generate_sem_dataset.py                   # default: 200 SRAM + 200 Logic per split
    python generate_sem_dataset.py --n_train 400 --region sram --seed 0
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
    region_type       : 'sram' (continuous gates) or 'logic' (gates with cuts).
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
    cut_height_range  : (min, max) height of gate cuts in pixels (default (3, 6)).
    cuts_per_gate     : (min, max) number of cuts per gate stripe for Logic (default (1, 3)).
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        region_type: str = 'sram',
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
        cut_height_range: tuple = (3, 6),
        cuts_per_gate: tuple = (1, 3),
    ):
        assert gate_width < gate_period, "gate_width must be less than gate_period"
        assert defect_size >= 1
        assert region_type in ('sram', 'logic'), f"region_type must be 'sram' or 'logic', got '{region_type}'"

        self.width = width
        self.height = height
        self.region_type = region_type
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
        self.cut_height_range = cut_height_range
        self.cuts_per_gate = cuts_per_gate

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

    def _generate_gate_cuts(self, gates, bands, rng):
        """
        Generate random cut positions for Logic region gates.

        Cuts are placed near PEPI/NEPI band boundaries (typical CPODE placement).
        Returns a list of (x_start, x_end, y_start, y_end) rectangles where
        gate material is removed.
        """
        cuts = []
        # Collect band boundary y-positions (where PEPI meets NEPI)
        band_boundaries = []
        for i in range(len(bands) - 1):
            _, ye, _ = bands[i]
            band_boundaries.append(ye)

        for (xs, xe) in gates:
            n_cuts = int(rng.integers(self.cuts_per_gate[0], self.cuts_per_gate[1] + 1))
            # Randomly pick band boundaries for this gate's cuts
            if not band_boundaries:
                continue
            chosen = rng.choice(
                len(band_boundaries),
                size=min(n_cuts, len(band_boundaries)),
                replace=False,
            )
            for idx in chosen:
                by = band_boundaries[idx]
                cut_h = int(rng.integers(self.cut_height_range[0], self.cut_height_range[1] + 1))
                # Center the cut on the band boundary
                cut_y_start = max(0, by - cut_h // 2)
                cut_y_end = min(self.height, cut_y_start + cut_h)
                cuts.append((xs, xe, cut_y_start, cut_y_end))

        return cuts

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

        # ── 2b. Gate cuts for Logic region ────────────────────────────────────
        gate_cuts = []
        if self.region_type == 'logic':
            gate_cuts = self._generate_gate_cuts(gates, bands, rng)
            # Build a lookup: for each row, what is the underlying band brightness
            band_brightness_map = np.zeros(H, dtype=np.float32)
            for (ys, ye, kind) in bands:
                val = pepi_val if kind == 'PEPI' else nepi_val
                band_brightness_map[ys:ye] = val
            # Remove gate material at cut positions → expose underlying band
            for (xs, xe, cy_start, cy_end) in gate_cuts:
                for row in range(cy_start, cy_end):
                    canvas[row, xs:xe] = band_brightness_map[row]

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
            region_type=self.region_type,
            gate_phase_offset=phase_offset,
            gate_brightness=gate_val,
            pepi_brightness=pepi_val,
            nepi_brightness=nepi_val,
            num_gates=len(gates),
            num_gate_cuts=len(gate_cuts),
            has_defect=with_defect,
            defect_bbox=defect_bbox,
        )

        return image_rgb, mask, meta


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def _mask_to_yolo_label(mask: np.ndarray, class_id: int = 0) -> str:
    """
    Convert a binary mask to YOLO segmentation label format.

    Returns a string where each line is:
        class_id x1 y1 x2 y2 ... xn yn   (normalized to [0, 1])

    If mask has no foreground pixels, returns an empty string.
    """
    H, W = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # contour shape: (N, 1, 2) — squeeze and normalize
        pts = contour.squeeze(1)  # (N, 2)
        coords = []
        for (x, y) in pts:
            coords.append(f"{x / W:.6f}")
            coords.append(f"{y / H:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    return "\n".join(lines)


def generate_dataset(
    output_dir: str = 'data/sem_defect',
    n_per_region: int = 200,
    regions: list = None,
    n_train_ratio: float = 0.7,
    n_val_ratio: float = 0.15,
    n_test_ratio: float = 0.15,
    defect_ratio: float = 0.7,
    seed: int = 42,
    generator_kwargs: dict = None,
):
    """
    Generate a full train/val/test dataset for one or more region types.

    Parameters
    ----------
    output_dir      : root output directory.
    n_per_region    : total images per region type (split into train/val/test).
    regions         : list of region types to generate, e.g. ['sram', 'logic'].
    n_train_ratio   : fraction for train split (default 0.7).
    n_val_ratio     : fraction for val split (default 0.15).
    n_test_ratio    : fraction for test split (default 0.15).
    defect_ratio    : fraction of images with a defect.
    seed            : global random seed.
    generator_kwargs: extra kwargs forwarded to SEMImageGenerator.__init__.
    """
    if regions is None:
        regions = ['sram', 'logic']

    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    all_metadata = {}

    for region in regions:
        kwargs = dict(generator_kwargs or {})
        kwargs['region_type'] = region
        gen = SEMImageGenerator(**kwargs)

        # Compute split counts
        n_train = int(n_per_region * n_train_ratio)
        n_val = int(n_per_region * n_val_ratio)
        n_test = n_per_region - n_train - n_val

        splits = {'train': n_train, 'val': n_val, 'test': n_test}

        for split, n in splits.items():
            img_dir   = output_dir / 'images' / split
            mask_dir  = output_dir / 'masks'  / split
            label_dir = output_dir / 'labels' / split
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            print(f"Generating {n} {region.upper()} images for split='{split}' ...")
            split_meta = []

            for i in range(n):
                with_defect = (rng.random() < defect_ratio)
                img, mask, meta = gen.generate(with_defect=with_defect, rng=rng)

                stem = f"{region}_{split}_{i:05d}"
                cv2.imwrite(str(img_dir   / f"{stem}.png"), img)
                cv2.imwrite(str(mask_dir  / f"{stem}.png"), mask)

                # Write YOLO segmentation label
                yolo_txt = _mask_to_yolo_label(mask, class_id=0)
                with open(label_dir / f"{stem}.txt", 'w') as f:
                    f.write(yolo_txt)

                meta['filename'] = stem
                split_meta.append(meta)

            key = f"{region}_{split}"
            all_metadata[key] = split_meta
            n_defect = sum(1 for m in split_meta if m['has_defect'])
            print(f"  {n_defect}/{n} images have defects ({100*n_defect/n:.1f}%)")

    # Save metadata
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
    p.add_argument('--n_per_region',  type=int,   default=200,
                   help='Total images per region type (split into train/val/test)')
    p.add_argument('--region',        nargs='+',  default=['sram', 'logic'],
                   choices=['sram', 'logic'],
                   help='Region types to generate (default: both)')
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
    p.add_argument('--preview_region', default='logic', choices=['sram', 'logic'],
                   help='Region type for preview (default: logic)')
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

    if args.preview or args.preview_save:
        gen = SEMImageGenerator(region_type=args.preview_region, **generator_kwargs)
        rng = np.random.default_rng(args.seed)
        img, mask, meta = gen.generate(with_defect=True, rng=rng)
        print("Preview metadata:", json.dumps(meta, indent=2))
        visualise_sample(img, mask, meta,
                         zoom=3,
                         save_path=args.preview_save)
        if not args.preview_save:
            return

    generate_dataset(
        output_dir=args.output_dir,
        n_per_region=args.n_per_region,
        regions=args.region,
        defect_ratio=args.defect_ratio,
        seed=args.seed,
        generator_kwargs=generator_kwargs,
    )


if __name__ == '__main__':
    main()

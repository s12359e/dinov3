"""Phase-3 synthetic-defect injection (behind a config flag).

Applies a CutPaste-style transplant / thin bright-or-dark line / small spot to the
*target* view of a fraction of NORMAL triplets. Returns the modified image and a pixel
mask of the pasted region (used by the repulsion term; the mask is a training signal
here, unlike the GT masks which are eval-only).

Also provides `augmentation_survival_check` to verify synthetic defects survive the view
augmentation pipeline (not cropped out / blurred away).
"""

import numpy as np


class SyntheticDefect:
    def __init__(self, prob=0.5, types=("psf",),
                 spot_size=(3, 7), line_len=(10, 30),
                 psf_sigma=(1.1, 1.7), psf_amplitude=(15, 80), seed=0):
        self.prob = prob
        self.types = types
        self.spot_size = spot_size
        self.line_len = line_len
        self.psf_sigma = psf_sigma          # footprint (>20% peak) ~ 3.6*sigma px
        self.psf_amplitude = psf_amplitude  # gray levels; sign randomized
        self.rng = np.random.default_rng(seed)

    def _psf_blob(self, out, mask):
        """Low-contrast Gaussian PSF blob — the realistic optical point-defect
        morphology (~4-6 px footprint at sigma 1.0-1.7). Sub-pixel center; bright
        or dark; amplitude sampled over a nuisance-to-obvious range so the model
        learns a margin, not a single contrast. Mask = region > 20% of peak."""
        h, w = out.shape[:2]
        sigma = float(self.rng.uniform(*self.psf_sigma))
        amp = float(self.rng.uniform(*self.psf_amplitude))
        if self.rng.random() < 0.5:
            amp = -amp
        r = int(np.ceil(4 * sigma))
        cy = float(self.rng.uniform(r, h - 1 - r))
        cx = float(self.rng.uniform(r, w - 1 - r))
        y0, y1 = int(cy) - r, int(cy) + r + 1
        x0, x1 = int(cx) - r, int(cx) + r + 1
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                   / (2 * sigma * sigma)).astype(np.float32)
        region = out[y0:y1, x0:x1].astype(np.float32) + amp * g[..., None]
        out[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
        mask[y0:y1, x0:x1] |= (g > 0.2).astype(np.uint8)
        return out, mask

    def __call__(self, img, is_defect=False):
        # Only inject into normal triplets; never touch already-defective ones.
        if is_defect or self.rng.random() > self.prob:
            return img, None
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        out = img.copy()
        kind = self.types[self.rng.integers(0, len(self.types))]

        if kind == "psf":
            out, mask = self._psf_blob(out, mask)
        elif kind == "spot":
            s = int(self.rng.integers(*self.spot_size))
            y, x = int(self.rng.integers(0, h - s)), int(self.rng.integers(0, w - s))
            val = 255 if self.rng.random() < 0.5 else 0
            out[y:y + s, x:x + s] = val
            mask[y:y + s, x:x + s] = 1
        elif kind == "line":
            length = int(self.rng.integers(*self.line_len))
            thick = int(self.rng.integers(1, 3))
            y, x = int(self.rng.integers(0, h)), int(self.rng.integers(0, max(1, w - length)))
            val = 255 if self.rng.random() < 0.5 else 0
            out[y:y + thick, x:x + length] = val
            mask[y:y + thick, x:x + length] = 1
        else:  # cutpaste: transplant a random patch elsewhere
            s = int(self.rng.integers(8, 20))
            sy, sx = int(self.rng.integers(0, h - s)), int(self.rng.integers(0, w - s))
            dy, dx = int(self.rng.integers(0, h - s)), int(self.rng.integers(0, w - s))
            out[dy:dy + s, dx:dx + s] = img[sy:sy + s, sx:sx + s]
            mask[dy:dy + s, dx:dx + s] = 1

        return out, mask


def augmentation_survival_check(dataset, n=100):
    """Fraction of injected synthetic defects that survive augmentation (mask non-empty
    after the shared crop). Reports survival rate. Requires dataset.synth_defect set."""
    survived = total = 0
    for i in range(min(n, len(dataset))):
        s = dataset[i]
        if s.get("synth_injected"):
            total += 1
            if s["synth_mask"].sum() > 0:
                survived += 1
    rate = survived / total if total else 0.0
    print(f"[synth-survival] {survived}/{total} synthetic defects survived augmentation "
          f"(rate={rate:.2f})")
    return rate

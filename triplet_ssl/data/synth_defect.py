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
    def __init__(self, prob=0.5, types=("cutpaste", "line", "spot"),
                 spot_size=(3, 7), line_len=(10, 30), seed=0):
        self.prob = prob
        self.types = types
        self.spot_size = spot_size
        self.line_len = line_len
        self.rng = np.random.default_rng(seed)

    def __call__(self, img, is_defect=False):
        # Only inject into normal triplets; never touch already-defective ones.
        if is_defect or self.rng.random() > self.prob:
            return img, None
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        out = img.copy()
        kind = self.types[self.rng.integers(0, len(self.types))]

        if kind == "spot":
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

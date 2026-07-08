"""Triplet dataset + shared-geometry / independent-photometric augmentation.

Design (respects the spec):

* Geometric augmentation (crop box + scale, horizontal flip) is sampled ONCE per
  triplet and applied identically to ref1/ref2/target, so patch k corresponds to the
  same spatial location in every view -> patch-level pairing is valid.
* Photometric augmentation (brightness / contrast / gamma) is sampled INDEPENDENTLY per
  image. Two photometric variants per image ("student" / "teacher" views) enable the
  traditional same-image DINO pair while staying geometrically aligned.
* Vertical flip is FORBIDDEN (directional pattern). Blur / colour jitter are kept mild so
  small low-contrast defects survive.
* Optional phase-correlation registration (config flag) corrects die-to-die shift at
  SUB-PIXEL precision before cropping (warpAffine, replicated border); warns if the
  measured shift exceeds half a patch (patch pairing validity).

TRAINING NEVER READS GT MASKS. `TripletDataset(load_masks=False)` (the default for
training) does not open `mask.png`; `assert_no_masks()` enforces it.
"""

import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from triplet_ssl import IMG_MEAN, IMG_STD


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
class SharedGeomTripletAug:
    """Produce 2 photometric variants of each of ref1/ref2/target on a shared crop."""

    def __init__(self, img_size=224, crop_scale=(0.4, 1.0), flip_prob=0.5,
                 brightness_delta=15, contrast_range=(0.9, 1.1), gamma_range=(0.9, 1.1),
                 blur_prob=0.0, blur_sigma=(0.1, 0.6)):
        self.img_size = img_size
        self.crop_scale = crop_scale
        self.flip_prob = flip_prob
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.mean = np.array(IMG_MEAN, np.float32).reshape(1, 1, 3)
        self.std = np.array(IMG_STD, np.float32).reshape(1, 1, 3)

    # -- geometric (shared) -------------------------------------------------- #
    def sample_geom(self, h, w):
        scale = np.random.uniform(*self.crop_scale)
        ch, cw = int(h * scale), int(w * scale)
        top = np.random.randint(0, max(h - ch, 1) + 1)
        left = np.random.randint(0, max(w - cw, 1) + 1)
        flip = np.random.random() < self.flip_prob   # horizontal only
        return dict(top=top, left=left, ch=ch, cw=cw, flip=flip)

    def apply_geom(self, img, g, interp=cv2.INTER_LINEAR):
        crop = img[g["top"]:g["top"] + g["ch"], g["left"]:g["left"] + g["cw"]]
        crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=interp)
        if g["flip"]:
            crop = np.ascontiguousarray(np.fliplr(crop))
        return crop

    # -- photometric (independent) ------------------------------------------ #
    def photometric(self, img_u8):
        x = img_u8.astype(np.float32)
        if np.random.random() < 0.5:
            x += np.random.uniform(-self.brightness_delta, self.brightness_delta)
        if np.random.random() < 0.5:
            x *= np.random.uniform(*self.contrast_range)
        if np.random.random() < 0.5:
            g = np.random.uniform(*self.gamma_range)
            x = 255.0 * np.clip(x / 255.0, 0, 1) ** g
        x = np.clip(x, 0, 255)
        if self.blur_prob > 0 and np.random.random() < self.blur_prob:
            x = cv2.GaussianBlur(x, (0, 0), np.random.uniform(*self.blur_sigma))
        x = (x - self.mean) / self.std
        return torch.from_numpy(x).permute(2, 0, 1).float()

    def __call__(self, imgs):
        """imgs: dict name->HxWx3 uint8 (already registered). Returns name->(2,3,H,W)."""
        h, w = next(iter(imgs.values())).shape[:2]
        g = self.sample_geom(h, w)
        out = {}
        for name, im in imgs.items():
            cropped = self.apply_geom(im, g)
            out[name] = torch.stack([self.photometric(cropped),      # student view
                                     self.photometric(cropped)])     # teacher view
        return out, g


# --------------------------------------------------------------------------- #
# Registration (optional)
# --------------------------------------------------------------------------- #
def coarse_register(ref, mov, max_shift_assert):
    """Register `mov` onto `ref` at sub-pixel precision. Returns (aligned, (dx,dy)).

    Sub-pixel matters for optical die-to-die: a 0.3-0.7 px residual misalignment
    produces edge residuals comparable to a low-contrast 4-6 px PSF defect.

    Two stages: Hanning-windowed phase correlation for the coarse shift (robust to
    large offsets, but its sub-pixel peak is biased ~0.5 px on periodic patterns),
    then ECC gradient refinement (~0.03 px measured; falls back to the coarse shift
    if it fails to converge). The float shift is applied with cv2.warpAffine and a
    replicated border -- no wraparound of opposite-edge content, unlike np.roll.
    """
    r = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.cvtColor(mov, cv2.COLOR_RGB2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(r.shape[::-1], cv2.CV_32F)
    (sx, sy), _ = cv2.phaseCorrelate(r, m, win)

    warp_mat = np.float32([[1, 0, sx], [0, 1, sy]])  # seed ECC with the coarse shift
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        cv2.findTransformECC(r, m, warp_mat, cv2.MOTION_TRANSLATION, crit)
        sx, sy = float(warp_mat[0, 2]), float(warp_mat[1, 2])
    except cv2.error:
        warnings.warn("ECC refinement failed to converge; using phase-correlation shift")

    if max(abs(sx), abs(sy)) > max_shift_assert:
        warnings.warn(f"registration residual {max(abs(sx), abs(sy)):.1f}px "
                      f"> {max_shift_assert}px (half patch): patch pairing may be invalid")
    h, w = mov.shape[:2]
    M = np.float32([[1, 0, -sx], [0, 1, -sy]])
    aligned = cv2.warpAffine(mov, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned, (sx, sy)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TripletDataset(Dataset):
    def __init__(self, root, split="train", transform=None, register=False,
                 patch_size=16, load_masks=False, synth_defect=None):
        self.dir = Path(root) / split
        self.transform = transform
        self.register = register
        self.patch_size = patch_size
        self.max_shift_assert = patch_size / 2.0   # half a patch
        self.load_masks = load_masks
        self.synth_defect = synth_defect            # phase-3 injector or None

        with open(self.dir / "manifest.json") as f:
            self.items = json.load(f)
        self.defect_idx = [i for i, m in enumerate(self.items) if m["has_defect"]]
        self.normal_idx = [i for i, m in enumerate(self.items) if not m["has_defect"]]
        self.offset_log = []

    def __len__(self):
        return len(self.items)

    def assert_no_masks(self):
        assert not self.load_masks, "GT masks must not be loaded in the training path"

    def _read(self, tid, name):
        img = cv2.imread(str(self.dir / tid / f"{name}.png"), cv2.IMREAD_COLOR)
        # RGB to match the eval loader's channel order (no-op for grayscale-replicated).
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _pixel_to_patch_mask(self, pix_mask):
        """Downsample a pixel mask to a flat patch-grid mask (1 if any overlap)."""
        ps = self.patch_size
        g = pix_mask.shape[0] // ps
        m = pix_mask[: g * ps, : g * ps].reshape(g, ps, g, ps).max(axis=(1, 3))
        return (m > 0).astype(np.float32).reshape(-1)

    def __getitem__(self, idx):
        m = self.items[idx]
        tid = m["id"]
        ref1, ref2, target = (self._read(tid, n) for n in ("ref1", "ref2", "target"))

        if self.register:
            ref2, o2 = coarse_register(ref1, ref2, self.max_shift_assert)
            target, ot = coarse_register(ref1, target, self.max_shift_assert)
            self.offset_log.append((o2, ot))

        # Phase-3 synthetic defect injected BEFORE the crop; its pixel mask rides the
        # SAME geometric transform, so patch alignment holds AND the survival check is
        # meaningful (the defect can genuinely be cropped out).
        pix_mask = None
        if self.synth_defect is not None:
            target, pix_mask = self.synth_defect(target, is_defect=m["has_defect"])

        # Shared geometric crop (sampled once) applied to all three.
        h, w = ref1.shape[:2]
        g = self.transform.sample_geom(h, w)
        c1 = self.transform.apply_geom(ref1, g)
        c2 = self.transform.apply_geom(ref2, g)
        ct = self.transform.apply_geom(target, g)

        # Always emit a synth_mask when synth is enabled (zeros if not injected) so
        # every sample in a batch has the same keys for collation.
        synth_patch_mask = None
        if self.synth_defect is not None:
            if pix_mask is not None:
                mask_c = self.transform.apply_geom(pix_mask, g, interp=cv2.INTER_NEAREST)
                synth_patch_mask = self._pixel_to_patch_mask(mask_c)
            else:
                n = (self.transform.img_size // self.patch_size) ** 2
                synth_patch_mask = np.zeros(n, np.float32)

        def views(img):  # two independent photometric variants (student, teacher)
            return torch.stack([self.transform.photometric(img),
                                self.transform.photometric(img)])

        sample = dict(ref1=views(c1), ref2=views(c2), target=views(ct),
                      is_defect=bool(m["has_defect"]), id=tid)
        if synth_patch_mask is not None:
            sample["synth_mask"] = torch.from_numpy(synth_patch_mask).float()
            sample["synth_injected"] = pix_mask is not None
        return sample


def triplet_collate(batch):
    out = dict(
        ref1=torch.stack([b["ref1"] for b in batch]),
        ref2=torch.stack([b["ref2"] for b in batch]),
        target=torch.stack([b["target"] for b in batch]),
        is_defect=torch.tensor([b["is_defect"] for b in batch]),
        id=[b["id"] for b in batch],
    )
    if "synth_mask" in batch[0]:
        out["synth_mask"] = torch.stack([b["synth_mask"] for b in batch])
    return out


class DefectOversampleBatchSampler:
    """Yield index batches where a target fraction are defect triplets (5-20%).

    Phase 1 disables this (defect_frac=None) -> plain shuffled batches.
    """

    def __init__(self, dataset, batch_size, defect_frac=None, num_batches=None,
                 seed=0):
        self.ds = dataset
        self.bs = batch_size
        self.defect_frac = defect_frac
        self.rng = np.random.default_rng(seed)
        self.num_batches = num_batches or max(1, len(dataset) // batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        n_all = len(self.ds)
        for _ in range(self.num_batches):
            if self.defect_frac is None or not self.ds.defect_idx:
                yield list(self.rng.integers(0, n_all, size=self.bs))
                continue
            n_def = int(round(self.defect_frac * self.bs))
            n_def = min(max(n_def, 1), self.bs)
            defs = self.rng.choice(self.ds.defect_idx, size=n_def, replace=True)
            norms = self.rng.choice(self.ds.normal_idx or self.ds.defect_idx,
                                    size=self.bs - n_def, replace=True)
            batch = np.concatenate([defs, norms])
            self.rng.shuffle(batch)
            yield list(batch)

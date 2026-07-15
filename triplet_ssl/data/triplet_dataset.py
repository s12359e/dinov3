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

        # Phase-3 synthetic PSF events injected AFTER registration, BEFORE the crop:
        # nuisance combos scattered across all three dies + optional true (1,0,0)
        # defect on target. Only the defect's pixel mask comes back; it rides the
        # SAME geometric transform, so patch alignment holds AND the survival check
        # is meaningful (the defect can genuinely be cropped out).
        pix_mask = None
        if self.synth_defect is not None:
            ref1, ref2, target, pix_mask = self.synth_defect(
                ref1, ref2, target, is_defect=m["has_defect"])

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


def read_tiff3(path, channel_order=(0, 1, 2)):
    """3-channel TIFF -> (target, ref1, ref2), each grayscale replicated to 3ch
    uint8. Supports uint8/uint16 and channels-last or 3-page layouts. Shared by
    the training dataset and the inference script."""
    from PIL import Image
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 2 and getattr(img, "n_frames", 1) >= 3:   # 3-page layout
        chans = []
        for i in range(3):
            img.seek(i)
            chans.append(np.array(img))
        arr = np.stack(chans, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"{path}: expected 3-channel TIFF, got shape {arr.shape}")
    if arr.dtype == np.uint16:
        arr = (arr / 256).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    co = channel_order
    to3 = lambda c: np.repeat(arr[:, :, c:c + 1], 3, axis=2)
    return to3(co[0]), to3(co[1]), to3(co[2])   # target, ref1, ref2


class TiffTripletDataset(Dataset):
    """One 3-channel TIFF per pattern location: ch0=target, ch1=ref1, ch2=ref2.

    Flow (per the user's training recipe):
      TIFF -> split channels -> [optional target-anchored sub-pixel registration]
           -> shared 128x128 window crop at NATIVE resolution (no resize -- a 4-6px
              PSF defect survives) + shared h-flip
           -> on-the-fly TripletSyntheticPSF injection ON THE CROP
           -> independent photometric x2 per image.

    No manifest / GT needed: all triplets are treated as normal; the defect signal
    comes entirely from synthetic (1,0,0) events. Supports uint8 / uint16 TIFFs and
    both channels-last and 3-page layouts.
    """

    def __init__(self, root, crop_size=128, transform=None, register=False,
                 patch_size=16, synth_defect=None, channel_order=(0, 1, 2),
                 cls_local_size=64):
        self.files = sorted(list(Path(root).glob("*.tif")) + list(Path(root).glob("*.tiff")))
        if not self.files:
            raise FileNotFoundError(f"no .tif/.tiff in {root}")
        self.crop = crop_size
        self.transform = transform
        self.register = register
        self.patch_size = patch_size
        self.max_shift_assert = patch_size / 2.0
        self.synth_defect = synth_defect
        self.channel_order = channel_order   # (target, ref1, ref2) channel indices
        if cls_local_size is not None:
            assert cls_local_size % patch_size == 0 and cls_local_size < crop_size
        self.cls_local_size = cls_local_size  # student-side local CLS view (no resize)
        self.load_masks = False              # no GT anywhere in this dataset
        self.defect_idx = []                 # real-defect flags unknown -> all "normal"
        self.normal_idx = list(range(len(self.files)))
        self.offset_log = []

    def __len__(self):
        return len(self.files)

    def assert_no_masks(self):
        assert not self.load_masks, "GT masks must not be loaded in the training path"

    def _read_tiff(self, path):
        return read_tiff3(path, self.channel_order)

    def __getitem__(self, idx):
        path = self.files[idx]
        target, ref1, ref2 = self._read_tiff(path)

        # Registration BEFORE crop, anchored on TARGET (target never warped, so any
        # defect coordinates stay put; refs are warped onto it).
        if self.register:
            ref1, o1 = coarse_register(target, ref1, self.max_shift_assert)
            ref2, o2 = coarse_register(target, ref2, self.max_shift_assert)
            self.offset_log.append((o1, o2))

        # Shared native-resolution window crop + shared h-flip (no resize).
        h, w = target.shape[:2]
        c = self.crop
        top = np.random.randint(0, max(h - c, 0) + 1)
        left = np.random.randint(0, max(w - c, 0) + 1)
        flip = np.random.random() < 0.5
        def cut(img):
            win = img[top:top + c, left:left + c]
            return np.ascontiguousarray(np.fliplr(win)) if flip else win.copy()
        tgt_c, r1_c, r2_c = cut(target), cut(ref1), cut(ref2)

        # On-the-fly synthetic PSF events ON THE CROP (user's recipe: crop then paste).
        pix_mask = None
        if self.synth_defect is not None:
            r1_c, r2_c, tgt_c, pix_mask = self.synth_defect(r1_c, r2_c, tgt_c,
                                                            is_defect=False)
        synth_patch_mask = None
        if self.synth_defect is not None:
            if pix_mask is not None:
                g = c // self.patch_size
                m = pix_mask[: g * self.patch_size, : g * self.patch_size]
                m = m.reshape(g, self.patch_size, g, self.patch_size).max(axis=(1, 3))
                synth_patch_mask = (m > 0).astype(np.float32).reshape(-1)
            else:
                synth_patch_mask = np.zeros((c // self.patch_size) ** 2, np.float32)

        def views(img):   # two independent photometric variants (student, teacher)
            return torch.stack([self.transform.photometric(img),
                                self.transform.photometric(img)])

        sample = dict(ref1=views(r1_c), ref2=views(r2_c), target=views(tgt_c),
                      is_defect=False, id=path.stem)

        # CLS local views, DINO-style asymmetry in the CORRECT direction:
        # STUDENT sees a small local sub-window (varied / partial context),
        # TEACHER sees only the full native crop (stable, complete target)
        # -> local-to-global prediction, targets computed from full context.
        # No resize anywhere: the sub-window is fed at native scale (e.g. 64px
        # -> 4x4 tokens, cheap). Location shared across the triplet (same
        # physical site on every die); photometric independent per image.
        if self.cls_local_size is not None:
            # CENTER crop: together with the injector's center-jittered defect
            # placement this guarantees the local view always contains the pasted
            # defect (jitter + footprint < ls/2).
            ls = self.cls_local_size
            lt = ll = (c - ls) // 2
            for name, img in (("ref1", r1_c), ("ref2", r2_c), ("target", tgt_c)):
                win = np.ascontiguousarray(img[lt:lt + ls, ll:ll + ls])
                sample[name + "_cls"] = self.transform.photometric(win)

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
    for k in ("ref1_cls", "ref2_cls", "target_cls"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
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

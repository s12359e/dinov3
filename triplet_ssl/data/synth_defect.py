"""Phase-3 triplet-combinatoric synthetic PSF events.

Real optical short-wavelength targets carry MANY PSF-like points; morphology alone
cannot tell defect from nuisance -- only the CROSS-DIE presence pattern can. We
therefore inject PSF blobs into random subsets of (target, ref1, ref2):

    presence (t, r1, r2)        meaning       handled by
    (1, 0, 0)                   DEFECT        target-only positive
    every other combination     nuisance      hard negative / common mode

ONLY the (1,0,0) events enter the loss mask. In particular, (0,1,1) is not a
target-unique defect and remains a nuisance under the production truth table.

Injection happens AFTER registration (positions shared across aligned dies) and
BEFORE the shared crop (so events can genuinely be cropped out -> survival check).
"""

import copy

import numpy as np

# Presence order: (target, ref1, ref2). Only (1,0,0) is positive.
NUISANCE_COMBOS = ((0, 1, 0), (0, 0, 1), (0, 1, 1),
                   (1, 1, 0), (1, 0, 1), (1, 1, 1))
PULL_UNMATCHED_COMBOS = ((1, 0, 0), (0, 1, 1))


def presence_requires_pull_exemption(presence):
    """True when target has no same-presence reference to match at this event."""
    return tuple(int(bool(v)) for v in presence) in PULL_UNMATCHED_COMBOS


class TripletSyntheticPSF:
    def __init__(self, n_events=(3, 8), defect_prob=0.5, missing_frac=0.0,
                 psf_sigma=(1.1, 1.7), psf_amplitude=(15, 80),
                 amp_jitter=0.15, defect_center_jitter=None, seed=0):
        if missing_frac != 0:
            raise ValueError("missing_frac must be 0: only (1,0,0) is a defect")
        self.n_events = n_events            # nuisance PSF points per triplet
        self.defect_prob = defect_prob      # chance of ONE true (1,0,0) defect event
        self.psf_sigma = psf_sigma          # footprint (>20% peak) ~ 3.6*sigma px
        self.psf_amplitude = psf_amplitude  # gray levels (before sign)
        self.amp_jitter = amp_jitter        # per-die amplitude variation
        # If set (J px): the TRUE defect is placed at image center +- J instead of
        # uniformly -- guarantees a center crop of size >= 2*(J + footprint) always
        # contains it, while the jitter kills the "center patch = defect" positional
        # shortcut (RoPE gives the model position info). Nuisance stays uniform, so
        # position alone can never identify a defect.
        self.defect_center_jitter = defect_center_jitter
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._worker_id = None

    def _ensure_worker_rng(self):
        """Give each DataLoader worker an independent deterministic stream."""
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
        except ImportError:
            info = None
        worker_id = info.id if info is not None else None
        if worker_id is not None and worker_id != self._worker_id:
            self.rng = np.random.default_rng(self.seed + 1009 * (worker_id + 1))
            self._worker_id = worker_id

    # -- blob primitives ------------------------------------------------------ #
    def _sample_event(self, h, w):
        sigma = float(self.rng.uniform(*self.psf_sigma))
        amp = float(self.rng.uniform(*self.psf_amplitude))
        r = int(np.ceil(4 * sigma))
        cy = float(self.rng.uniform(r, h - 1 - r))
        cx = float(self.rng.uniform(r, w - 1 - r))
        return cy, cx, sigma, amp, r

    def _sign_for(self, img, cy, cx, r, amp):
        """Local-background-aware polarity: avoid clip saturation (a +amp blob on a
        ~240-gray gate would vanish after clipping -> noise supervision)."""
        win = img[int(cy) - r:int(cy) + r + 1, int(cx) - r:int(cx) + r + 1]
        bg = float(win.mean())
        if bg + amp > 250:
            return -1.0
        if bg - amp < 5:
            return 1.0
        return 1.0 if self.rng.random() < 0.5 else -1.0

    def _add_blob(self, img, cy, cx, sigma, amp, r):
        y0, y1 = int(cy) - r, int(cy) + r + 1
        x0, x1 = int(cx) - r, int(cx) + r + 1
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                   / (2 * sigma * sigma)).astype(np.float32)
        region = img[y0:y1, x0:x1].astype(np.float32) + amp * g[..., None]
        region = np.clip(region, 0, 255)
        if np.issubdtype(img.dtype, np.integer):
            region = np.rint(region).astype(img.dtype)
        else:
            region = region.astype(img.dtype, copy=False)
        img[y0:y1, x0:x1] = region
        return g > 0.2, (y0, y1, x0, x1)   # footprint mask + window

    # -- main ------------------------------------------------------------------ #
    def __call__(self, ref1, ref2, target, is_defect=False,
                 return_event_mask=False, return_supervision=False):
        """Returns (ref1, ref2, target, defect_pixel_mask | None).

        The mask covers ONLY true (1,0,0) defect events. Nuisance events get no
        positive mask by design.  With ``return_event_mask=True`` a fifth value
        is returned: the union of every injected event footprint.  It marks the
        only pixels where the complete synthetic truth table is known, allowing
        the fusion head to learn nuisance negatives without labelling unknown
        real-image background as normal.  ``return_supervision=True`` returns
        both that event mask and a sixth pull-exemption mask for directional
        ``100``/``011`` sites where target cannot legally match either ref."""
        self._ensure_worker_rng()
        h, w = target.shape[:2]
        imgs = {"t": target.copy(), "r1": ref1.copy(), "r2": ref2.copy()}
        event_mask = np.zeros((h, w), np.uint8)
        unmatched_mask = np.zeros((h, w), np.uint8)

        # 1) nuisance events: random presence combos, shared position across dies
        n = int(self.rng.integers(self.n_events[0], self.n_events[1] + 1))
        windows = []                      # (cy, cx, r) of every nuisance event
        for _ in range(n):
            cy, cx, sigma, amp, r = self._sample_event(h, w)
            combo = NUISANCE_COMBOS[int(self.rng.integers(len(NUISANCE_COMBOS)))]
            present = [k for k, p in zip(("t", "r1", "r2"), combo) if p]
            sign = self._sign_for(imgs[present[0]], cy, cx, r, amp)
            event_fp = event_win = None
            for k in present:
                a = amp * sign * (1 + self.rng.uniform(-self.amp_jitter, self.amp_jitter))
                fp, win = self._add_blob(imgs[k], cy, cx, sigma, a, r)
                if event_fp is None:
                    event_fp, event_win = fp, win
            y0, y1, x0, x1 = event_win
            event_mask[y0:y1, x0:x1] |= event_fp.astype(np.uint8)
            if presence_requires_pull_exemption(combo):
                unmatched_mask[y0:y1, x0:x1] |= event_fp.astype(np.uint8)
            windows.append((cy, cx, r))

        # 2) ONE true (1,0,0) defect event on normal triplets only, at a
        #    position clear of nuisance windows (keeps mask semantics clean).
        defect_mask = None
        if not is_defect and self.rng.random() < self.defect_prob:
            for _ in range(20):
                cy, cx, sigma, amp, r = self._sample_event(h, w)
                if self.defect_center_jitter is not None:
                    j = self.defect_center_jitter
                    cy = h / 2 + float(self.rng.uniform(-j, j))
                    cx = w / 2 + float(self.rng.uniform(-j, j))
                if all(abs(cy - wy) > r + wr or abs(cx - wx) > r + wr
                       for wy, wx, wr in windows):
                    break
            sign = self._sign_for(imgs["t"], cy, cx, r, amp)
            fp, (y0, y1, x0, x1) = self._add_blob(imgs["t"], cy, cx, sigma,
                                                  amp * sign, r)
            defect_mask = np.zeros((h, w), np.uint8)
            defect_mask[y0:y1, x0:x1] |= fp.astype(np.uint8)
            event_mask[y0:y1, x0:x1] |= fp.astype(np.uint8)
            unmatched_mask[y0:y1, x0:x1] |= fp.astype(np.uint8)

        result = (imgs["r1"], imgs["r2"], imgs["t"], defect_mask)
        if return_supervision:
            return result + (event_mask, unmatched_mask)
        return result + (event_mask,) if return_event_mask else result


def augmentation_survival_check(dataset, n=100):
    """Fraction of injected TRUE-defect events that survive augmentation (mask
    non-empty after the shared crop). Requires dataset.synth_defect set."""
    survived = total = 0
    synth = dataset.synth_defect
    rng_state = copy.deepcopy(synth.rng.bit_generator.state)
    worker_id = synth._worker_id
    try:
        for i in range(min(n, len(dataset))):
            s = dataset[i]
            if s.get("synth_injected"):
                total += 1
                if s["synth_mask"].sum() > 0:
                    survived += 1
    finally:
        synth.rng.bit_generator.state = rng_state
        synth._worker_id = worker_id
    rate = survived / total if total else 0.0
    print(f"[synth-survival] {survived}/{total} synthetic defects survived augmentation "
          f"(rate={rate:.2f})")
    return rate

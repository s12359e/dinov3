"""Synthetic *triplet* SEM generator for triplet-based SSL.

Reuses `sem_defect_pipeline.data_gen.generate_sem_dataset.SEMImageGenerator` so the
rendering (SRAM stripes / Logic gates+cuts, 4x4 metal-extrusion defect) is identical
to the existing single-image dataset. A triplet is three renders of the *same pattern
location* (same structure canvas):

  - ref1, ref2 : defect-free neighbouring "dies" (independent shot/readout noise).
  - target     : same location, with an OPTIONAL defect. Independent noise.

Die-to-die misalignment is simulated with an independent small integer pixel shift per
view (``--misalign_px``). The target's GT mask is shifted together with the target view
so it stays pixel-accurate. GT masks are for EVALUATION ONLY (see README); the SSL
training path never reads them.

Layout produced::

    <out>/<split>/<id>/ref1.png
                       /ref2.png
                       /target.png
                       /mask.png          # target GT (eval only)
    <out>/<split>/manifest.json           # per-triplet: id, has_defect, applied shifts

Usage::

    python triplet_ssl/data_gen/generate_triplets.py --out data/sem_triplet \
        --n_train 200 --n_val 60 --n_test 60 --defect_frac 0.5 --misalign_px 0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the fixed single-image generator's rendering primitives.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sem_defect_pipeline.data_gen.generate_sem_dataset import SEMImageGenerator


class TripletSEMGenerator(SEMImageGenerator):
    """Renders three noisy views of one shared clean structure canvas."""

    def _build_clean_canvas(self, rng):
        """Deterministic structure (bands + gates + logic cuts), no noise, no defect.

        Mirrors the structure-building half of ``SEMImageGenerator.generate`` so the
        three views share identical geometry.
        """
        W, H = self.width, self.height
        gate_val = self._jitter(self.gate_brightness, rng)
        pepi_val = self._jitter(self.pepi_brightness, rng)
        nepi_val = self._jitter(self.nepi_brightness, rng)

        canvas = np.zeros((H, W), dtype=np.float32)
        bands = self._band_regions()
        for (ys, ye, kind) in bands:
            canvas[ys:ye, :] = pepi_val if kind == "PEPI" else nepi_val

        phase_offset = int(rng.integers(0, self.gate_period))
        gates = self._gate_columns(phase_offset)
        for (xs, xe) in gates:
            canvas[:, xs:xe] = gate_val

        if self.region_type == "logic":
            gate_cuts = self._generate_gate_cuts(gates, bands, rng)
            band_map = np.zeros(H, dtype=np.float32)
            for (ys, ye, kind) in bands:
                band_map[ys:ye] = pepi_val if kind == "PEPI" else nepi_val
            for (xs, xe, cy_start, cy_end) in gate_cuts:
                for row in range(cy_start, cy_end):
                    canvas[row, xs:xe] = band_map[row]

        return canvas, gates, bands, gate_val

    def _render_view(self, canvas, rng, shift):
        """blur + independent noise + optional integer die-to-die shift -> uint8 3ch."""
        blurred = cv2.GaussianBlur(canvas, ksize=(0, 0),
                                   sigmaX=self.blur_sigma, sigmaY=self.blur_sigma)
        noise = rng.normal(0.0, self.noise_std, blurred.shape).astype(np.float32)
        noisy = np.clip(blurred + noise, 0, 255).astype(np.uint8)
        dx, dy = shift
        if dx or dy:
            noisy = np.roll(noisy, shift=(dy, dx), axis=(0, 1))
        return cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)

    def generate_triplet(self, with_defect, misalign_px, rng):
        clean, gates, bands, gate_val = self._build_clean_canvas(rng)
        ds = self.defect_size

        # Defect goes on a COPY used only for the target view.
        target_canvas = clean.copy()
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        defect_bbox = None
        if with_defect:
            candidates = self._find_defect_candidates(gates, bands)
            if candidates:
                dx, y_lo, y_hi = candidates[int(rng.integers(0, len(candidates)))]
                dy = int(rng.integers(y_lo, y_hi + 1))
                target_canvas[dy:dy + ds, dx:dx + ds] = gate_val
                mask[dy:dy + ds, dx:dx + ds] = 1
                defect_bbox = dict(x=int(dx), y=int(dy), w=int(ds), h=int(ds))
            else:
                with_defect = False

        def _shift():
            if misalign_px <= 0:
                return (0, 0)
            return (int(rng.integers(-misalign_px, misalign_px + 1)),
                    int(rng.integers(-misalign_px, misalign_px + 1)))

        s1, s2, st = _shift(), _shift(), _shift()
        ref1 = self._render_view(clean, rng, s1)
        ref2 = self._render_view(clean, rng, s2)
        target = self._render_view(target_canvas, rng, st)
        # Keep the mask registered to the (shifted) target view.
        if st != (0, 0):
            mask = np.roll(mask, shift=(st[1], st[0]), axis=(0, 1))

        meta = dict(region_type=self.region_type, has_defect=with_defect,
                    defect_bbox=defect_bbox,
                    shifts=dict(ref1=s1, ref2=s2, target=st))
        return dict(ref1=ref1, ref2=ref2, target=target, mask=mask, meta=meta)


def _build_split(out_dir, split, n, regions, defect_frac, misalign_px, size, seed):
    rng = np.random.default_rng(seed)
    split_dir = Path(out_dir) / split
    manifest = []
    for i in range(n):
        region = regions[i % len(regions)]
        gen = TripletSEMGenerator(width=size, height=size, region_type=region)
        with_defect = rng.random() < defect_frac
        tri = gen.generate_triplet(with_defect, misalign_px, rng)

        tid = f"{region}_{split}_{i:05d}"
        d = split_dir / tid
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "ref1.png"), tri["ref1"])
        cv2.imwrite(str(d / "ref2.png"), tri["ref2"])
        cv2.imwrite(str(d / "target.png"), tri["target"])
        cv2.imwrite(str(d / "mask.png"), tri["mask"] * 255)
        manifest.append(dict(id=tid, has_defect=bool(tri["meta"]["has_defect"]),
                             region=region, defect_bbox=tri["meta"]["defect_bbox"],
                             shifts=tri["meta"]["shifts"]))

    with open(split_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    n_def = sum(m["has_defect"] for m in manifest)
    print(f"[{split}] wrote {n} triplets ({n_def} defective) -> {split_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sem_triplet")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--regions", nargs="+", default=["sram", "logic"])
    ap.add_argument("--defect_frac", type=float, default=0.5,
                    help="Fraction of triplets whose target contains a defect. "
                         "Eval/guardrail splits benefit from a mix; training oversampling "
                         "is handled by the dataloader, not here.")
    ap.add_argument("--misalign_px", type=int, default=0,
                    help="Max |die-to-die| integer pixel shift per view (0 = aligned).")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for split, n, s in (("train", args.n_train, 0), ("val", args.n_val, 1),
                        ("test", args.n_test, 2)):
        if n > 0:
            _build_split(args.out, split, n, args.regions, args.defect_frac,
                         args.misalign_px, args.size, args.seed + 1000 * s)


if __name__ == "__main__":
    main()

"""Compute IMG_MEAN / IMG_STD from a directory of triplet TIFFs.

Run this ONCE on the real training data BEFORE training, then paste the printed
values into `triplet_ssl/__init__.py`. The same constants are imported by the
training aug, the eval, and the inference script -- that file is the single
point of truth. Never change them after training (the model is bound to them).

    python triplet_ssl/tools/compute_stats.py --input data/tiff_train [--max-files 500]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from triplet_ssl.data.triplet_dataset import read_tiff3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory of 3-channel TIFFs")
    ap.add_argument("--max-files", type=int, default=500)
    args = ap.parse_args()

    files = sorted(list(Path(args.input).glob("*.tif"))
                   + list(Path(args.input).glob("*.tiff")))[: args.max_files]
    if not files:
        sys.exit(f"no .tif/.tiff under {args.input}")

    # Grayscale-replicated data: all three dies pooled into ONE distribution
    # (target/ref are the same population), single mean/std replicated x3.
    s = ss = n = 0.0
    for f in files:
        for img in read_tiff3(f):                      # target, ref1, ref2
            g = img[:, :, 0].astype(np.float64)        # channels identical
            s += g.sum(); ss += (g ** 2).sum(); n += g.size
    mean = s / n
    std = float(np.sqrt(ss / n - mean ** 2))

    print(f"files: {len(files)}  pixels: {int(n):,}")
    print(f"mean = {mean:.2f}   std = {std:.2f}")
    print("\npaste into triplet_ssl/__init__.py:")
    print(f"IMG_MEAN = ({mean:.2f}, {mean:.2f}, {mean:.2f})")
    print(f"IMG_STD = ({std:.2f}, {std:.2f}, {std:.2f})")
    if std < 10:
        print("\nWARNING: std < 10 -- extremely flat data; check bit depth / contrast.")


if __name__ == "__main__":
    main()

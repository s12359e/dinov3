"""Create a MIMIC DINOv3 ViT-B/16 checkpoint with random weights.

The real continued-pretraining run must initialise from official DINOv3 weights
("never train from scratch"). Until those weights are placed on disk, this produces a
structurally identical random-weight checkpoint so the *entire* pipeline
(load -> student/teacher init -> train -> eval) runs end-to-end. Swapping in the real
`.pth` later requires no code change: `train_triplet.py` loads it through the same hook.

The saved state_dict keys are prefixed with ``backbone.`` to match the loader's key
normalisation (it strips ``backbone.``).

Usage::

    python triplet_ssl/make_mimic_checkpoint.py --out checkpoints/dinov3_vitb16_mimic.pth
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.models.vision_transformer import vit_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/dinov3_vitb16_mimic.pth")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    model = vit_base(patch_size=args.patch_size, img_size=args.img_size)
    model.init_weights()  # proper trunc-normal init, not left uninitialised

    state = {f"backbone.{k}": v for k, v in model.state_dict().items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "mimic": True}, out)
    n = sum(v.numel() for v in model.state_dict().values())
    print(f"Wrote MIMIC checkpoint ({n:,} params) -> {out}")
    print("NOTE: random weights. Replace with official DINOv3 ViT-B/16 for real runs.")


if __name__ == "__main__":
    main()

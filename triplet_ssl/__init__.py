"""Triplet-based continued SSL pretraining on DINOv3 for industrial inspection.

Additive module. Nothing outside `triplet_ssl/` (except the read-only reuse of
`dinov3.*` components and the SEM data generator) is modified.

Shared normalization constants match the SEM synthetic data (grayscale replicated
to 3 channels). Real optical data should recompute these.
"""

# Grayscale-replicated normalization (0-255 scale), shared with sem_defect.
IMG_MEAN = (109.65, 104.81, 75.48)
IMG_STD = (54.32, 39.78, 36.47)

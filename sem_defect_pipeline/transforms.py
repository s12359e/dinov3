"""
Custom MMSeg transforms for the SEM defect pipeline.
"""

import numpy as np
import cv2
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadYOLOAnnotations(BaseTransform):
    """Load semantic segmentation masks from YOLO segmentation label files.

    YOLO segmentation format (.txt):
        Each line: ``class_id x1 y1 x2 y2 ... xn yn``
        Coordinates are normalized to [0, 1].
        Multiple lines = multiple object instances.

    This transform converts polygon annotations into a dense pixel mask
    (``gt_seg_map``) compatible with MMSeg's segmentation pipeline.

    For binary segmentation all object classes are mapped to label 1
    (defect) by default; background is 0.  To preserve the original
    YOLO class IDs set ``binary=False`` and provide ``class_offset=1``
    so that class 0 in YOLO becomes label 1 in the mask (reserving 0
    for background).

    Required Keys:
        - seg_map_path (str): Path to the YOLO .txt label file.
        - img_shape (tuple): (H, W) of the image — set by LoadImageFromFile.

    Added Keys:
        - seg_fields (list): ['gt_seg_map']
        - gt_seg_map (np.ndarray): uint8 mask of shape (H, W).

    Args:
        binary (bool): If True, all YOLO classes → label 1. Default True.
        class_offset (int): Added to YOLO class_id when ``binary=False``.
            Use 1 to reserve 0 for background. Default 1.
    """

    def __init__(self, binary: bool = True, class_offset: int = 1):
        super().__init__()
        self.binary = binary
        self.class_offset = class_offset

    def transform(self, results: dict) -> dict:
        seg_map_path = results['seg_map_path']
        img_shape = results['img_shape']  # (H, W) or (H, W, C)
        H, W = img_shape[0], img_shape[1]

        mask = np.zeros((H, W), dtype=np.uint8)

        try:
            with open(seg_map_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            # No label file → pure background (negative sample)
            lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) < 6:
                # Need at least 3 points (6 values) for a polygon
                continue

            # Denormalize coordinates
            points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * W
                y = coords[i + 1] * H
                points.append([x, y])

            polygon = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

            if self.binary:
                label = 1
            else:
                label = class_id + self.class_offset

            cv2.fillPoly(mask, [polygon], color=int(label))

        results['gt_seg_map'] = mask
        results['seg_fields'] = results.get('seg_fields', [])
        if 'gt_seg_map' not in results['seg_fields']:
            results['seg_fields'].append('gt_seg_map')

        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'binary={self.binary}, '
                f'class_offset={self.class_offset})')

"""
augmentation.py

Random-crop-and-zoom augmentation for detector training.

The detector always squashes a full source image down to a fixed 256x256
input. If every training image happens to have objects occupying a similar
fraction of the frame (true of both our synthetic set and PKLot: ~15-57
spots at a roughly consistent camera distance), the model only ever learns
one narrow band of apparent object scale -- and fails badly on arbitrary
real photos with very different density/resolution/framing (see
report_assets/ for the ivana/john test cases that motivated this).

random_crops() takes each source image and cuts several random sub-regions
at varying zoom levels, keeping the boxes (clipped) that survive within
each crop. Training on the original image *plus* these crops exposes the
model to a much wider range of apparent object sizes using the exact same
underlying data -- no new photos needed.
"""

import numpy as np


def random_crops(img, boxes, class_ids, rng, n_crops=3, min_scale=0.25, max_scale=0.85, min_keep_frac=0.4):
    """boxes: list of [x0, y0, x1, y1] in img's pixel space.
    Returns a list of (crop_img, crop_boxes, crop_class_ids) -- crop_boxes
    are in the crop's own local pixel space. Crops with zero surviving
    boxes are skipped (an empty crop teaches nothing useful and wastes a
    training slot)."""
    H, W = img.shape[:2]
    views = []
    for _ in range(n_crops):
        scale = rng.uniform(min_scale, max_scale)
        cw, ch = max(int(W * scale), 32), max(int(H * scale), 32)
        x0 = int(rng.randint(0, max(W - cw, 1) + 1))
        y0 = int(rng.randint(0, max(H - ch, 1) + 1))
        x1, y1 = min(x0 + cw, W), min(y0 + ch, H)

        crop_boxes, crop_classes = [], []
        for (bx0, by0, bx1, by1), cid in zip(boxes, class_ids):
            ix0, iy0 = max(bx0, x0), max(by0, y0)
            ix1, iy1 = min(bx1, x1), min(by1, y1)
            iw, ih = max(ix1 - ix0, 0), max(iy1 - iy0, 0)
            inter_area = iw * ih
            box_area = max(bx1 - bx0, 1e-6) * max(by1 - by0, 1e-6)
            if inter_area / box_area < min_keep_frac:
                continue
            crop_boxes.append([ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0])
            crop_classes.append(cid)

        if crop_boxes:
            views.append((img[y0:y1, x0:x1], crop_boxes, crop_classes))
    return views

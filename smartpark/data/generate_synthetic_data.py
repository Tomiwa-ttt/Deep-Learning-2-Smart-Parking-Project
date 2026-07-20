"""
generate_synthetic_data.py

Generates a synthetic parking-lot dataset in two complementary formats from
the same underlying scene:

  1. Per-spot crops labeled Empty/Occupied (spot_crops/, manifest.json) --
     used by train.py to train the occupancy CNN classifier.
  2. Full, uncropped lot images with COCO-style object detection annotations
     for two classes, "empty_spot" and "occupied_spot" (annotations.json) --
     used by train_detector.py to train a real object detector that finds
     and classifies spots directly in an unmarked photo, no manual spot
     calibration required.

Each generated lot has a randomized grid layout (rows, columns, spot size,
margin) so spot positions actually vary across images -- a detector trained
on this has to learn to localize spots, not memorize fixed coordinates.

This is a stand-in for real datasets. Swap it out for PKLot / CNRPark-EXT
once downloaded (see data/README.md for instructions and folder format).

Usage:
    python generate_synthetic_data.py --out ./synthetic_dataset --n_lots 40
"""

import argparse
import os
import random
import json
import numpy as np
import cv2

CATEGORIES = [{"id": 0, "name": "empty_spot"}, {"id": 1, "name": "occupied_spot"}]


def draw_parking_lot(seed=None):
    """Draws one synthetic top-down parking lot image with a randomized grid
    layout, painted spot lines, randomly occupied spots, and occasional
    'improperly parked' cars that cross over the spot boundary. Returns
    (image, list_of_spot_dicts)."""
    rng = random.Random(seed)

    spot_rows = rng.choice([2, 3])
    spot_cols = rng.choice([4, 5, 6, 7, 8])
    spot_w = rng.randint(70, 95)
    spot_h = rng.randint(120, 160)
    margin = rng.randint(15, 32)

    w = spot_cols * spot_w + margin * 2
    h = spot_rows * spot_h + margin * 2
    img = np.full((h, w, 3), (70, 70, 70), dtype=np.uint8)  # asphalt gray

    spots = []
    spot_id = 0
    for r in range(spot_rows):
        for c in range(spot_cols):
            # small per-spot jitter so painted lines aren't perfectly
            # regular -- keeps the layout genuinely variable, not just
            # variable-per-image-but-fixed-within-image
            jx, jy = rng.randint(-3, 3), rng.randint(-3, 3)
            x0 = margin + c * spot_w + jx
            y0 = margin + r * spot_h + jy
            x1 = x0 + spot_w
            y1 = y0 + spot_h

            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)

            occupied = rng.random() < 0.55
            improperly_parked = False
            car_box = None

            if occupied:
                car_color = tuple(int(v) for v in rng.choices(
                    [(180, 60, 60), (60, 60, 180), (200, 200, 200), (30, 30, 30), (60, 140, 200)]
                )[0])

                pad_x = int(spot_w * 0.12)
                pad_y = int(spot_h * 0.08)
                cx0, cy0 = x0 + pad_x, y0 + pad_y
                cx1, cy1 = x1 - pad_x, y1 - pad_y

                if rng.random() < 0.22:
                    improperly_parked = True
                    shift_x = int(spot_w * rng.uniform(0.35, 0.6)) * rng.choice([-1, 1])
                    shift_y = int(spot_h * rng.uniform(0.05, 0.15)) * rng.choice([-1, 1])
                    cx0 += shift_x
                    cx1 += shift_x
                    cy0 += shift_y
                    cy1 += shift_y

                cx0c, cy0c = max(cx0, 0), max(cy0, 0)
                cx1c, cy1c = min(cx1, w), min(cy1, h)
                cv2.rectangle(img, (cx0c, cy0c), (cx1c, cy1c), car_color, -1)
                cv2.rectangle(img, (cx0c, cy0c), (cx1c, cy1c), (20, 20, 20), 2)
                car_box = [cx0c, cy0c, cx1c, cy1c]  # clipped, used for detection ground truth

            spots.append({
                "id": spot_id,
                "bbox": [x0, y0, x1, y1],
                "occupied": occupied,
                "improperly_parked": improperly_parked,
                "car_bbox": car_box,
            })
            spot_id += 1

    noise = rng.randint(-10, 10)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return img, spots


def crop_spots(img, spots, out_size=64):
    """Crop each spot region (with small context padding) to a fixed size,
    matching the PKLot/CNRPark-EXT convention of per-spot image patches."""
    crops = []
    h, w = img.shape[:2]
    for s in spots:
        x0, y0, x1, y1 = s["bbox"]
        pad = 6
        x0p, y0p = max(x0 - pad, 0), max(y0 - pad, 0)
        x1p, y1p = min(x1 + pad, w), min(y1 + pad, h)
        crop = img[y0p:y1p, x0p:x1p]
        crop = cv2.resize(crop, (out_size, out_size))
        crops.append(crop)
    return crops


def detection_box_for_spot(spot):
    """The ground-truth detection box differs from the painted spot outline
    for occupied spots: it's the actual car footprint (which shifts for
    improperly-parked cases), not the static marked rectangle. This is what
    makes the task real localization rather than fixed-position lookup."""
    if spot["occupied"]:
        return spot["car_bbox"], 1  # occupied_spot
    return spot["bbox"], 0  # empty_spot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./synthetic_dataset")
    ap.add_argument("--n_lots", type=int, default=40, help="number of full lot images to generate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    lots_dir = os.path.join(args.out, "full_lots")
    spots_empty_dir = os.path.join(args.out, "spot_crops", "empty")
    spots_occ_dir = os.path.join(args.out, "spot_crops", "occupied")
    for d in [lots_dir, spots_empty_dir, spots_occ_dir]:
        os.makedirs(d, exist_ok=True)

    manifest = []
    coco_images = []
    coco_annotations = []
    ann_id = 0
    crop_i = 0

    for lot_i in range(args.n_lots):
        img, spots = draw_parking_lot(seed=args.seed * 1000 + lot_i)
        h, w = img.shape[:2]
        lot_path = os.path.join(lots_dir, f"lot_{lot_i:04d}.jpg")
        cv2.imwrite(lot_path, img)

        coco_images.append({
            "id": lot_i,
            "file_name": os.path.basename(lot_path),
            "width": w,
            "height": h,
        })

        crops = crop_spots(img, spots)
        for spot, crop in zip(spots, crops):
            label_dir = spots_occ_dir if spot["occupied"] else spots_empty_dir
            crop_path = os.path.join(label_dir, f"crop_{crop_i:05d}.jpg")
            cv2.imwrite(crop_path, crop)
            manifest.append({
                "crop_path": crop_path,
                "lot_path": lot_path,
                **spot,
            })
            crop_i += 1

            box, category_id = detection_box_for_spot(spot)
            bx0, by0, bx1, by1 = box
            bx0, by0 = max(bx0, 0), max(by0, 0)
            bx1, by1 = min(bx1, w), min(by1, h)
            bw, bh = max(bx1 - bx0, 1), max(by1 - by0, 1)
            coco_annotations.append({
                "id": ann_id,
                "image_id": lot_i,
                "category_id": category_id,
                "bbox": [bx0, by0, bw, bh],  # COCO format: [x, y, width, height]
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(args.out, "annotations.json"), "w") as f:
        json.dump({
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": CATEGORIES,
        }, f, indent=2)

    n_occ = sum(1 for m in manifest if m["occupied"])
    n_bad = sum(1 for m in manifest if m["improperly_parked"])
    print(f"Generated {args.n_lots} lot images, {len(manifest)} spot instances.")
    print(f"  Occupied: {n_occ}  Empty: {len(manifest) - n_occ}")
    print(f"  Improperly parked (subset of occupied): {n_bad}")
    print(f"Detection annotations: {len(coco_annotations)} boxes across 2 classes "
          f"(empty_spot, occupied_spot) -> {os.path.join(args.out, 'annotations.json')}")
    print(f"Saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

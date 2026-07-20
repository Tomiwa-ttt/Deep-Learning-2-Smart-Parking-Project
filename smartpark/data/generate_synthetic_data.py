"""
generate_synthetic_data.py

Generates a synthetic parking-lot dataset (spot crops labeled Empty/Occupied,
plus "properly parked" vs "improperly parked" ground truth) so the full
training + inference pipeline can be built, run, and demoed *today* without
needing to collect real photos first.

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


def draw_parking_lot(spot_rows=2, spot_cols=5, spot_w=80, spot_h=140, margin=20, seed=None):
    """Draws one synthetic top-down parking lot image with painted spot lines,
    randomly occupied spots, and occasional 'improperly parked' cars that
    cross over the spot boundary. Returns (image, list_of_spot_dicts)."""
    rng = random.Random(seed)

    w = spot_cols * spot_w + margin * 2
    h = spot_rows * spot_h + margin * 2
    img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)  # asphalt-ish green/gray base
    img[:, :] = (70, 70, 70)  # asphalt gray

    spots = []
    spot_id = 0
    for r in range(spot_rows):
        for c in range(spot_cols):
            x0 = margin + c * spot_w
            y0 = margin + r * spot_h
            x1 = x0 + spot_w
            y1 = y0 + spot_h

            # paint spot boundary lines (white)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)

            occupied = rng.random() < 0.55
            improperly_parked = False
            car_box = None

            if occupied:
                car_color = tuple(int(v) for v in rng.choices(
                    [(180, 60, 60), (60, 60, 180), (200, 200, 200), (30, 30, 30), (60, 140, 200)]
                )[0])

                # Normal car footprint, slightly smaller than the spot
                pad_x = int(spot_w * 0.12)
                pad_y = int(spot_h * 0.08)
                cx0, cy0 = x0 + pad_x, y0 + pad_y
                cx1, cy1 = x1 - pad_x, y1 - pad_y

                # ~22% chance the car is badly parked: shifted/rotated so it
                # crosses the spot boundary into a neighbor or the lane
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
                car_box = [cx0, cy0, cx1, cy1]  # uncapped, used for IoU math

            spots.append({
                "id": spot_id,
                "bbox": [x0, y0, x1, y1],
                "occupied": occupied,
                "improperly_parked": improperly_parked,
                "car_bbox": car_box,
            })
            spot_id += 1

    # slight random noise for lighting variation realism
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
    crop_i = 0
    for lot_i in range(args.n_lots):
        img, spots = draw_parking_lot(seed=args.seed * 1000 + lot_i)
        lot_path = os.path.join(lots_dir, f"lot_{lot_i:04d}.jpg")
        cv2.imwrite(lot_path, img)

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

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    n_occ = sum(1 for m in manifest if m["occupied"])
    n_bad = sum(1 for m in manifest if m["improperly_parked"])
    print(f"Generated {args.n_lots} lot images, {len(manifest)} spot crops.")
    print(f"  Occupied: {n_occ}  Empty: {len(manifest) - n_occ}")
    print(f"  Improperly parked (subset of occupied): {n_bad}")
    print(f"Saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

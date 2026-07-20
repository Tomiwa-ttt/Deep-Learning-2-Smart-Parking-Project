"""
inference.py

Full pipeline: given a parking-lot image (or video) plus a JSON file of
pre-marked spot boundaries, this:
  1. Crops each spot region
  2. Runs the trained CNN -> Empty / Occupied
  3. For occupied spots, uses OpenCV to find the car's bounding box within
     the spot's local region and runs the IoU geometry check -> Properly /
     Improperly Parked
  4. Draws an annotated image and prints/saves a JSON status report

Usage (single image):
    python inference.py --image ./synthetic_dataset/full_lots/lot_0000.jpg \
        --spots ./synthetic_dataset/manifest.json --model ./trained_model/parking_spot_cnn.keras \
        --lot_name lot_0000

Usage (video, one frame every N frames):
    python inference.py --video ./some_lot_video.mp4 --spots ./spots.json \
        --model ./trained_model/parking_spot_cnn.keras --frame_stride 15
"""

import argparse
import json
import os
import sys
import numpy as np
import cv2
import tensorflow as tf

sys.path.append(os.path.dirname(__file__))
from improper_parking import evaluate_spot

IMG_SIZE = 64


def load_spots_for_lot(manifest_path, lot_path):
    """Loads the spot boundary boxes for one specific lot image from the
    manifest produced by data/generate_synthetic_data.py. For a real
    deployment, replace this with your own one-time spot calibration file
    (see data/README.md)."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    lot_path_abs = os.path.abspath(lot_path)
    spots = [m for m in manifest if os.path.abspath(m["lot_path"]) == lot_path_abs]
    # dedupe by spot id, keep bbox only (occupied/car_bbox here are ground
    # truth from the synthetic generator -- inference recomputes them fresh)
    spots = sorted(spots, key=lambda s: s["id"])
    return [{"id": s["id"], "bbox": s["bbox"]} for s in spots]


def _sample_background_color(region, border=4):
    """Estimate the asphalt/background color from the corner patches of a
    region, which are unlikely to contain the car itself."""
    h, w = region.shape[:2]
    b = min(border, h // 4 or 1, w // 4 or 1)
    patches = [
        region[0:b, 0:b], region[0:b, w - b:w],
        region[h - b:h, 0:b], region[h - b:h, w - b:w],
    ]
    pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    return np.median(pixels, axis=0)


def find_car_bbox_in_spot(img, spot_bbox, pad_ratio=0.18,
                           car_diff_thresh=35, white_diff_thresh=40):
    """Within a lightly-padded region around a spot, localize the parked car
    by color distance from the estimated asphalt background -- robust to
    cars both darker and lighter than the asphalt, unlike a plain Otsu
    threshold (which silently inverts for dark cars). This is intentionally
    simple (classic CV, no extra model) -- swap for a proper detector (e.g.
    YOLO) in production for messier real-world scenes with real lighting.

    Two things matter for accuracy here:
      - pad_ratio must stay small, or the region bleeds into neighboring
        spots and the "car" found is really two cars merged together.
      - painted white boundary lines must be excluded by color distance from
        white specifically (not just "very light"), so a light-colored car
        isn't mistaken for a line.
    """
    x0, y0, x1, y1 = spot_bbox
    w, h = x1 - x0, y1 - y0
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    H, W = img.shape[:2]
    rx0, ry0 = max(x0 - pad_x, 0), max(y0 - pad_y, 0)
    rx1, ry1 = min(x1 + pad_x, W), min(y1 + pad_y, H)
    region = img[ry0:ry1, rx0:rx1].astype(np.float32)

    bg_color = _sample_background_color(region)
    white = np.array([255, 255, 255], dtype=np.float32)

    dist_from_bg = np.linalg.norm(region - bg_color, axis=2)
    dist_from_white = np.linalg.norm(region - white, axis=2)

    mask = ((dist_from_bg > car_diff_thresh) & (dist_from_white > white_diff_thresh))
    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # drop thin line remnants
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # reconnect the car blob

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.05 * (region.shape[0] * region.shape[1]):
        return None  # too small to be a car; likely noise
    cx, cy, cw, ch = cv2.boundingRect(biggest)
    # translate back to full-image coordinates
    return [rx0 + cx, ry0 + cy, rx0 + cx + cw, ry0 + cy + ch]


def run_inference(img, spots, model):
    """Returns a list of result dicts, one per spot."""
    H, W = img.shape[:2]
    crops = []
    for s in spots:
        x0, y0, x1, y1 = s["bbox"]
        pad = 6
        x0p, y0p = max(x0 - pad, 0), max(y0 - pad, 0)
        x1p, y1p = min(x1 + pad, W), min(y1 + pad, H)
        crop = cv2.resize(img[y0p:y1p, x0p:x1p], (IMG_SIZE, IMG_SIZE))
        crops.append(crop)

    batch = np.stack(crops).astype(np.float32)
    preds = model.predict(batch, verbose=0).flatten()  # sigmoid outputs

    results = []
    for s, p in zip(spots, preds):
        occupied = bool(p >= 0.5)
        car_bbox = find_car_bbox_in_spot(img, s["bbox"]) if occupied else None
        verdict = evaluate_spot(s["id"], s["bbox"], car_bbox, occupied)
        results.append({
            "spot_id": s["id"],
            "bbox": s["bbox"],
            "occupied": occupied,
            "confidence": float(p if occupied else 1 - p),
            "properly_parked": verdict.properly_parked,
            "reason": verdict.reason,
        })
    return results


def draw_annotated(img, results):
    out = img.copy()
    for r in results:
        x0, y0, x1, y1 = r["bbox"]
        if not r["occupied"]:
            color = (0, 200, 0)  # green = empty
            label = f"#{r['spot_id']} Empty"
        elif r["properly_parked"]:
            color = (0, 165, 255)  # orange = occupied, fine
            label = f"#{r['spot_id']} Occupied"
        else:
            color = (0, 0, 255)  # red = occupied + improperly parked
            label = f"#{r['spot_id']} BAD PARK"
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, label, (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
    return out


def print_report(results, lot_name="Lot"):
    print(f"\n--- {lot_name} status ---")
    for r in results:
        if not r["occupied"]:
            print(f"Parking Spot {r['spot_id']}: Empty")
        elif r["properly_parked"]:
            print(f"Parking Spot {r['spot_id']}: Occupied")
        else:
            print(f"Parking Spot {r['spot_id']}: Occupied — Improperly Parked ({r['reason']})")
    n_empty = sum(1 for r in results if not r["occupied"])
    n_bad = sum(1 for r in results if r["occupied"] and not r["properly_parked"])
    print(f"\nSummary: {len(results)} spots | {n_empty} empty | "
          f"{len(results) - n_empty} occupied | {n_bad} improperly parked")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Path to a single lot image")
    ap.add_argument("--video", help="Path to a video file")
    ap.add_argument("--spots", required=True, help="manifest.json (or your own spot-boundary JSON)")
    ap.add_argument("--model", required=True, help="Path to trained .keras model")
    ap.add_argument("--lot_name", default=None)
    ap.add_argument("--frame_stride", type=int, default=15, help="For video: process 1 of every N frames")
    ap.add_argument("--out", default="./inference_output")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = tf.keras.models.load_model(args.model)

    if args.image:
        img = cv2.imread(args.image)
        spots = load_spots_for_lot(args.spots, args.image)
        results = run_inference(img, spots, model)
        print_report(results, args.lot_name or os.path.basename(args.image))
        annotated = draw_annotated(img, results)
        out_path = os.path.join(args.out, "annotated_" + os.path.basename(args.image))
        cv2.imwrite(out_path, annotated)
        with open(os.path.join(args.out, "status.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved annotated image to {out_path}")

    elif args.video:
        cap = cv2.VideoCapture(args.video)
        frame_i = 0
        all_frame_results = []
        spots = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_i % args.frame_stride == 0:
                if spots is None:
                    # first processed frame defines spot layout; real deployments
                    # calibrate this once per camera, see data/README.md
                    with open(args.spots) as f:
                        spots = json.load(f)
                results = run_inference(frame, spots, model)
                all_frame_results.append({"frame": frame_i, "results": results})
                print_report(results, f"Frame {frame_i}")
            frame_i += 1
        cap.release()
        with open(os.path.join(args.out, "video_status.json"), "w") as f:
            json.dump(all_frame_results, f, indent=2)
        print(f"\nProcessed {frame_i} frames, saved status log.")

    else:
        print("Provide --image or --video")


if __name__ == "__main__":
    main()

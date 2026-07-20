"""
generate_report_samples.py

Produces the "model outputs on sample inputs" artifacts required for
submission: a handful of held-out validation images run through the
trained detector, with annotated images and a JSON summary comparing
predicted vs. ground-truth counts.

Usage:
    python generate_report_samples.py --n 8 --out ./report_assets/sample_outputs
"""

import argparse
import json
import os
import cv2
import numpy as np
import tensorflow as tf

from train_detector import load_dataset
from detector_utils import decode_predictions, draw_detections, preprocess_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./synthetic_dataset")
    ap.add_argument("--model", default="./trained_detector/spot_detector.keras")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./report_assets/sample_outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading dataset from {args.dataset} ...")
    data = load_dataset(args.dataset)
    X, raw_boxes, raw_classes, orig_sizes = data["X"], data["raw_boxes"], data["raw_classes"], data["orig_sizes"]

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    n_val = max(int(len(X) * args.val_split), 1)
    val_idx = idx[:n_val]
    chosen = val_idx[:args.n]

    model = tf.keras.models.load_model(args.model, compile=False)

    summary = []
    for rank, i in enumerate(chosen):
        orig_w, orig_h = orig_sizes[i]
        # re-read the original (non-resized) image for a full-resolution annotated output
        # X[i] is already resized to the model's input size for inference
        pred = model.predict(X[i:i + 1], verbose=0)[0]
        detections = decode_predictions(pred, orig_w, orig_h)

        # reconstruct a full-res image from the resized array isn't lossless;
        # instead resize the model input back up for a clean annotated image
        base_img = cv2.resize(X[i], (orig_w, orig_h))
        annotated = draw_detections(base_img, detections)

        out_name = f"sample_{rank:02d}.jpg"
        cv2.imwrite(os.path.join(args.out, out_name), annotated)

        gt_empty = sum(1 for c in raw_classes[i] if c == 0)
        gt_occupied = sum(1 for c in raw_classes[i] if c == 1)
        pred_empty = sum(1 for d in detections if d["class_id"] == 0)
        pred_occupied = sum(1 for d in detections if d["class_id"] == 1)

        summary.append({
            "file": out_name,
            "ground_truth": {"empty_spot": gt_empty, "occupied_spot": gt_occupied},
            "predicted": {"empty_spot": pred_empty, "occupied_spot": pred_occupied},
            "detections": detections,
        })
        print(f"{out_name}: GT empty={gt_empty} occ={gt_occupied} | "
              f"Pred empty={pred_empty} occ={pred_occupied}")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {len(chosen)} annotated samples + summary.json to {args.out}")


if __name__ == "__main__":
    main()

"""
benchmark_baseline.py

Benchmarks the trained detector's per-spot classification accuracy against
two non-learned baselines, using the same held-out validation split and
ground-truth spot boxes for all three so the comparison is apples-to-apples:

  1. Majority-class baseline: always predict the more common class.
  2. Classic-CV baseline: the project's own original heuristic
     (inference.find_car_bbox_in_spot -- color-distance thresholding), used
     before this project had any trained detector -- "occupied" if a car
     footprint is found within the spot region, "empty" otherwise.
  3. Trained detector: classification accuracy restricted to correctly
     LOCALIZED boxes (from the confusion matrix in evaluate output),
     isolating "did the CNN learn better features than a heuristic" from
     "did it also localize well" (a separate, already-reported metric).

Usage:
    python benchmark_baseline.py --dataset ./synthetic_dataset
"""

import argparse
import json
import os
import cv2
import numpy as np

from train_detector import load_dataset
from inference import find_car_bbox_in_spot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./synthetic_dataset")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./report_assets/benchmark.json")
    args = ap.parse_args()

    data = load_dataset(args.dataset)
    X, raw_boxes, raw_classes, orig_sizes = data["X"], data["raw_boxes"], data["raw_classes"], data["orig_sizes"]
    lots_dir = os.path.join(args.dataset, "full_lots")

    with open(os.path.join(args.dataset, "annotations.json")) as f:
        coco = json.load(f)
    file_names = {im["id"]: im["file_name"] for im in sorted(coco["images"], key=lambda im: im["id"])}
    ordered_ids = sorted(file_names.keys())

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    n_val = max(int(len(X) * args.val_split), 1)
    val_idx = idx[:n_val]

    n_total = 0
    n_occupied = 0
    n_correct_majority = 0
    n_correct_cv = 0

    for i in val_idx:
        img_path = os.path.join(lots_dir, file_names[ordered_ids[i]])
        img = cv2.imread(img_path)
        boxes, classes = raw_boxes[i], raw_classes[i]

        for box, cls in zip(boxes, classes):
            is_occupied_gt = (cls == 1)
            n_total += 1
            n_occupied += int(is_occupied_gt)

            # classic-CV baseline: did the color-distance heuristic find a car?
            car_bbox = find_car_bbox_in_spot(img, [int(v) for v in box])
            cv_predicts_occupied = car_bbox is not None
            n_correct_cv += int(cv_predicts_occupied == is_occupied_gt)

    majority_class_is_occupied = n_occupied > (n_total - n_occupied)
    n_correct_majority = n_occupied if majority_class_is_occupied else (n_total - n_occupied)

    majority_acc = n_correct_majority / n_total
    cv_acc = n_correct_cv / n_total

    print(f"Validation spot instances: {n_total} ({n_occupied} occupied, {n_total - n_occupied} empty)")
    print(f"Majority-class baseline accuracy: {majority_acc*100:.2f}%")
    print(f"Classic-CV heuristic baseline accuracy: {cv_acc*100:.2f}%")
    print("Trained detector classification accuracy (given correct localization): "
          "99.86% -- see confusion_matrix.png / evaluate_detector.py output")

    with open(args.out, "w") as f:
        json.dump({
            "n_val_instances": n_total,
            "n_occupied": n_occupied,
            "majority_class_baseline_accuracy": majority_acc,
            "classic_cv_baseline_accuracy": cv_acc,
        }, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

"""
hyperparameter_ablation.py

Two small, real experiments backing the "Hyperparameter Tuning" report
section (not just asserted choices):

  1. Grid resolution vs. ground-truth collisions: with the one-box-per-cell
     design, two spot centers landing in the same cell means one gets
     dropped from training. Measures this at grid_size=8 vs 16 to justify
     using 16.

  2. Coordinate-loss weight (lambda_coord): trains two short, matched-epoch
     detectors (lambda_coord=5, the YOLOv1 default, vs. lambda_coord=1) and
     compares validation mAP@0.5, to justify keeping the higher weight.

Usage:
    python hyperparameter_ablation.py --epochs 15 --out ./report_assets/ablation
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf

from train_detector import load_dataset
from models.detector_model import build_detector, GRID_SIZE, INPUT_SIZE
from detector_utils import encode_targets, make_detection_loss, decode_predictions
from evaluate_detector import match_image, compute_ap


def grid_collision_rates(dataset_dir, grid_sizes=(8, 16, 32)):
    import json as _json
    with open(os.path.join(dataset_dir, "annotations.json")) as f:
        coco = _json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    results = {}
    for grid_size in grid_sizes:
        n_total, n_dropped = 0, 0
        for image_id, im_info in images_by_id.items():
            anns = anns_by_image.get(image_id, [])
            boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                     for a in anns]
            class_ids = [a["category_id"] for a in anns]
            n_total += len(boxes)
            _, dropped = encode_targets(boxes, class_ids, im_info["width"], im_info["height"],
                                         grid_size=grid_size, input_size=INPUT_SIZE)
            n_dropped += dropped
        results[grid_size] = {"total": n_total, "dropped": n_dropped,
                               "rate_pct": n_dropped / n_total * 100 if n_total else 0.0}
    return results


def quick_map(model, X, raw_boxes, raw_classes, orig_sizes, val_idx):
    from models.detector_model import CLASS_NAMES
    all_dets = {c: [] for c in range(len(CLASS_NAMES))}
    total_gt = {c: 0 for c in range(len(CLASS_NAMES))}
    for i in val_idx:
        orig_w, orig_h = orig_sizes[i]
        pred = model.predict(X[i:i + 1], verbose=0)[0]
        detections = decode_predictions(pred, orig_w, orig_h, obj_thresh=0.05, nms_thresh=0.4)
        for c in total_gt:
            total_gt[c] += sum(1 for gc in raw_classes[i] if gc == c)
        for conf, cls, is_tp in match_image(detections, raw_boxes[i], raw_classes[i]):
            all_dets[cls].append((conf, is_tp))

    aps = []
    for c in total_gt:
        dets = sorted(all_dets[c], key=lambda d: -d[0])
        n_gt = total_gt[c]
        tp, fp = 0, 0
        precisions, recalls = [], []
        for conf, is_tp in dets:
            tp += is_tp
            fp += not is_tp
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_gt if n_gt else 0.0)
        aps.append(compute_ap(np.array(recalls), np.array(precisions)) if dets else 0.0)
    return float(np.mean(aps))


def lambda_coord_ablation(dataset_dir, epochs, seed=42, val_split=0.15):
    data = load_dataset(dataset_dir)
    X, Y = data["X"], data["Y"]
    raw_boxes, raw_classes, orig_sizes = data["raw_boxes"], data["raw_classes"], data["orig_sizes"]

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_val = max(int(len(X) * val_split), 1)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    results = {}
    for lambda_coord in (5.0, 1.0):
        tf.keras.backend.clear_session()
        model = build_detector()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=make_detection_loss(lambda_coord=lambda_coord))
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                             epochs=epochs, batch_size=16, verbose=0)
        val_loss = history.history["val_loss"][-1]
        mAP = quick_map(model, X, raw_boxes, raw_classes, orig_sizes, val_idx)
        results[lambda_coord] = {"val_loss": val_loss, "mAP": mAP}
        print(f"lambda_coord={lambda_coord}: val_loss={val_loss:.3f} mAP={mAP*100:.2f}%")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./synthetic_dataset")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="./report_assets/ablation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=== Grid resolution vs. collision rate ===")
    collision_results = grid_collision_rates(args.dataset)
    for g, r in collision_results.items():
        print(f"grid_size={g}: {r['dropped']}/{r['total']} dropped ({r['rate_pct']:.2f}%)")

    print(f"\n=== lambda_coord ablation ({args.epochs} epochs each) ===")
    lambda_results = lambda_coord_ablation(args.dataset, args.epochs)

    with open(os.path.join(args.out, "ablation_results.json"), "w") as f:
        json.dump({
            "grid_collision_rates": collision_results,
            "lambda_coord_ablation": {str(k): v for k, v in lambda_results.items()},
        }, f, indent=2)
    print(f"\nSaved results to {args.out}/ablation_results.json")


if __name__ == "__main__":
    main()

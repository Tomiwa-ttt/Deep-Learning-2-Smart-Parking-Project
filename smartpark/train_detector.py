"""
train_detector.py

Trains the single-shot spot detector (models/detector_model.py) on a
COCO-style annotations.json produced by data/generate_synthetic_data.py
(or, for real data, a compatible export -- see data/README.md).

Usage:
    python train_detector.py --dataset ./synthetic_dataset --epochs 30 \
        --out ./trained_detector
"""

import argparse
import json
import os
import sys
import numpy as np
import cv2
import tensorflow as tf

sys.path.append(os.path.dirname(__file__))
from models.detector_model import build_detector, INPUT_SIZE, GRID_SIZE, NUM_CLASSES
from detector_utils import encode_targets, detection_loss


def load_dataset(dataset_dir, input_size=INPUT_SIZE, grid_size=GRID_SIZE, num_classes=NUM_CLASSES):
    """Returns a dict with the model-ready arrays (X, Y) plus the raw,
    un-lossy ground truth (raw_boxes/raw_classes/orig_sizes, one entry per
    image) so evaluation can score against the true boxes rather than the
    grid-encoded (and occasionally collision-dropped) targets."""
    with open(os.path.join(dataset_dir, "annotations.json")) as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    X, Y = [], []
    raw_boxes, raw_classes, orig_sizes = [], [], []
    n_dropped_total = 0
    n_boxes_total = 0
    lots_dir = os.path.join(dataset_dir, "full_lots")

    for image_id, im_info in sorted(images_by_id.items()):
        img_path = os.path.join(lots_dir, im_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        anns = anns_by_image.get(image_id, [])
        boxes, class_ids = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            class_ids.append(ann["category_id"])
        n_boxes_total += len(boxes)

        target, n_dropped = encode_targets(boxes, class_ids, orig_w, orig_h,
                                            grid_size=grid_size, input_size=input_size,
                                            num_classes=num_classes)
        n_dropped_total += n_dropped

        resized = cv2.resize(img, (input_size, input_size))
        X.append(resized)
        Y.append(target)
        raw_boxes.append(boxes)
        raw_classes.append(class_ids)
        orig_sizes.append((orig_w, orig_h))

    X = np.stack(X).astype(np.uint8)
    Y = np.stack(Y).astype(np.float32)
    return {
        "X": X, "Y": Y,
        "raw_boxes": raw_boxes, "raw_classes": raw_classes, "orig_sizes": orig_sizes,
        "n_boxes_total": n_boxes_total, "n_dropped_total": n_dropped_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./synthetic_dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--out", default="./trained_detector")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume_from", default=None,
                     help="Path to an existing .keras detector to fine-tune (e.g. the "
                          "synthetic-trained checkpoint, continued on real data)")
    ap.add_argument("--lr", type=float, default=1e-3,
                     help="Lower this (e.g. 1e-4) when fine-tuning from --resume_from")
    args = ap.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    data = load_dataset(args.dataset)
    X, Y = data["X"], data["Y"]
    n_boxes_total, n_dropped_total = data["n_boxes_total"], data["n_dropped_total"]
    print(f"Loaded {len(X)} images, {n_boxes_total} ground-truth boxes "
          f"({n_dropped_total} dropped to grid-cell collisions, "
          f"{n_dropped_total / max(n_boxes_total, 1) * 100:.1f}%).")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    n_val = max(int(len(X) * args.val_split), 1)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        model = tf.keras.models.load_model(args.resume_from, compile=False)
    else:
        model = build_detector()
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=detection_loss)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "spot_detector.keras")
    model.save(model_path)
    print(f"\nSaved trained detector to {model_path}")

    val_loss = model.evaluate(X_val, Y_val, verbose=0)
    with open(os.path.join(args.out, "training_report.txt"), "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Resumed from: {args.resume_from or 'scratch'}\n")
        f.write(f"Images: {len(X)} (train {len(X_train)} / val {len(X_val)})\n")
        f.write(f"Ground-truth boxes: {n_boxes_total} "
                f"({n_dropped_total} dropped to grid-cell collisions, "
                f"{n_dropped_total / max(n_boxes_total, 1) * 100:.1f}%)\n")
        f.write(f"Epochs run: {len(history.history['loss'])}\n")
        f.write(f"Final train loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final val loss: {val_loss:.4f}\n")
    print(f"Final val loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()

"""
train_detector_expanded.py

Second-stage rehearsal fine-tune that pulls in the much larger real PKLot
pool that train_detector_mixed.py never used: that script's --real_dataset
was built from only the Roboflow "test" split (1,242 images). The "train"
(8,691) and "valid" (2,483) splits sit alongside it, fully annotated,
totalling 12,416 real images once combined.

Using all 12,416 at the original 4x crop-augmentation ratio would need a
training array in the tens of GB (this machine has 16GB RAM, no GPU) --
infeasible. Instead this script pools all three real datasets together,
randomly samples a bounded subset (--max_real_images, default 5000) from
the combined 12,416, and skips crop augmentation for this stage entirely:
with 5,000 distinct real photographs already available, more unique real
scenes are the point, not synthetic crops of a smaller pool (that was
train_detector_mixed.py's job, solving a different problem).

Resumes from trained_detector_multibox_mixed/spot_detector.keras (the
current best checkpoint, already rehearsal fine-tuned) at a low LR, mixes
in synthetic data every epoch (same rehearsal principle, again to avoid
forgetting), and evaluates the result against both a fresh real held-out
split (drawn from the new combined pool -- not the same 186 images the
68.76% figure was measured on, composition changed) and the original
synthetic validation set (to confirm no regression there).

Usage:
    python train_detector_expanded.py \
        --real_datasets ./real_pklot_dataset ./real_pklot_dataset_train ./real_pklot_dataset_valid \
        --synthetic_dataset ./synthetic_dataset \
        --resume_from ./trained_detector_multibox_mixed/spot_detector.keras \
        --max_real_images 5000 --epochs 25 --lr 5e-5 \
        --out ./trained_detector_expanded
"""

import argparse
import time

import numpy as np
import tensorflow as tf

from train_detector import load_dataset
from detector_utils import detection_loss, decode_predictions
from evaluate_detector import match_image, compute_ap
from models.detector_model import CLASS_NAMES


def concat_real_datasets(paths):
    X_parts, Y_parts, raw_boxes, raw_classes, orig_sizes, image_paths = [], [], [], [], [], []
    for p in paths:
        print(f"Loading {p} ...")
        d = load_dataset(p)
        X_parts.append(d["X"])
        Y_parts.append(d["Y"])
        raw_boxes.extend(d["raw_boxes"])
        raw_classes.extend(d["raw_classes"])
        orig_sizes.extend(d["orig_sizes"])
        image_paths.extend(d["image_paths"])
        print(f"  -> {len(d['X'])} images")
    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    return {
        "X": X, "Y": Y,
        "raw_boxes": raw_boxes, "raw_classes": raw_classes,
        "orig_sizes": orig_sizes, "image_paths": image_paths,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_datasets", nargs="+",
                     default=["./real_pklot_dataset", "./real_pklot_dataset_train", "./real_pklot_dataset_valid"])
    ap.add_argument("--synthetic_dataset", default="./synthetic_dataset")
    ap.add_argument("--max_real_images", type=int, default=5000)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume_from", default="./trained_detector_multibox_mixed/spot_detector.keras")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="./trained_detector_expanded")
    args = ap.parse_args()

    t_start = time.time()
    rng = np.random.RandomState(args.seed)

    print(f"Loading synthetic dataset from {args.synthetic_dataset} ...")
    syn = load_dataset(args.synthetic_dataset)

    real = concat_real_datasets(args.real_datasets)
    n_real_total = len(real["X"])
    print(f"\nCombined real pool: {n_real_total} images from {len(args.real_datasets)} datasets")

    if n_real_total > args.max_real_images:
        sample_idx = rng.choice(n_real_total, size=args.max_real_images, replace=False)
        print(f"Sampling {args.max_real_images} of {n_real_total} real images "
              f"(memory/time bound for this machine)")
        real = {
            "X": real["X"][sample_idx],
            "Y": real["Y"][sample_idx],
            "raw_boxes": [real["raw_boxes"][i] for i in sample_idx],
            "raw_classes": [real["raw_classes"][i] for i in sample_idx],
            "orig_sizes": [real["orig_sizes"][i] for i in sample_idx],
            "image_paths": [real["image_paths"][i] for i in sample_idx],
        }

    def split(n):
        idx = rng.permutation(n)
        n_val = max(int(n * args.val_split), 1)
        return idx[n_val:], idx[:n_val]

    syn_train_idx, syn_val_idx = split(len(syn["X"]))
    real_train_idx, real_val_idx = split(len(real["X"]))

    X_train = np.concatenate([syn["X"][syn_train_idx], real["X"][real_train_idx]], axis=0)
    Y_train = np.concatenate([syn["Y"][syn_train_idx], real["Y"][real_train_idx]], axis=0)
    X_val_syn, Y_val_syn = syn["X"][syn_val_idx], syn["Y"][syn_val_idx]
    X_val_real, Y_val_real = real["X"][real_val_idx], real["Y"][real_val_idx]

    print(f"\nTrain: {len(X_train)} (synthetic {len(syn_train_idx)} + real {len(real_train_idx)}, no crop augmentation this stage)")
    print(f"Val: synthetic {len(X_val_syn)}, real {len(X_val_real)}")
    print(f"Data loading took {time.time()-t_start:.0f}s\n")

    print(f"Resuming from {args.resume_from}")
    model = tf.keras.models.load_model(args.resume_from, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=detection_loss)

    t_train = time.time()
    history = model.fit(
        X_train, Y_train,
        validation_data=(np.concatenate([X_val_syn, X_val_real], axis=0),
                          np.concatenate([Y_val_syn, Y_val_real], axis=0)),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
    )
    print(f"\nTraining took {time.time()-t_train:.0f}s")

    import os
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "spot_detector.keras"))

    def eval_mAP(X, raw_boxes, raw_classes, orig_sizes, label):
        all_dets = {c: [] for c in range(len(CLASS_NAMES))}
        total_gt = {c: 0 for c in range(len(CLASS_NAMES))}
        for i in range(len(X)):
            orig_w, orig_h = orig_sizes[i]
            pred = model.predict(X[i:i+1], verbose=0)[0]
            detections = decode_predictions(pred, orig_w, orig_h, obj_thresh=0.05, nms_thresh=0.4)
            gt_boxes_i, gt_classes_i = raw_boxes[i], raw_classes[i]
            for c in total_gt:
                total_gt[c] += sum(1 for gc in gt_classes_i if gc == c)
            matched = match_image(detections, gt_boxes_i, gt_classes_i, iou_thresh=0.5)
            for conf, cls, is_tp in matched:
                all_dets[cls].append((conf, is_tp))

        aps = []
        for c, name in enumerate(CLASS_NAMES):
            dets = sorted(all_dets[c], key=lambda d: -d[0])
            n_gt = total_gt[c]
            tp_cum, fp_cum = 0, 0
            precisions, recalls = [], []
            for conf, is_tp in dets:
                if is_tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                precisions.append(tp_cum / (tp_cum + fp_cum))
                recalls.append(tp_cum / n_gt if n_gt else 0.0)
            ap_value = compute_ap(np.array(recalls), np.array(precisions)) if dets else 0.0
            aps.append(ap_value)
        mAP = float(np.mean(aps)) if aps else 0.0
        print(f"{label} mAP@0.5: {mAP*100:.2f}% ({len(X)} images)")
        return mAP

    print("\n=== Final evaluation ===")
    real_val_boxes = [real["raw_boxes"][i] for i in real_val_idx]
    real_val_classes = [real["raw_classes"][i] for i in real_val_idx]
    real_val_sizes = [real["orig_sizes"][i] for i in real_val_idx]
    mAP_real = eval_mAP(X_val_real, real_val_boxes, real_val_classes, real_val_sizes,
                        "Real PKLot (expanded, fresh held-out split)")

    syn_val_boxes = [syn["raw_boxes"][i] for i in syn_val_idx]
    syn_val_classes = [syn["raw_classes"][i] for i in syn_val_idx]
    syn_val_sizes = [syn["orig_sizes"][i] for i in syn_val_idx]
    mAP_syn = eval_mAP(X_val_syn, syn_val_boxes, syn_val_classes, syn_val_sizes,
                       "Synthetic (unchanged validation set)")

    report_lines = [
        f"Real datasets combined: {args.real_datasets}",
        f"Combined real pool: {n_real_total} images, sampled down to {len(real['X'])}",
        f"Train: {len(X_train)} (synthetic {len(syn_train_idx)} + real {len(real_train_idx)})",
        f"Val: synthetic {len(X_val_syn)}, real {len(X_val_real)}",
        f"Epochs: {args.epochs}, lr={args.lr}",
        f"Resumed from: {args.resume_from}",
        "",
        f"Real PKLot mAP@0.5 (expanded, fresh split, {len(X_val_real)} images): {mAP_real*100:.2f}%",
        f"Synthetic mAP@0.5 (unchanged val set, {len(X_val_syn)} images): {mAP_syn*100:.2f}%",
        "",
        "Reference -- prior checkpoint (trained_detector_multibox_mixed), different val composition:",
        "  Real PKLot mAP@0.5: 68.76% (186 images from the original 1,242-image test-split-only pool)",
        "  Synthetic mAP@0.5: 90.60%",
    ]
    report = "\n".join(report_lines)
    print("\n" + report)
    with open(os.path.join(args.out, "training_report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nTotal wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()

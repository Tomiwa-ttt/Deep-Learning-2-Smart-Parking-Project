"""
evaluate_detector.py

Evaluates the trained spot detector (models/detector_model.py /
trained_detector/spot_detector.keras) on the held-out validation split,
using real object-detection metrics: per-class precision/recall/F1 at a
practical confidence threshold, plus average precision (AP) per class and
mAP@0.5 (IoU) -- computed by matching decoded, NMS'd predictions against
the raw (non-grid-lossy) ground-truth boxes.

Usage:
    python evaluate_detector.py --dataset ./synthetic_dataset \
        --model ./trained_detector/spot_detector.keras
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from train_detector import load_dataset
from detector_utils import decode_predictions
from models.detector_model import CLASS_NAMES


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    area_a = max(ax1 - ax0, 0) * max(ay1 - ay0, 0)
    area_b = max(bx1 - bx0, 0) * max(by1 - by0, 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(detections, gt_boxes, gt_classes, iou_thresh=0.5):
    """Greedy match, highest-confidence detection first. Returns a list of
    (confidence, class_id, is_true_positive) for this one image."""
    used = [False] * len(gt_boxes)
    out = []
    for det in sorted(detections, key=lambda d: -d["confidence"]):
        best_iou, best_j = 0.0, -1
        for j, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
            if used[j] or gc != det["class_id"]:
                continue
            i = iou(det["bbox"], gb)
            if i > best_iou:
                best_iou, best_j = i, j
        is_tp = best_iou >= iou_thresh and best_j >= 0
        if is_tp:
            used[best_j] = True
        out.append((det["confidence"], det["class_id"], is_tp))
    return out


def compute_ap(recalls, precisions):
    """All-point interpolated average precision (VOC2010+/COCO style)."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./synthetic_dataset")
    ap.add_argument("--model", default="./trained_detector/spot_detector.keras")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--practical_conf", type=float, default=0.5,
                     help="confidence threshold used for the reported precision/recall/F1 operating point")
    ap.add_argument("--out", default="./trained_detector/eval_report.txt")
    args = ap.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    data = load_dataset(args.dataset)
    X, raw_boxes, raw_classes, orig_sizes = data["X"], data["raw_boxes"], data["raw_classes"], data["orig_sizes"]

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    n_val = max(int(len(X) * args.val_split), 1)
    val_idx = idx[:n_val]
    print(f"Evaluating on {len(val_idx)} held-out validation images ...")

    model = tf.keras.models.load_model(args.model, compile=False)

    all_dets = {c: [] for c in range(len(CLASS_NAMES))}   # (confidence, is_tp)
    total_gt = {c: 0 for c in range(len(CLASS_NAMES))}

    for i in val_idx:
        orig_w, orig_h = orig_sizes[i]
        pred = model.predict(X[i:i + 1], verbose=0)[0]
        # low threshold here so the full confidence range is available for AP;
        # the practical operating-point metrics are filtered out of this same list below
        detections = decode_predictions(pred, orig_w, orig_h, obj_thresh=0.05, nms_thresh=0.4)

        gt_boxes_i = raw_boxes[i]
        gt_classes_i = raw_classes[i]
        for c in total_gt:
            total_gt[c] += sum(1 for gc in gt_classes_i if gc == c)

        matched = match_image(detections, gt_boxes_i, gt_classes_i, iou_thresh=args.iou_thresh)
        for conf, cls, is_tp in matched:
            all_dets[cls].append((conf, is_tp))

    lines = []
    lines.append(f"Validation images: {len(val_idx)}")
    lines.append(f"IoU match threshold: {args.iou_thresh}")
    lines.append(f"Practical confidence operating point: {args.practical_conf}\n")

    aps = []
    for c, name in enumerate(CLASS_NAMES):
        dets = sorted(all_dets[c], key=lambda d: -d[0])
        n_gt = total_gt[c]

        tp_cum, fp_cum = 0, 0
        precisions, recalls = [], []
        tp_at_practical, fp_at_practical = 0, 0
        for conf, is_tp in dets:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt if n_gt else 0.0)
            if conf >= args.practical_conf:
                if is_tp:
                    tp_at_practical += 1
                else:
                    fp_at_practical += 1

        ap_value = compute_ap(np.array(recalls), np.array(precisions)) if dets else 0.0
        aps.append(ap_value)

        fn_at_practical = n_gt - tp_at_practical
        precision = tp_at_practical / (tp_at_practical + fp_at_practical) if (tp_at_practical + fp_at_practical) else 0.0
        recall = tp_at_practical / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        lines.append(f"[{name}]")
        lines.append(f"  Ground-truth instances: {n_gt}")
        lines.append(f"  AP@{args.iou_thresh}: {ap_value*100:.2f}%")
        lines.append(f"  At confidence>={args.practical_conf}: "
                     f"precision={precision*100:.1f}%  recall={recall*100:.1f}%  F1={f1*100:.1f}%  "
                     f"(TP={tp_at_practical} FP={fp_at_practical} FN={fn_at_practical})\n")

    mAP = float(np.mean(aps)) if aps else 0.0
    lines.append(f"mAP@{args.iou_thresh}: {mAP*100:.2f}%")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()

"""
generate_report_charts.py

Produces the visual results needed for the "Training & Evaluation" and
"Agile development" report sections:
  - training_curve.png    : train/val loss per epoch (parsed from the saved
                            training log, since the run happened in a
                            separate process)
  - pr_curves.png         : precision-recall curve per class
  - confusion_matrix.png  : predicted-vs-true class, restricted to
                            correctly-LOCALIZED boxes (IoU>=0.5 regardless
                            of class), isolating classification error from
                            localization error
  - ap_bar.png            : AP per class + mAP
  - burndown.png          : stage-level burndown reconstructed from actual
                            file history (see report for caveats)

Usage:
    python generate_report_charts.py --log <path to training stdout log> \
        --out ./report_assets/charts
"""

import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from train_detector import load_dataset
from detector_utils import decode_predictions
from models.detector_model import CLASS_NAMES
from evaluate_detector import iou, compute_ap


def parse_training_log(path):
    train_losses, val_losses = [], []
    pattern = re.compile(r"loss:\s*([\d.]+)\s*-\s*val_loss:\s*([\d.]+)")
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                train_losses.append(float(m.group(1)))
                val_losses.append(float(m.group(2)))
    return train_losses, val_losses


def plot_training_curve(train_losses, val_losses, out_path):
    plt.figure(figsize=(6, 4))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Detection loss")
    plt.title("Object detector training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def collect_matches(model, X, raw_boxes, raw_classes, orig_sizes, val_idx, iou_thresh=0.5):
    """Returns:
      per_class_dets: {class_id: [(confidence, is_tp), ...]}
      total_gt: {class_id: n}
      confusion: 2x2 array [true][pred] over IoU-matched boxes (class-agnostic match)
    """
    per_class_dets = {c: [] for c in range(len(CLASS_NAMES))}
    total_gt = {c: 0 for c in range(len(CLASS_NAMES))}
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    for i in val_idx:
        orig_w, orig_h = orig_sizes[i]
        pred = model.predict(X[i:i + 1], verbose=0)[0]
        detections = decode_predictions(pred, orig_w, orig_h, obj_thresh=0.05, nms_thresh=0.4)
        gt_boxes_i, gt_classes_i = raw_boxes[i], raw_classes[i]

        for c in total_gt:
            total_gt[c] += sum(1 for gc in gt_classes_i if gc == c)

        # class-aware matching (for PR/AP), same as evaluate_detector.match_image
        used = [False] * len(gt_boxes_i)
        for det in sorted(detections, key=lambda d: -d["confidence"]):
            best_iou, best_j = 0.0, -1
            for j, (gb, gc) in enumerate(zip(gt_boxes_i, gt_classes_i)):
                if used[j] or gc != det["class_id"]:
                    continue
                v = iou(det["bbox"], gb)
                if v > best_iou:
                    best_iou, best_j = v, j
            is_tp = best_iou >= iou_thresh and best_j >= 0
            if is_tp:
                used[best_j] = True
            per_class_dets[det["class_id"]].append((det["confidence"], is_tp))

        # class-agnostic matching (for the confusion matrix): does the box
        # localize a real object, and if so, was the class right?
        used2 = [False] * len(gt_boxes_i)
        for det in sorted(detections, key=lambda d: -d["confidence"]):
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes_i):
                if used2[j]:
                    continue
                v = iou(det["bbox"], gb)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thresh and best_j >= 0:
                used2[best_j] = True
                true_c = gt_classes_i[best_j]
                confusion[true_c, det["class_id"]] += 1

    return per_class_dets, total_gt, confusion


def plot_pr_curves(per_class_dets, total_gt, out_path):
    plt.figure(figsize=(6, 5))
    for c, name in enumerate(CLASS_NAMES):
        dets = sorted(per_class_dets[c], key=lambda d: -d[0])
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
        plt.plot(recalls, precisions, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves (IoU >= 0.5)")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ap_bar(per_class_dets, total_gt, out_path):
    aps = []
    for c, name in enumerate(CLASS_NAMES):
        dets = sorted(per_class_dets[c], key=lambda d: -d[0])
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
        aps.append(compute_ap(np.array(recalls), np.array(precisions)) if dets else 0.0)
    mAP = float(np.mean(aps))

    plt.figure(figsize=(5, 4))
    labels = CLASS_NAMES + ["mAP"]
    values = [a * 100 for a in aps] + [mAP * 100]
    colors = ["#3b5bfd", "#3b5bfd", "#18181b"]
    plt.bar(labels, values, color=colors)
    plt.ylabel("AP @ IoU 0.5 (%)")
    plt.title("Average Precision by class")
    plt.ylim(0, 105)
    for i, v in enumerate(values):
        plt.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return aps, mAP


def plot_confusion_matrix(confusion, out_path):
    plt.figure(figsize=(4.5, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=20)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion matrix\n(localized boxes only, IoU >= 0.5)")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, str(confusion[i, j]), ha="center", va="center",
                      color="white" if confusion[i, j] > confusion.max() / 2 else "black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_burndown(out_path):
    """Stage-level burndown reconstructed from the project's real file
    history (see report for how these stage boundaries were derived).
    Not a per-day Jira burndown -- an honest, coarser retrospective view."""
    stages = ["Sprint 1\n(pipeline +\nsynthetic data)", "Sprint 2\n(real-data\nintegration)",
              "Sprint 3\n(reporting)", "Sprint 4\n(env fixes +\nOD pivot)", "Sprint 5\n(real-data\nclosure)",
              "Sprint 6\n(stress-test +\nrehearsal fix)"]
    planned_total = 11
    remaining = [11, 8, 6, 2, 1, 0]
    plt.figure(figsize=(6, 4))
    plt.plot(stages, remaining, marker="o", linewidth=2, color="#3b5bfd")
    plt.fill_between(stages, remaining, alpha=0.15, color="#3b5bfd")
    plt.ylabel("Open backlog items")
    plt.title("Stage-level burndown (reconstructed)")
    plt.ylim(0, planned_total + 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to the saved detector training stdout log")
    ap.add_argument("--dataset", default="./synthetic_dataset")
    ap.add_argument("--model", default="./trained_detector/spot_detector.keras")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./report_assets/charts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Parsing training log ...")
    train_losses, val_losses = parse_training_log(args.log)
    plot_training_curve(train_losses, val_losses, os.path.join(args.out, "training_curve.png"))

    print("Loading dataset + model for evaluation charts ...")
    data = load_dataset(args.dataset)
    X, raw_boxes, raw_classes, orig_sizes = data["X"], data["raw_boxes"], data["raw_classes"], data["orig_sizes"]
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    n_val = max(int(len(X) * args.val_split), 1)
    val_idx = idx[:n_val]

    model = tf.keras.models.load_model(args.model, compile=False)
    per_class_dets, total_gt, confusion = collect_matches(model, X, raw_boxes, raw_classes, orig_sizes, val_idx)

    plot_pr_curves(per_class_dets, total_gt, os.path.join(args.out, "pr_curves.png"))
    aps, mAP = plot_ap_bar(per_class_dets, total_gt, os.path.join(args.out, "ap_bar.png"))
    plot_confusion_matrix(confusion, os.path.join(args.out, "confusion_matrix.png"))
    plot_burndown(os.path.join(args.out, "burndown.png"))

    print(f"AP per class: {dict(zip(CLASS_NAMES, aps))}")
    print(f"mAP: {mAP:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred):\n{confusion}")
    print(f"\nSaved charts to {args.out}")


if __name__ == "__main__":
    main()

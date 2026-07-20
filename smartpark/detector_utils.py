"""
detector_utils.py

Shared pieces for training and running the single-shot spot detector
(models/detector_model.py):
  - encode_targets(): ground-truth boxes -> the (S, S, 5+C) grid tensor
    the network is trained to predict
  - detection_loss(): the YOLOv1-style multi-part loss (coordinate,
    objectness, no-object, classification)
  - decode_predictions(): grid tensor -> a list of (bbox, class, confidence)
    detections in the original image's pixel coordinates, after NMS
  - draw_detections(): renders detections on an image for the demo/report
"""

import numpy as np
import cv2
import tensorflow as tf

from models.detector_model import INPUT_SIZE, GRID_SIZE, NUM_CLASSES, CLASS_NAMES

LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
LAMBDA_CLASS = 1.0

BOX_COLORS = {0: (0, 200, 0), 1: (0, 165, 255)}  # empty=green, occupied=orange


def encode_targets(boxes, class_ids, orig_w, orig_h,
                    grid_size=GRID_SIZE, input_size=INPUT_SIZE, num_classes=NUM_CLASSES):
    """boxes: list of [x0, y0, x1, y1] in the original image's pixel space.
    Returns a (grid_size, grid_size, 5+num_classes) float32 target tensor,
    and the number of boxes dropped because two centers landed in the same
    cell (only the larger box is kept per cell -- a known limitation of the
    one-box-per-cell design, see report)."""
    target = np.zeros((grid_size, grid_size, 5 + num_classes), dtype=np.float32)
    assigned_area = np.zeros((grid_size, grid_size), dtype=np.float32)
    scale_x = input_size / orig_w
    scale_y = input_size / orig_h
    cell = input_size / grid_size
    n_dropped = 0

    for (x0, y0, x1, y1), cls in zip(boxes, class_ids):
        cx = (x0 + x1) / 2 * scale_x
        cy = (y0 + y1) / 2 * scale_y
        bw = max((x1 - x0) * scale_x, 1e-3)
        bh = max((y1 - y0) * scale_y, 1e-3)
        col = int(min(max(cx // cell, 0), grid_size - 1))
        row = int(min(max(cy // cell, 0), grid_size - 1))
        area = bw * bh

        if target[row, col, 0] == 1.0:
            if assigned_area[row, col] >= area:
                n_dropped += 1
                continue
            n_dropped += 1  # the previously-assigned smaller box is now dropped instead

        tx = (cx % cell) / cell
        ty = (cy % cell) / cell
        tw = bw / input_size
        th = bh / input_size

        target[row, col, 0] = 1.0
        target[row, col, 1] = tx
        target[row, col, 2] = ty
        target[row, col, 3] = tw
        target[row, col, 4] = th
        target[row, col, 5:] = 0.0
        target[row, col, 5 + cls] = 1.0
        assigned_area[row, col] = area

    return target, n_dropped


def make_detection_loss(lambda_coord=LAMBDA_COORD, lambda_noobj=LAMBDA_NOOBJ, lambda_class=LAMBDA_CLASS):
    """Factory so hyperparameter-tuning runs can compare loss weightings
    (see hyperparameter_ablation.py) without touching the default used for
    the main trained model."""
    def loss_fn(y_true, y_pred):
        obj_mask = y_true[..., 0:1]
        noobj_mask = 1.0 - obj_mask

        obj_loss = tf.reduce_sum(obj_mask * tf.square(y_true[..., 0:1] - y_pred[..., 0:1]), axis=[1, 2, 3])
        noobj_loss = tf.reduce_sum(noobj_mask * tf.square(y_true[..., 0:1] - y_pred[..., 0:1]), axis=[1, 2, 3])

        txy_loss = tf.reduce_sum(
            obj_mask * tf.square(y_true[..., 1:3] - y_pred[..., 1:3]), axis=[1, 2, 3]
        )
        true_wh_sqrt = tf.sqrt(tf.maximum(y_true[..., 3:5], 1e-6))
        pred_wh_sqrt = tf.sqrt(tf.maximum(y_pred[..., 3:5], 1e-6))
        twh_loss = tf.reduce_sum(obj_mask * tf.square(true_wh_sqrt - pred_wh_sqrt), axis=[1, 2, 3])
        coord_loss = lambda_coord * (txy_loss + twh_loss)

        class_loss = lambda_class * tf.reduce_sum(
            obj_mask * tf.square(y_true[..., 5:] - y_pred[..., 5:]), axis=[1, 2, 3]
        )

        per_sample = coord_loss + obj_loss + lambda_noobj * noobj_loss + class_loss
        return tf.reduce_mean(per_sample)
    return loss_fn


detection_loss = make_detection_loss()


def decode_predictions(pred, orig_w, orig_h, grid_size=GRID_SIZE, input_size=INPUT_SIZE,
                        obj_thresh=0.5, nms_thresh=0.4):
    """pred: (grid_size, grid_size, 5+num_classes) numpy array (single image,
    already through the model's sigmoid/softmax head). Returns a list of
    dicts with bbox in the ORIGINAL image's pixel coordinates."""
    cell = input_size / grid_size
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    boxes_by_class = {c: [] for c in range(len(CLASS_NAMES))}
    scores_by_class = {c: [] for c in range(len(CLASS_NAMES))}

    rows, cols = np.where(pred[..., 0] >= obj_thresh)
    for row, col in zip(rows, cols):
        obj = pred[row, col, 0]
        tx, ty, tw, th = pred[row, col, 1:5]
        cx = (col + tx) * cell
        cy = (row + ty) * cell
        bw = tw * input_size
        bh = th * input_size
        x0, y0 = cx - bw / 2, cy - bh / 2

        class_probs = pred[row, col, 5:]
        cls = int(np.argmax(class_probs))
        conf = float(obj * class_probs[cls])

        boxes_by_class[cls].append([float(x0), float(y0), float(bw), float(bh)])
        scores_by_class[cls].append(conf)

    detections = []
    for cls, boxes in boxes_by_class.items():
        scores = scores_by_class[cls]
        if not boxes:
            continue
        keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=obj_thresh, nms_threshold=nms_thresh)
        keep = np.array(keep).flatten()
        for i in keep:
            x0, y0, bw, bh = boxes[i]
            x0o, y0o = x0 * scale_x, y0 * scale_y
            x1o, y1o = (x0 + bw) * scale_x, (y0 + bh) * scale_y
            detections.append({
                "bbox": [int(round(x0o)), int(round(y0o)), int(round(x1o)), int(round(y1o))],
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "confidence": scores[i],
            })
    return detections


def draw_detections(img, detections):
    out = img.copy()
    for d in detections:
        x0, y0, x1, y1 = d["bbox"]
        color = BOX_COLORS.get(d["class_id"], (255, 255, 255))
        label = f"{d['class_name']} {d['confidence']*100:.0f}%"
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, label, (x0 + 2, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    return out


def preprocess_image(img, input_size=INPUT_SIZE):
    """Resize a BGR image to the model's fixed square input size."""
    return cv2.resize(img, (input_size, input_size))

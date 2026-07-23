"""
detector_utils.py

Shared pieces for training and running the single-shot spot detector
(models/detector_model.py):
  - encode_targets(): ground-truth boxes -> the (S, S, B*(5+C)) grid tensor
    the network is trained to predict, B boxes per cell
  - detection_loss(): the YOLOv1-style multi-part loss (coordinate,
    objectness, no-object, classification), summed over box slots
  - decode_predictions(): grid tensor -> a list of (bbox, class, confidence)
    detections in the original image's pixel coordinates, after NMS
  - draw_detections(): renders detections on an image for the demo/report

Each cell predicts NUM_BOXES_PER_CELL candidate boxes rather than one. A
one-box-per-cell version dropped 28.2% of real PKLot ground-truth boxes to
grid-cell collisions (real lots are much denser than anywhere in training);
measuring collision rate directly showed 3 boxes/cell at the same 16x16
grid eliminates this (28.25% -> 0.00%) -- see models/detector_model.py and
the report's Hyperparameter Tuning section.
"""

import numpy as np
import cv2
import tensorflow as tf

from models.detector_model import INPUT_SIZE, GRID_SIZE, NUM_CLASSES, NUM_BOXES_PER_CELL, CLASS_NAMES

LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
LAMBDA_CLASS = 1.0

BOX_COLORS = {0: (0, 200, 0), 1: (0, 165, 255)}  # empty=green, occupied=orange


def encode_targets(boxes, class_ids, orig_w, orig_h,
                    grid_size=GRID_SIZE, input_size=INPUT_SIZE, num_classes=NUM_CLASSES,
                    boxes_per_cell=NUM_BOXES_PER_CELL):
    """boxes: list of [x0, y0, x1, y1] in the original image's pixel space.
    Returns a flat (grid_size, grid_size, boxes_per_cell*(5+num_classes))
    float32 target tensor (matching the model's output shape) and the
    number of boxes dropped because more than boxes_per_cell centers landed
    in the same cell (only the boxes_per_cell largest are kept per cell)."""
    box_size = 5 + num_classes
    target = np.zeros((grid_size, grid_size, boxes_per_cell, box_size), dtype=np.float32)
    assigned_area = np.zeros((grid_size, grid_size, boxes_per_cell), dtype=np.float32)
    occupied = np.zeros((grid_size, grid_size, boxes_per_cell), dtype=bool)
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

        slot = None
        for b in range(boxes_per_cell):
            if not occupied[row, col, b]:
                slot = b
                break
        if slot is None:
            # every slot taken -- replace the smallest assigned box if this
            # one is bigger, else drop this one instead
            areas = assigned_area[row, col, :]
            min_b = int(np.argmin(areas))
            n_dropped += 1
            if area <= areas[min_b]:
                continue
            slot = min_b

        tx = (cx % cell) / cell
        ty = (cy % cell) / cell
        tw = bw / input_size
        th = bh / input_size

        target[row, col, slot, 0] = 1.0
        target[row, col, slot, 1] = tx
        target[row, col, slot, 2] = ty
        target[row, col, slot, 3] = tw
        target[row, col, slot, 4] = th
        target[row, col, slot, 5:] = 0.0
        target[row, col, slot, 5 + cls] = 1.0
        occupied[row, col, slot] = True
        assigned_area[row, col, slot] = area

    return target.reshape(grid_size, grid_size, boxes_per_cell * box_size), n_dropped


def make_detection_loss(lambda_coord=LAMBDA_COORD, lambda_noobj=LAMBDA_NOOBJ, lambda_class=LAMBDA_CLASS,
                         boxes_per_cell=NUM_BOXES_PER_CELL, num_classes=NUM_CLASSES):
    """Factory so hyperparameter-tuning runs can compare loss weightings
    (see hyperparameter_ablation.py) without touching the default used for
    the main trained model. y_true/y_pred are the flat (batch, S, S,
    boxes_per_cell*(5+C)) tensors; reshaped internally to (..., boxes_per_cell,
    5+C) so each box slot is scored independently, then summed."""
    box_size = 5 + num_classes

    def loss_fn(y_true, y_pred):
        shape = tf.shape(y_true)
        box_shape = tf.concat([shape[:-1], [boxes_per_cell, box_size]], axis=0)
        yt = tf.reshape(y_true, box_shape)
        yp = tf.reshape(y_pred, box_shape)

        obj_mask = yt[..., 0:1]
        noobj_mask = 1.0 - obj_mask

        obj_loss = tf.reduce_sum(obj_mask * tf.square(yt[..., 0:1] - yp[..., 0:1]), axis=[1, 2, 3, 4])
        noobj_loss = tf.reduce_sum(noobj_mask * tf.square(yt[..., 0:1] - yp[..., 0:1]), axis=[1, 2, 3, 4])

        txy_loss = tf.reduce_sum(
            obj_mask * tf.square(yt[..., 1:3] - yp[..., 1:3]), axis=[1, 2, 3, 4]
        )
        true_wh_sqrt = tf.sqrt(tf.maximum(yt[..., 3:5], 1e-6))
        pred_wh_sqrt = tf.sqrt(tf.maximum(yp[..., 3:5], 1e-6))
        twh_loss = tf.reduce_sum(obj_mask * tf.square(true_wh_sqrt - pred_wh_sqrt), axis=[1, 2, 3, 4])
        coord_loss = lambda_coord * (txy_loss + twh_loss)

        class_loss = lambda_class * tf.reduce_sum(
            obj_mask * tf.square(yt[..., 5:] - yp[..., 5:]), axis=[1, 2, 3, 4]
        )

        per_sample = coord_loss + obj_loss + lambda_noobj * noobj_loss + class_loss
        return tf.reduce_mean(per_sample)
    return loss_fn


detection_loss = make_detection_loss()


def decode_predictions(pred, orig_w, orig_h, grid_size=GRID_SIZE, input_size=INPUT_SIZE,
                        boxes_per_cell=NUM_BOXES_PER_CELL, num_classes=NUM_CLASSES,
                        obj_thresh=0.5, nms_thresh=0.4):
    """pred: flat (grid_size, grid_size, boxes_per_cell*(5+num_classes))
    numpy array (single image, already through the model's sigmoid/softmax
    head). Returns a list of dicts with bbox in the ORIGINAL image's pixel
    coordinates, after per-class NMS across every (cell, box slot)."""
    box_size = 5 + num_classes
    pred = pred.reshape(grid_size, grid_size, boxes_per_cell, box_size)
    cell = input_size / grid_size
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    boxes_by_class = {c: [] for c in range(len(CLASS_NAMES))}
    scores_by_class = {c: [] for c in range(len(CLASS_NAMES))}

    rows, cols, slots = np.where(pred[..., 0] >= obj_thresh)
    for row, col, slot in zip(rows, cols, slots):
        obj = pred[row, col, slot, 0]
        tx, ty, tw, th = pred[row, col, slot, 1:5]
        cx = (col + tx) * cell
        cy = (row + ty) * cell
        bw = tw * input_size
        bh = th * input_size
        x0, y0 = cx - bw / 2, cy - bh / 2

        class_probs = pred[row, col, slot, 5:]
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

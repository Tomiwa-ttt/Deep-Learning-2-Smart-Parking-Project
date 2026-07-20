"""
make_video_assets.py

Runs the real-trained CNN on several actual PKLot photos and produces:
  - the raw (unannotated) real photo
  - an annotated version with colored boxes drawn directly on the real photo
    (green = empty, orange = occupied) based on genuine model predictions
  - per-image stats (counts, accuracy vs ground truth) for video captions

Output goes to demo/video_assets/
"""
import json
import os
import random
import cv2
import numpy as np
import tensorflow as tf

IMAGES_DIR = "/home/claude/real_data/test_extracted/test"
ANNOTATIONS = os.path.join(IMAGES_DIR, "_annotations.coco.json")
MODEL_PATH = "./trained_model_real/parking_spot_cnn.keras"
OUT_DIR = "demo/video_assets"
IMG_SIZE = 64

random.seed(11)
os.makedirs(OUT_DIR, exist_ok=True)

with open(ANNOTATIONS) as f:
    coco = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
image_id_to_info = {img["id"]: img for img in coco["images"]}

anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

candidate_ids = [iid for iid, anns in anns_by_image.items() if 25 <= len(anns) <= 45]
chosen_ids = random.sample(candidate_ids, min(3, len(candidate_ids)))

model = tf.keras.models.load_model(MODEL_PATH)

manifest = []
for idx, iid in enumerate(chosen_ids):
    info = image_id_to_info[iid]
    img_path = os.path.join(IMAGES_DIR, info["file_name"])
    img = cv2.imread(img_path)
    anns = sorted(anns_by_image[iid], key=lambda a: a["id"])

    crops, boxes, true_labels = [], [], []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        crop = cv2.resize(img[max(y0, 0):y1, max(x0, 0):x1], (IMG_SIZE, IMG_SIZE))
        crops.append(crop)
        boxes.append([x0, y0, x1, y1])
        true_labels.append(cat_id_to_name[ann["category_id"]])

    batch = np.stack(crops).astype(np.float32)
    preds = model.predict(batch, verbose=0).flatten()

    raw_path = os.path.join(OUT_DIR, f"lot{idx}_raw.jpg")
    cv2.imwrite(raw_path, img)

    annotated = img.copy()
    n_empty = n_occ = n_correct = 0
    for box, p, true_label in zip(boxes, preds, true_labels):
        occupied = bool(p >= 0.5)
        conf = float(p if occupied else 1 - p)
        x0, y0, x1, y1 = box
        color = (0, 200, 0) if not occupied else (0, 165, 255)  # BGR: green / orange
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)

        if occupied:
            n_occ += 1
        else:
            n_empty += 1
        pred_label = "space-occupied" if occupied else "space-empty"
        if pred_label == true_label:
            n_correct += 1

    ann_path = os.path.join(OUT_DIR, f"lot{idx}_annotated.jpg")
    cv2.imwrite(ann_path, annotated)

    stats = {
        "lot_name": os.path.splitext(info["file_name"])[0].split("_jpg")[0],
        "total": len(boxes),
        "empty": n_empty,
        "occupied": n_occ,
        "correct": n_correct,
        "raw_path": raw_path,
        "annotated_path": ann_path,
    }
    manifest.append(stats)
    print(stats)

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

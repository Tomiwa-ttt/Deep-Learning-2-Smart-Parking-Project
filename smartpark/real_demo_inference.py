"""
real_demo_inference.py

Runs the final CNN (trained on real PKLot data) on a handful of real lot
images, using the dataset's own annotated space boxes (not re-detected).
Produces the same result format the demo app expects, but grounded in
real photos and real ground truth -- for an honest, real-data demo.

Note: this does NOT run the geometric "improperly parked" check -- that
heuristic was tuned against clean synthetic asphalt/car contrast and needs
recalibration (or a real detector) before it's trustworthy on real photos
with shadows/texture. See README "Known limitations" section.
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
IMG_SIZE = 64

random.seed(7)

with open(ANNOTATIONS) as f:
    coco = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
image_id_to_info = {img["id"]: img for img in coco["images"]}

anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

# pick a few images with a reasonable number of spaces (not too few, not huge)
candidate_ids = [iid for iid, anns in anns_by_image.items() if 25 <= len(anns) <= 45]
chosen_ids = random.sample(candidate_ids, min(4, len(candidate_ids)))

model = tf.keras.models.load_model(MODEL_PATH)

output = {}
for iid in chosen_ids:
    info = image_id_to_info[iid]
    img_path = os.path.join(IMAGES_DIR, info["file_name"])
    img = cv2.imread(img_path)
    anns = sorted(anns_by_image[iid], key=lambda a: a["id"])

    crops = []
    boxes = []
    true_labels = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        crop = cv2.resize(img[max(y0, 0):y1, max(x0, 0):x1], (IMG_SIZE, IMG_SIZE))
        crops.append(crop)
        boxes.append([x0, y0, x1, y1])
        true_labels.append(cat_id_to_name[ann["category_id"]])

    batch = np.stack(crops).astype(np.float32)
    preds = model.predict(batch, verbose=0).flatten()

    results = []
    for i, (box, p, true_label) in enumerate(zip(boxes, preds, true_labels)):
        occupied = bool(p >= 0.5)
        conf = float(p if occupied else 1 - p)
        results.append({
            "spot_id": i,
            "bbox": box,
            "occupied": occupied,
            "confidence": conf,
            "properly_parked": True if occupied else None,  # not evaluated on real data yet, see README
            "reason": "Occupied" if occupied else "Empty",
            "ground_truth": true_label,
        })

    lot_key = os.path.splitext(info["file_name"])[0][:20]
    output[lot_key] = results

    n_correct = sum(1 for r in results if (r["occupied"] and r["ground_truth"] == "space-occupied")
                    or (not r["occupied"] and r["ground_truth"] == "space-empty"))
    print(f"{lot_key}: {n_correct}/{len(results)} correct")

with open("demo/real_sample_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved demo/real_sample_results.json")

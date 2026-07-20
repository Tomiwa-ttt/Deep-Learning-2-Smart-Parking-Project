import json
import os
import random
import cv2
import numpy as np
import sys
sys.path.append(".")
from real_car_detection import find_car_bbox_real
from improper_parking import evaluate_spot

IMAGES_DIR = "/home/claude/real_data/test_extracted/test"
ANNOTATIONS = os.path.join(IMAGES_DIR, "_annotations.coco.json")

with open(ANNOTATIONS) as f:
    coco = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
image_id_to_info = {img["id"]: img for img in coco["images"]}
anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

random.seed(3)
candidate_ids = [
    iid for iid, anns in anns_by_image.items()
    if sum(1 for a in anns if cat_id_to_name[a["category_id"]] == "space-occupied") >= 15
]
test_id = random.choice(candidate_ids)

info = image_id_to_info[test_id]
img = cv2.imread(os.path.join(IMAGES_DIR, info["file_name"]))
anns = anns_by_image[test_id]

vis = img.copy()
n_proper = n_improper = n_no_car_found = 0

for ann in anns:
    if cat_id_to_name[ann["category_id"]] != "space-occupied":
        continue
    x, y, w, h = ann["bbox"]
    spot_box = [int(x), int(y), int(x + w), int(y + h)]

    car_box = find_car_bbox_real(img, spot_box)

    cv2.rectangle(vis, (spot_box[0], spot_box[1]), (spot_box[2], spot_box[3]), (255, 255, 255), 1)

    if car_box is None:
        n_no_car_found += 1
        continue

    verdict = evaluate_spot(ann["id"], spot_box, car_box, occupied=True,
                             iou_threshold=0.35, overflow_threshold=0.45)
    color = (0, 165, 255) if verdict.properly_parked else (0, 0, 255)
    cv2.rectangle(vis, (car_box[0], car_box[1]), (car_box[2], car_box[3]), color, 2)
    if verdict.properly_parked:
        n_proper += 1
    else:
        n_improper += 1

out_path = "demo/video_assets/real_detector_test.jpg"
cv2.imwrite(out_path, vis)
print(f"Proper: {n_proper}  Improper: {n_improper}  No car found: {n_no_car_found}")
print(f"Saved: {out_path}")

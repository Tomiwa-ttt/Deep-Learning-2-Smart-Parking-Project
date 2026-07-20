"""
validate.py

Runs the full inference pipeline (CNN occupancy + geometry improper-parking
check) against every lot in the synthetic dataset and compares to ground
truth, so we have a real accuracy number instead of eyeballing one image.
"""
import json
import os
import sys
import cv2
import tensorflow as tf

sys.path.append(os.path.dirname(__file__))
from inference import run_inference, load_spots_for_lot

MODEL_PATH = "./trained_model/parking_spot_cnn.keras"
MANIFEST = "./synthetic_dataset/manifest.json"
LOTS_DIR = "./synthetic_dataset/full_lots"

model = tf.keras.models.load_model(MODEL_PATH)
with open(MANIFEST) as f:
    manifest = json.load(f)

gt_by_lot_and_id = {}
for m in manifest:
    gt_by_lot_and_id.setdefault(m["lot_path"], {})[m["id"]] = m

occ_correct = occ_total = 0
park_correct = park_total = 0

for lot_path in sorted(set(m["lot_path"] for m in manifest)):
    img = cv2.imread(lot_path)
    spots = load_spots_for_lot(MANIFEST, lot_path)
    results = run_inference(img, spots, model)
    gt = gt_by_lot_and_id[lot_path]

    for r in results:
        truth = gt[r["spot_id"]]
        occ_total += 1
        if r["occupied"] == truth["occupied"]:
            occ_correct += 1

        if truth["occupied"]:  # only score parking-quality on truly occupied spots
            park_total += 1
            true_proper = not truth["improperly_parked"]
            pred_proper = r["properly_parked"]
            if pred_proper == true_proper:
                park_correct += 1

print(f"Occupancy accuracy:        {occ_correct}/{occ_total} = {100*occ_correct/occ_total:.1f}%")
print(f"Proper-parking accuracy:   {park_correct}/{park_total} = {100*park_correct/park_total:.1f}%")

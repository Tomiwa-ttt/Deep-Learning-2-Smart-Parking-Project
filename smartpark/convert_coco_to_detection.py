"""
convert_coco_to_detection.py

Converts a Roboflow COCO-format PKLot export (full images + per-space
bounding-box annotations) into the same dataset layout
data/generate_synthetic_data.py produces for the object detector:

    <out>/full_lots/*.jpg
    <out>/annotations.json   (images / annotations / categories,
                               categories = [empty_spot, occupied_spot]
                               matching models.detector_model.CLASS_NAMES)

This is the detection-side counterpart to convert_coco_to_crops.py (which
produces per-spot crops for the classifier instead). Both scripts read the
same source export.

Usage:
    python convert_coco_to_detection.py --images_dir ./pklot_raw/test \
        --annotations ./pklot_raw/test/_annotations.coco.json \
        --out ./real_pklot_dataset
"""

import argparse
import json
import os
import shutil

from models.detector_model import CLASS_NAMES

# Roboflow's PKLot export category names -> our fixed class order
SOURCE_CATEGORY_TO_CLASS_ID = {
    "space-empty": CLASS_NAMES.index("empty_spot"),
    "space-occupied": CLASS_NAMES.index("occupied_spot"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.annotations) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    out_lots_dir = os.path.join(args.out, "full_lots")
    os.makedirs(out_lots_dir, exist_ok=True)

    out_images = []
    out_annotations = []
    n_skipped = 0

    for img_info in coco["images"]:
        src_path = os.path.join(args.images_dir, img_info["file_name"])
        if not os.path.exists(src_path):
            continue
        shutil.copy(src_path, os.path.join(out_lots_dir, img_info["file_name"]))
        out_images.append({
            "id": img_info["id"],
            "file_name": img_info["file_name"],
            "width": img_info["width"],
            "height": img_info["height"],
        })

    ann_id = 0
    for ann in coco["annotations"]:
        cat_name = cat_id_to_name.get(ann["category_id"])
        class_id = SOURCE_CATEGORY_TO_CLASS_ID.get(cat_name)
        if class_id is None:
            n_skipped += 1
            continue
        out_annotations.append({
            "id": ann_id,
            "image_id": ann["image_id"],
            "category_id": class_id,
            "bbox": ann["bbox"],
            "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
            "iscrowd": 0,
        })
        ann_id += 1

    categories = [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]
    with open(os.path.join(args.out, "annotations.json"), "w") as f:
        json.dump({"images": out_images, "annotations": out_annotations, "categories": categories}, f, indent=2)

    print(f"Converted {len(out_images)} images, {len(out_annotations)} annotations "
          f"({n_skipped} skipped, non-space-empty/occupied categories) -> {args.out}")


if __name__ == "__main__":
    main()

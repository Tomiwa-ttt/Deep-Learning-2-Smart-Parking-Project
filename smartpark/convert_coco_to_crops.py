"""
convert_coco_to_crops.py

Converts a Roboflow COCO-format PKLot export (full images + per-space
bounding box annotations) into the empty/occupied cropped-patch folder
structure that train.py expects.

Usage:
    python convert_coco_to_crops.py --images_dir ./test_extracted/test \
        --annotations ./test_extracted/test/_annotations.coco.json \
        --out ./real_dataset
"""
import argparse
import json
import os
import cv2

CATEGORY_MAP = {
    "space-empty": "empty",
    "space-occupied": "occupied",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=int, default=2, help="pixels of context padding around each crop")
    args = ap.parse_args()

    with open(args.annotations) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    image_id_to_info = {img["id"]: img for img in coco["images"]}

    out_empty = os.path.join(args.out, "spot_crops", "empty")
    out_occupied = os.path.join(args.out, "spot_crops", "occupied")
    os.makedirs(out_empty, exist_ok=True)
    os.makedirs(out_occupied, exist_ok=True)

    # cache loaded images since many annotations share the same image
    image_cache = {}
    n_written = {"empty": 0, "occupied": 0}
    n_skipped = 0

    for ann in coco["annotations"]:
        cat_name = cat_id_to_name.get(ann["category_id"])
        label = CATEGORY_MAP.get(cat_name)
        if label is None:
            continue  # skip the generic "spaces" supercategory annotations, if any

        img_info = image_id_to_info[ann["image_id"]]
        file_name = img_info["file_name"]
        img_path = os.path.join(args.images_dir, file_name)

        if img_path not in image_cache:
            img = cv2.imread(img_path)
            image_cache[img_path] = img
        img = image_cache[img_path]
        if img is None:
            n_skipped += 1
            continue

        H, W = img.shape[:2]
        x, y, w, h = ann["bbox"]
        x0 = max(int(x) - args.pad, 0)
        y0 = max(int(y) - args.pad, 0)
        x1 = min(int(x + w) + args.pad, W)
        y1 = min(int(y + h) + args.pad, H)
        if x1 <= x0 or y1 <= y0:
            n_skipped += 1
            continue

        crop = img[y0:y1, x0:x1]
        crop = cv2.resize(crop, (64, 64))

        out_dir = out_occupied if label == "occupied" else out_empty
        out_path = os.path.join(out_dir, f"{ann['id']:07d}.jpg")
        cv2.imwrite(out_path, crop)
        n_written[label] += 1

    print(f"Wrote {n_written['empty']} empty crops, {n_written['occupied']} occupied crops")
    print(f"Skipped {n_skipped} annotations (missing image or invalid box)")
    print(f"Saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

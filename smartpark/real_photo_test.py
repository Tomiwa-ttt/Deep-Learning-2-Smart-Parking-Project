"""
real_photo_test.py

A qualitative real-world generalization test: runs the trained detector
(trained only on synthetic data) on real, full-scene photos from
CNRPark-EXT -- a genuinely different visual domain (angled real camera,
no painted spot boundary lines, natural lighting/shadows/clutter).

This is NOT a quantitative mAP evaluation: CNRPark-EXT's public release
ships per-spot labels for pre-cropped patches (see splits.zip) but not
bounding-box ground truth for the full-scene images, so there is no
ground truth here to score against. The point is to honestly observe
what the detector does out of its training distribution.

Real images are pulled from a public source at runtime (not committed to
the repo) -- see download_real_photos() for the source and how the sample
was selected.

Usage:
    python real_photo_test.py --n 6 --out ./real_photo_test_output
"""

import argparse
import glob
import os
import random
import tarfile
import urllib.request

import cv2
import tensorflow as tf

from detector_utils import decode_predictions, draw_detections, preprocess_image

SOURCE_URL = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT_FULL_IMAGE_1000x750.tar"
# Only a partial download: the tar is ~450MB, but a small byte-range prefix
# already contains dozens of complete, valid JPEGs (tar entries are
# sequential, uncompressed) -- more than enough for a qualitative look.
PARTIAL_BYTES = 30_000_000


def download_real_photos(dest_dir, n_bytes=PARTIAL_BYTES):
    os.makedirs(dest_dir, exist_ok=True)
    partial_path = os.path.join(dest_dir, "_partial.tar")
    if not os.path.exists(partial_path):
        req = urllib.request.Request(SOURCE_URL, headers={"Range": f"bytes=0-{n_bytes}"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(partial_path, "wb") as f:
            f.write(resp.read())

    n = 0
    try:
        tf_ = tarfile.open(partial_path, "r|")
        for member in tf_:
            if member.isfile() and member.name.endswith(".jpg"):
                try:
                    data = tf_.extractfile(member).read()
                except Exception:
                    break
                flat_name = "_".join(member.name.split("/")[1:])
                with open(os.path.join(dest_dir, flat_name), "wb") as out_f:
                    out_f.write(data)
                n += 1
    except Exception:
        pass  # expected: the partial tar ends mid-entry
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./trained_detector/spot_detector.keras")
    ap.add_argument("--images_dir", default="./real_cnrpark_sample")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="./real_cnrpark_sample/detector_output")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")))
    if not files:
        print(f"No images found in {args.images_dir}; downloading a real sample from CNRPark-EXT ...")
        n = download_real_photos(args.images_dir)
        print(f"Saved {n} real full-scene photos to {args.images_dir}")
        files = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")))

    os.makedirs(args.out, exist_ok=True)
    model = tf.keras.models.load_model(args.model, compile=False)

    random.seed(args.seed)
    sample = random.sample(files, min(args.n, len(files)))

    for f in sample:
        img = cv2.imread(f)
        orig_h, orig_w = img.shape[:2]
        resized = preprocess_image(img)
        pred = model.predict(resized[None, ...], verbose=0)[0]
        detections = decode_predictions(pred, orig_w, orig_h, obj_thresh=0.5, nms_thresh=0.4)
        annotated = draw_detections(img, detections)
        cv2.imwrite(os.path.join(args.out, os.path.basename(f)), annotated)

        n_empty = sum(1 for d in detections if d["class_name"] == "empty_spot")
        n_occ = sum(1 for d in detections if d["class_name"] == "occupied_spot")
        print(f"{os.path.basename(f)}: {len(detections)} detections "
              f"({n_empty} empty_spot, {n_occ} occupied_spot)")

    print(f"\nAnnotated outputs saved to {args.out}")
    print("Reminder: no ground truth is available for these images -- inspect visually, "
          "don't read the counts above as accuracy.")


if __name__ == "__main__":
    main()

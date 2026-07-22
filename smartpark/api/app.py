"""
api/app.py

Flask REST API for the SmartPark system. Two complementary pipelines are
served side by side:

  1. A general object DETECTOR (models/detector_model.py) that finds and
     classifies spots -- "empty_spot" / "occupied_spot" -- directly in any
     uncalibrated lot photo. This is what powers photo uploads.
  2. The original per-spot occupancy CLASSIFIER + geometric misparking
     check, for cameras with pre-calibrated spot boundaries (the sample
     lots below), which additionally flags improperly parked cars.

Run:
    cd smartpark
    python api/app.py
    then open http://localhost:5050 in a browser

Endpoints:
    GET  /                                -> demo UI
    GET  /api/lots                        -> list available sample lots
    GET  /api/lots/<lot_id>/image         -> raw image for a sample lot
    GET  /api/lots/<lot_id>/status        -> calibrated classifier status for every spot
    POST /api/lots/<lot_id>/refresh       -> re-run the classifier on the lot's latest image
    POST /api/analyze                     -> upload any photo, get object-detected occupancy back
"""

import base64
import os
import sys
import glob
from flask import Flask, jsonify, request, send_file, send_from_directory

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import tensorflow as tf
from inference import run_inference, load_spots_for_lot, draw_annotated
from detector_utils import decode_predictions, draw_detections, preprocess_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model", "parking_spot_cnn.keras")
DETECTOR_PATH = os.path.join(BASE_DIR, "trained_detector_mixed", "spot_detector.keras")
MANIFEST_PATH = os.path.join(BASE_DIR, "synthetic_dataset", "manifest.json")
LOTS_DIR = os.path.join(BASE_DIR, "synthetic_dataset", "full_lots")
DEMO_DIR = os.path.join(BASE_DIR, "demo")

app = Flask(__name__)

print("Loading occupancy classifier...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Loading spot detector...")
detector = tf.keras.models.load_model(DETECTOR_PATH, compile=False)
print("Models loaded.")

# in-memory cache of last computed status per lot, so /status is instant
# and /refresh is the only endpoint that actually re-runs inference
_status_cache = {}


def _lot_ids():
    files = sorted(glob.glob(os.path.join(LOTS_DIR, "*.jpg")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _lot_image_path(lot_id):
    return os.path.join(LOTS_DIR, f"{lot_id}.jpg")


def _encode_jpeg_b64(img):
    ok, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _analyze(img, spots):
    """Runs the shared spot-classification pipeline and packages the result
    (including an annotated preview image) the same way regardless of
    whether the spots came from a known lot's manifest or a single
    whole-image bounding box for an arbitrary uploaded photo."""
    results = run_inference(img, spots, model)
    annotated = draw_annotated(img, results)
    n_empty = sum(1 for r in results if not r["occupied"])
    n_bad = sum(1 for r in results if r["occupied"] and not r["properly_parked"])
    return {
        "total_spots": len(results),
        "empty_spots": n_empty,
        "occupied_spots": len(results) - n_empty,
        "improperly_parked": n_bad,
        "spots": results,
        "annotated_image": _encode_jpeg_b64(annotated),
    }


def _compute_status(lot_id):
    img_path = _lot_image_path(lot_id)
    img = cv2.imread(img_path)
    if img is None:
        return None
    spots = load_spots_for_lot(MANIFEST_PATH, img_path)
    status = _analyze(img, spots)
    status["lot_id"] = lot_id
    _status_cache[lot_id] = status
    return status


def _detect(img):
    """Runs the trained object detector on a full, uncalibrated image and
    packages the result the same way _analyze() does for the classifier
    pipeline, so the frontend can render either shape."""
    orig_h, orig_w = img.shape[:2]
    resized = preprocess_image(img)
    pred = detector.predict(resized[None, ...], verbose=0)[0]
    detections = decode_predictions(pred, orig_w, orig_h)
    annotated = draw_detections(img, detections)
    n_empty = sum(1 for d in detections if d["class_name"] == "empty_spot")
    n_occupied = sum(1 for d in detections if d["class_name"] == "occupied_spot")
    return {
        "total_spots": len(detections),
        "empty_spots": n_empty,
        "occupied_spots": n_occupied,
        "detections": detections,
        "annotated_image": _encode_jpeg_b64(annotated),
    }


@app.route("/")
def index():
    return send_from_directory(DEMO_DIR, "park_check.html")


@app.route("/api/lots", methods=["GET"])
def list_lots():
    return jsonify({"lots": _lot_ids()})


@app.route("/api/lots/<lot_id>/image", methods=["GET"])
def lot_image(lot_id):
    path = _lot_image_path(lot_id)
    if not os.path.exists(path):
        return jsonify({"error": "lot not found"}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/lots/<lot_id>/status", methods=["GET"])
def get_status(lot_id):
    if lot_id not in _status_cache:
        status = _compute_status(lot_id)
        if status is None:
            return jsonify({"error": "lot not found"}), 404
        return jsonify(status)
    return jsonify(_status_cache[lot_id])


@app.route("/api/lots/<lot_id>/refresh", methods=["POST"])
def refresh_status(lot_id):
    status = _compute_status(lot_id)
    if status is None:
        return jsonify({"error": "lot not found"}), 404
    return jsonify(status)


@app.route("/api/analyze", methods=["POST"])
def analyze_upload():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image uploaded"}), 400
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read that image file"}), 400

    result = _detect(img)
    result["mode"] = "detector"
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)

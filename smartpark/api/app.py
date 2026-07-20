"""
api/app.py

Flask REST API that serves live parking spot status. This is what a mobile
app would poll (or connect to via WebSocket in a fancier version) to show
users real-time availability.

Run:
    cd smartpark
    python api/app.py

Endpoints:
    GET  /api/lots                       -> list available lots
    GET  /api/lots/<lot_id>/status       -> current status of every spot
    POST /api/lots/<lot_id>/refresh      -> re-run inference on the lot's latest image
"""

import os
import sys
import json
import glob
from flask import Flask, jsonify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import tensorflow as tf
from inference import run_inference, load_spots_for_lot

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model", "parking_spot_cnn.keras")
MANIFEST_PATH = os.path.join(BASE_DIR, "synthetic_dataset", "manifest.json")
LOTS_DIR = os.path.join(BASE_DIR, "synthetic_dataset", "full_lots")

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# in-memory cache of last computed status per lot, so /status is instant
# and /refresh is the only endpoint that actually re-runs inference
_status_cache = {}


def _lot_ids():
    files = sorted(glob.glob(os.path.join(LOTS_DIR, "*.jpg")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _lot_image_path(lot_id):
    return os.path.join(LOTS_DIR, f"{lot_id}.jpg")


def _compute_status(lot_id):
    img_path = _lot_image_path(lot_id)
    img = cv2.imread(img_path)
    if img is None:
        return None
    spots = load_spots_for_lot(MANIFEST_PATH, img_path)
    results = run_inference(img, spots, model)
    n_empty = sum(1 for r in results if not r["occupied"])
    n_bad = sum(1 for r in results if r["occupied"] and not r["properly_parked"])
    status = {
        "lot_id": lot_id,
        "total_spots": len(results),
        "empty_spots": n_empty,
        "occupied_spots": len(results) - n_empty,
        "improperly_parked": n_bad,
        "spots": results,
    }
    _status_cache[lot_id] = status
    return status


@app.route("/api/lots", methods=["GET"])
def list_lots():
    return jsonify({"lots": _lot_ids()})


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


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)

"""
streamlit_app.py

Standalone Streamlit demo for SmartPark -- loads both models directly
(no separate Flask API process needed), so this single script can run
locally OR be deployed as-is to Streamlit Community Cloud.

An earlier version talked to a separate Flask API over HTTP specifically
to avoid installing TensorFlow and Streamlit into the same environment
(that combination corrupted a shared native dependency and broke
TensorFlow, on this machine, when streamlit was added into an already-
resolved venv). Deploying to a fresh cloud container is a different
situation -- pip resolves every dependency together from scratch there,
which is a much safer path to the same combination -- so this version
takes that route to get a single, publicly-deployable app.

Run locally (in a fresh venv, separate from .venv/.venv_streamlit):
    pip install -r requirements_streamlit_cloud.txt
    streamlit run streamlit_app.py
"""

import glob
import os
import sys

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector_utils import decode_predictions, draw_detections, preprocess_image
from inference import run_inference, load_spots_for_lot, draw_annotated

HERE = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(HERE, "trained_detector_multibox_mixed", "spot_detector.keras")
CLASSIFIER_PATH = os.path.join(HERE, "trained_model", "parking_spot_cnn.keras")
MANIFEST_PATH = os.path.join(HERE, "synthetic_dataset", "manifest.json")
LOTS_DIR = os.path.join(HERE, "synthetic_dataset", "full_lots")
SHOWCASE_DIR = os.path.join(HERE, "demo_showcase")

st.set_page_config(page_title="SmartPark", page_icon="🅿️", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #f6f6f7; }
    .metric-card {
        background: white; border: 1px solid #e4e4e7; border-radius: 10px;
        padding: 12px 16px; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading detector...")
def load_detector():
    return tf.keras.models.load_model(DETECTOR_PATH, compile=False)


@st.cache_resource(show_spinner="Loading occupancy classifier...")
def load_classifier():
    return tf.keras.models.load_model(CLASSIFIER_PATH)


def cv2_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def bytes_to_cv2(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def run_detector(model, img_bgr):
    orig_h, orig_w = img_bgr.shape[:2]
    resized = preprocess_image(img_bgr)
    pred = model.predict(resized[None, ...], verbose=0)[0]
    detections = decode_predictions(pred, orig_w, orig_h)
    annotated = draw_detections(img_bgr, detections)
    n_empty = sum(1 for d in detections if d["class_name"] == "empty_spot")
    n_occ = sum(1 for d in detections if d["class_name"] == "occupied_spot")
    return annotated, detections, n_empty, n_occ


def render_detection_result(annotated_bgr, detections, n_empty, n_occ):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(cv2_to_pil(annotated_bgr), use_container_width=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>{len(detections)}</h3>spots detected</div>",
                    unsafe_allow_html=True)
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Empty", n_empty)
        m2.metric("Occupied", n_occ)
        st.write("")
        st.dataframe(
            [{"class": d["class_name"], "confidence": f"{d['confidence']*100:.0f}%"}
             for d in sorted(detections, key=lambda d: -d["confidence"])],
            use_container_width=True, height=300,
        )


st.title("SmartPark")
st.caption("Parking spot object detection — upload any photo, or try a calibrated sample lot.")

detector = load_detector()
classifier = load_classifier()

with st.sidebar:
    st.header("Model")
    st.markdown(
        "**Object detector** (primary)\n"
        "- Synthetic val: 90.60% mAP@0.5\n"
        "- Real PKLot val: 68.76% mAP@0.5\n\n"
        "**Occupancy classifier** (secondary, calibrated cameras)\n"
        "- Real PKLot: 98.69% accuracy\n\n"
        "Full results: `report_assets/SmartPark_Report.docx`"
    )

tab_upload, tab_samples = st.tabs(["Upload a photo", "Calibrated sample lots"])

with tab_upload:
    st.write("Runs the object detector directly on your photo — no calibration needed.")

    showcase_paths = sorted(glob.glob(os.path.join(SHOWCASE_DIR, "*.jpg")))
    if showcase_paths:
        st.markdown("**Real photos, verified strong results** — click one to analyze:")
        cols = st.columns(len(showcase_paths))
        for i, path in enumerate(showcase_paths):
            with cols[i]:
                st.image(path, use_container_width=True)
                if st.button("Analyze", key=f"showcase_{i}", use_container_width=True):
                    with open(path, "rb") as f:
                        st.session_state["chosen_bytes"] = f.read()

    st.markdown("**Or upload your own:**")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.session_state["chosen_bytes"] = uploaded.getvalue()

    if st.session_state.get("chosen_bytes"):
        img = bytes_to_cv2(st.session_state["chosen_bytes"])
        if img is None:
            st.error("Could not read that image file.")
        else:
            with st.spinner("Detecting spots..."):
                annotated, detections, n_empty, n_occ = run_detector(detector, img)
            render_detection_result(annotated, detections, n_empty, n_occ)

with tab_samples:
    st.write("Runs the calibrated classifier + geometry check — also flags improperly parked cars.")
    lot_ids = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(LOTS_DIR, "*.jpg"))
    )[:20]

    if lot_ids:
        lot_id = st.selectbox("Sample lot", lot_ids)
        if st.button("Analyze this lot", type="primary"):
            img_path = os.path.join(LOTS_DIR, f"{lot_id}.jpg")
            img = cv2.imread(img_path)
            with st.spinner("Running classifier + geometry check..."):
                spots = load_spots_for_lot(MANIFEST_PATH, img_path)
                results = run_inference(img, spots, classifier)
                annotated = draw_annotated(img, results)

            n_empty = sum(1 for r in results if not r["occupied"])
            n_bad = sum(1 for r in results if r["occupied"] and not r["properly_parked"])

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(cv2_to_pil(annotated), use_container_width=True)
            with col2:
                m1, m2, m3 = st.columns(3)
                m1.metric("Empty", n_empty)
                m2.metric("Occupied", len(results) - n_empty)
                m3.metric("Bad park", n_bad)
                st.write("")
                rows = []
                for r in results:
                    if not r["occupied"]:
                        label = "Empty"
                    elif r["properly_parked"]:
                        label = "Occupied"
                    else:
                        label = "Improperly parked"
                    rows.append({"spot": r["spot_id"], "status": label})
                st.dataframe(rows, use_container_width=True, height=300)

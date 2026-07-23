"""
streamlit_app.py

A Streamlit front-end for the SmartPark system. Deliberately talks to the
existing Flask API (api/app.py) over plain HTTP rather than importing
TensorFlow/OpenCV itself -- this runs in its own virtual environment
(.venv_streamlit), completely separate from the main one (.venv) that
Flask/TensorFlow use. Installing streamlit alongside tensorflow in the
same environment corrupted a shared native library and broke TensorFlow
for every process using that environment; two isolated environments,
talking over HTTP, sidesteps that entirely.

Run (with the Flask API already running separately on :5050):
    python -m venv .venv_streamlit
    source .venv_streamlit/bin/activate
    pip install streamlit requests
    streamlit run streamlit_app.py
"""

import base64
import glob
import io
import os

import requests
import streamlit as st
from PIL import Image

API_BASE = "http://localhost:5050"
HERE = os.path.dirname(os.path.abspath(__file__))
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


def api_healthy():
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def data_uri_to_image(data_uri):
    b64 = data_uri.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def analyze_bytes(image_bytes, filename):
    with st.spinner("Detecting spots..."):
        try:
            resp = requests.post(
                f"{API_BASE}/api/analyze",
                files={"image": (filename, image_bytes)},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            return None


def render_detection_result(data):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(data_uri_to_image(data["annotated_image"]), use_container_width=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>{data['total_spots']}</h3>spots detected</div>",
                    unsafe_allow_html=True)
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Empty", data["empty_spots"])
        m2.metric("Occupied", data["occupied_spots"])
        st.write("")
        st.dataframe(
            [{"class": d["class_name"], "confidence": f"{d['confidence']*100:.0f}%"}
             for d in sorted(data["detections"], key=lambda d: -d["confidence"])],
            use_container_width=True, height=300,
        )


st.title("SmartPark")
st.caption("Parking spot object detection — upload any photo, or try a calibrated sample lot.")

if not api_healthy():
    st.error(
        f"Can't reach the SmartPark API at {API_BASE}. Start it first in another terminal:\n\n"
        "```\ncd smartpark\nsource .venv/bin/activate\npython api/app.py\n```"
    )
    st.stop()

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
                    st.session_state["chosen_name"] = os.path.basename(path)

    st.markdown("**Or upload your own:**")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.session_state["chosen_bytes"] = uploaded.getvalue()
        st.session_state["chosen_name"] = uploaded.name

    if st.session_state.get("chosen_bytes"):
        data = analyze_bytes(st.session_state["chosen_bytes"], st.session_state["chosen_name"])
        if data:
            render_detection_result(data)

with tab_samples:
    st.write("Runs the calibrated classifier + geometry check — also flags improperly parked cars.")
    try:
        lots = requests.get(f"{API_BASE}/api/lots", timeout=10).json().get("lots", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Could not list sample lots: {e}")
        lots = []

    if lots:
        lot_id = st.selectbox("Sample lot", lots[:20])
        if st.button("Analyze this lot", type="primary"):
            with st.spinner("Running classifier + geometry check..."):
                try:
                    resp = requests.get(f"{API_BASE}/api/lots/{lot_id}/status", timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
                    data = None

            if data:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(data_uri_to_image(data["annotated_image"]), use_container_width=True)
                with col2:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Empty", data["empty_spots"])
                    m2.metric("Occupied", data["occupied_spots"])
                    m3.metric("Bad park", data["improperly_parked"])
                    st.write("")
                    rows = []
                    for s in data["spots"]:
                        if not s["occupied"]:
                            label = "Empty"
                        elif s["properly_parked"]:
                            label = "Occupied"
                        else:
                            label = "Improperly parked"
                        rows.append({"spot": s["spot_id"], "status": label})
                    st.dataframe(rows, use_container_width=True, height=300)

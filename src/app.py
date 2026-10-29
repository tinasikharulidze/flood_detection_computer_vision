"""Streamlit demo. Run from the repo root: streamlit run src/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" can be imported

import streamlit as st

from src.inference import DEFAULT_THRESHOLD, FloodSegmenter, make_overlay

st.set_page_config(page_title="SAR Flood Segmentation", layout="wide")


@st.cache_resource
def load_segmenter():
    return FloodSegmenter()  # I cache it so the model is not reloaded on every click


st.title("SAR flood segmentation")
st.write(
    "Upload the **VV** and **VH** images of one Sentinel-1 tile. "
    "The model colors the pixels it thinks are flooded."
)

try:
    segmenter = load_segmenter()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Settings")
    variant = st.selectbox("Model", segmenter.variants, help="fp32 is the accurate one, int8 is smaller and faster.")
    threshold = st.slider("Flood threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01,
                          help="Pixels with a probability above this value count as flood.")

col_vv, col_vh = st.columns(2)
vv_file = col_vv.file_uploader("VV image", type=["png", "jpg", "jpeg", "tif", "tiff"])
vh_file = col_vh.file_uploader("VH image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if vv_file and vh_file:
    try:
        result = segmenter.predict(vv_file.getvalue(), vh_file.getvalue(), variant, threshold)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    rgb, prob, mask = result["rgb"], result["prob"], result["mask"]

    c1, c2, c3, c4 = st.columns(4)
    c1.image(rgb, caption="Composite (VV, VH, VV/VH)", use_container_width=True)
    c2.image(prob, caption="Flood probability", clamp=True, use_container_width=True)
    c3.image(mask * 255, caption="Predicted flood mask", use_container_width=True)
    c4.image(make_overlay(rgb, mask), caption="Overlay (red = flood)", use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Flooded area", f"{result['flood_percent']:.2f} %")
    m2.metric("Flooded pixels", f"{result['flood_pixels']:,} / {result['total_pixels']:,}")
    m3.metric("Inference time", f"{result['latency_ms']:.0f} ms")
else:
    st.info("Upload both the VV and the VH image to see the prediction.")

"""FastAPI service. Run from the repo root: uvicorn src.api:app --port 8000"""
import base64
import io

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from src.inference import DEFAULT_THRESHOLD, FloodSegmenter, make_overlay

app = FastAPI(title="SAR Flood Segmentation API")
segmenter = FloodSegmenter()  # I load the model list once when the server starts


def to_png(image):
    """Turn a float (0-1) or uint8 image into PNG bytes."""
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


async def run_prediction(vv, vh, variant, threshold):
    if not 0 <= threshold <= 1:
        raise HTTPException(400, "threshold must be between 0 and 1")
    try:
        return segmenter.predict(await vv.read(), await vh.read(), variant, threshold)
    except ValueError as error:
        raise HTTPException(400, str(error))


@app.get("/health")
def health():
    return {"status": "ok", "variants": segmenter.variants, "default_threshold": DEFAULT_THRESHOLD}


@app.post("/predict")
async def predict(
    vv: UploadFile = File(...),
    vh: UploadFile = File(...),
    variant: str = Form("fp32"),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    """Return stats and the mask and overlay as base64 PNGs."""
    result = await run_prediction(vv, vh, variant, threshold)
    overlay = make_overlay(result["rgb"], result["mask"])
    b64 = lambda image: base64.b64encode(to_png(image)).decode()
    return {
        "variant": result["variant"],
        "threshold": result["threshold"],
        "latency_ms": round(result["latency_ms"], 1),
        "height": result["height"],
        "width": result["width"],
        "flood_pixels": result["flood_pixels"],
        "flood_percent": round(result["flood_percent"], 3),
        "mask_png_base64": b64(result["mask"] * 255),
        "overlay_png_base64": b64(overlay),
    }


@app.post("/predict/mask")
async def predict_mask(
    vv: UploadFile = File(...),
    vh: UploadFile = File(...),
    variant: str = Form("fp32"),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    """Return only the flood mask as a PNG image (white = flood)."""
    result = await run_prediction(vv, vh, variant, threshold)
    return Response(content=to_png(result["mask"] * 255), media_type="image/png")

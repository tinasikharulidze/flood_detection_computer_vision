"""Flood segmentation inference with ONNX Runtime.

The API and the Streamlit app both use this file, so the model logic lives in one place.
"""
import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRS = [ROOT / "models", ROOT]   # I look for the .onnx files in models/ first, then in the repo root
DEFAULT_THRESHOLD = 0.441              # best threshold I found on the validation set (Model 1)
DIVISOR = 32                           # the U-Net needs height and width divisible by 32


def find_models():
    """Return {"fp32": path, "int8": path} for every model_1_*.onnx file I find."""
    found = {}
    for folder in MODEL_DIRS:
        for path in sorted(folder.glob("model_1_*.onnx")):
            variant = path.stem.replace("model_1_", "")
            found.setdefault(variant, path)
    return found


def decode_gray(file_bytes):
    """Decode uploaded image bytes into a 2D uint8 array. I return None if the file is not an image."""
    try:
        return np.array(Image.open(io.BytesIO(file_bytes)).convert("L"))
    except (UnidentifiedImageError, OSError):
        return None


def s1_to_rgb(vv, vh):
    """Stack VV, VH and VV/VH into 3 channels, the same input I used for training."""
    eps = 1e-6
    ratio = np.clip(np.nan_to_num(vv / (vh + eps), nan=0.0), 0, 1)
    return np.stack((vv, vh, ratio), axis=2).astype(np.float32)


def pad_to_multiple(image, multiple=DIVISOR):
    """Pad the image (reflect) so height and width are divisible by 32."""
    h, w = image.shape[:2]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")


def make_overlay(rgb, mask, alpha=0.5):
    """Paint the flood mask red on top of the RGB composite (both are 0-1 floats)."""
    overlay = rgb.copy()
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    overlay[mask == 1] = (1 - alpha) * overlay[mask == 1] + alpha * red
    return overlay


class FloodSegmenter:
    def __init__(self):
        self.model_paths = find_models()
        if not self.model_paths:
            raise FileNotFoundError(
                "I could not find any model_1_*.onnx file. Put it in the repo root or in models/."
            )
        self.sessions = {}  # I load each model only when someone asks for it

    @property
    def variants(self):
        # I show fp32 first because it is the default
        return sorted(self.model_paths, key=lambda v: v != "fp32")

    def _session(self, variant):
        if variant not in self.model_paths:
            raise ValueError(f"Unknown variant '{variant}'. Available: {self.variants}")
        if variant not in self.sessions:
            self.sessions[variant] = ort.InferenceSession(
                str(self.model_paths[variant]), providers=["CPUExecutionProvider"]
            )
        return self.sessions[variant]

    def predict(self, vv_bytes, vh_bytes, variant="fp32", threshold=DEFAULT_THRESHOLD):
        vv = decode_gray(vv_bytes)
        vh = decode_gray(vh_bytes)
        if vv is None or vh is None:
            raise ValueError("I could not read one of the files as an image.")
        if vv.shape != vh.shape:
            raise ValueError(f"VV {vv.shape} and VH {vh.shape} must have the same size.")

        # Same preprocessing as training: scale to 0-1, then build the 3 channels
        rgb = s1_to_rgb(vv.astype(np.float32) / 255.0, vh.astype(np.float32) / 255.0)
        h, w = rgb.shape[:2]
        batch = pad_to_multiple(rgb).transpose(2, 0, 1)[None]  # (1, 3, H, W)

        session = self._session(variant)
        start = time.perf_counter()
        logits = session.run(None, {session.get_inputs()[0].name: batch})[0]
        latency_ms = (time.perf_counter() - start) * 1000

        # Model 1 has 1 output channel, so I use sigmoid to get the flood probability
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits[0, 0], -30, 30)))
        prob = prob[:h, :w]  # remove the padding again
        mask = (prob >= threshold).astype(np.uint8)

        return {
            "variant": variant,
            "threshold": threshold,
            "latency_ms": latency_ms,
            "height": h,
            "width": w,
            "flood_pixels": int(mask.sum()),
            "total_pixels": int(mask.size),
            "flood_percent": float(mask.mean() * 100),
            "rgb": rgb,
            "prob": prob,
            "mask": mask,
        }

# Flood detection from Sentinel-1 SAR (ETCI 2021)

U-Net (ResNet34 encoder) that finds flooded pixels in 256×256 Sentinel-1 tiles.
Input: the VV and VH polarization images of one tile. Output: a flood mask.

## Setup

```bash
uv sync
```

## Notebooks (training)

1. `00_EDA.ipynb` – look at the data
2. `01_Training_Evaluation.ipynb` – train and compare the models, Model 1 is the final one
3. `02_Quantization.ipynb` – export to ONNX and make the INT8 version

## Models

Put the ONNX files in the repo root or in a `models/` folder:

- `model_1_fp32.onnx` – default, most accurate
- `model_1_int8.onnx` – optional, 4x smaller and faster, but loses some F1 (about 6 points in my test)

The demo shows every `model_1_*.onnx` file it finds.

## Run the demo

Start the API (http://localhost:8000/docs):

```bash
uv run uvicorn src.api:app --port 8000
```

Start the Streamlit app (http://localhost:8501), in a second terminal:

```bash
uv run streamlit run src/app.py
```

Call the API from the terminal:

```bash
curl -X POST http://localhost:8000/predict \
  -F vv=@tile_vv.png -F vh=@tile_vh.png -F variant=fp32 -F threshold=0.441
```

- `GET /health` – available models and default threshold
- `POST /predict` – stats plus mask and overlay as base64 PNG
- `POST /predict/mask` – the mask as a PNG image

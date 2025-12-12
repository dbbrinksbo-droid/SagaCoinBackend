# modules/model_loader.py — SagaMoent AI Model Loader (LOCAL + Railway safe)

import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

# Paths (relative to project root)
MODEL_PATH = "sagacoin_full_model.onnx"
LABELS_FILE = "labels.json"

_session = None
_labels = None


def download_model_if_needed():
    """
    Lokal-first strategi:
    - Hvis modellen findes lokalt → brug den
    - Hvis den IKKE findes → forsøg MODEL_URL (Railway / prod)
    """

    # ✅ Lokal model findes → brug den
    if os.path.exists(MODEL_PATH):
        print("✔ Using local ONNX model")
        return True

    # ⛔ Lokal model mangler → prøv MODEL_URL
    model_url = os.getenv("MODEL_URL")

    if not model_url:
        print("❌ MODEL_URL not set and local model missing")
        return False

    print("⬇ Downloading ONNX model from MODEL_URL")

    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        print("✔ Model downloaded successfully")
        return True
    except Exception as e:
        print("❌ MODEL DOWNLOAD FAILED:", e)
        return False


def load_labels():
    global _labels

    if _labels is not None:
        return _labels

    if not os.path.exists(LABELS_FILE):
        print("⚠ labels.json not found")
        _labels = []
        return _labels

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sort labels by index
    _labels = [label for label, idx in sorted(data.items(), key=lambda x: x[1])]
    print(f"✔ Loaded {len(_labels)} labels")

    return _labels


def load_model():
    global _session

    if _session is not None:
        return _session

    if not download_model_if_needed():
        raise RuntimeError("ONNX model unavailable")

    print("🔄 Loading ONNX model…")

    _session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    print("✔ ONNX model ready")
    return _session


def preprocess(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)   # CHW
    arr = arr[np.newaxis, :]       # NCHW
    return arr


def predict_image(img: Image.Image):
    session = load_model()
    labels = load_labels()

    arr = preprocess(img)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: arr})
    vector = outputs[0][0]

    idx = int(np.argmax(vector))
    conf = float(vector[idx])
    label = labels[idx] if idx < len(labels) else f"label_{idx}"

    return {
        "label": label,
        "confidence": conf,
        "index": idx
    }

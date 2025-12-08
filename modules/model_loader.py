import os
import json
import numpy as np
import requests
from PIL import Image
import onnxruntime as ort

# ----------------------------------
# 1) LOAD MODEL FROM GOOGLE DRIVE
# ----------------------------------

MODEL_FILE = "sagacoin_full_model.onnx"
LABELS_FILE = "labels.json"

MODEL_URL = os.getenv("MODEL_URL")  # Railway variable


def ensure_model_exists():
    """Downloader modellen hvis den ikke findes lokalt."""
    if os.path.exists(MODEL_FILE):
        print("✔ Model findes lokalt.")
        return

    if not MODEL_URL:
        raise RuntimeError("❌ MODEL_URL mangler i Railway Variables!")

    print(f"⬇ Downloader model fra: {MODEL_URL}")

    resp = requests.get(MODEL_URL, allow_redirects=True)
    if resp.status_code != 200:
        raise RuntimeError(f"❌ Kunne ikke downloade modellen ({resp.status_code})")

    with open(MODEL_FILE, "wb") as f:
        f.write(resp.content)

    print("✔ Model downloaded og gemt.")


# ----------------------------------
# 2) LOAD LABELS
# ----------------------------------

def load_labels():
    if not os.path.exists(LABELS_FILE):
        print("⚠ labels.json mangler — laver fallback labels.")
        return [f"label_{i}" for i in range(100)]

    with open(LABELS_FILE, "r") as f:
        data = json.load(f)

    labels = [label for label, idx in sorted(data.items(), key=lambda x: x[1])]
    print("✔ Labels loaded:", len(labels))
    return labels


# ----------------------------------
# 3) LOAD MODEL INTO ONNX RUNTIME
# ----------------------------------

_session = None
_labels = None


def get_session():
    global _session
    if _session:
        return _session

    ensure_model_exists()

    print("🔄 Loader ONNX model…")
    _session = ort.InferenceSession(MODEL_FILE, providers=["CPUExecutionProvider"])
    print("✔ ONNX model loaded.")
    return _session


def get_labels():
    global _labels
    if _labels:
        return _labels

    _labels = load_labels()
    return _labels


# ----------------------------------
# 4) PREPROCESS + PREDICT
# ----------------------------------

def preprocess(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(img: Image.Image):
    session = get_session()
    labels = get_labels()

    arr = preprocess(img)
    input_name = session.get_inputs()[0].name

    output = session.run(None, {input_name: arr})
    vector = output[0][0]

    idx = int(np.argmax(vector))
    conf = float(vector[idx])

    label = labels[idx] if idx < len(labels) else f"label_{idx}"

    return {
        "label": label,
        "confidence": conf,
        "index": idx
    }

import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "sagacoin_full_model.onnx")
LABELS_PATH = os.path.join(BASE_DIR, "..", "models", "labels.json")

_session = None
_labels = None


# --------------------------------------------------------------------
# LOAD LABELS
# --------------------------------------------------------------------
def load_labels():
    global _labels
    if _labels is not None:
        return _labels

    if not os.path.exists(LABELS_PATH):
        print("⚠️ WARNING: labels.json mangler!")
        _labels = []
        return _labels

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # labels.json er et dict → vi sorterer efter index
    sorted_labels = sorted(data.items(), key=lambda x: x[1])
    _labels = [item[0] for item in sorted_labels]

    print(f"✔ Labels indlæst ({len(_labels)} classes)")
    return _labels


# --------------------------------------------------------------------
# LOAD MODEL (ONCE)
# --------------------------------------------------------------------
def load_model():
    global _session
    if _session is not None:
        return _session

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model fil ikke fundet: {MODEL_PATH}")

    print("✔ Loader sagacoin_full_model.onnx ...")

    _session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    print("✔ Full-model loaded successfully")
    return _session


# --------------------------------------------------------------------
# PREPROCESS
# --------------------------------------------------------------------
def preprocess(img: Image.Image):
    """
    Standardiseret preprocess:
    - resized til modelens input (224x224 typisk)
    - RGB float32
    - normaliseret til 0-1
    - CHW (3,224,224)
    """
    img = img.resize((224, 224)).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)
    return arr


# --------------------------------------------------------------------
# PREDICT (MAIN FUNCTION)
# --------------------------------------------------------------------
def predict_image(img: Image.Image):
    session = load_model()
    labels = load_labels()

    arr = preprocess(img)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: arr})
    vector = outputs[0][0]

    idx = int(np.argmax(vector))
    conf = float(np.max(vector))

    if labels and idx < len(labels):
        label = labels[idx]
    else:
        label = f"class_{idx}"

    return {
        "label": label,
        "confidence": conf,
        "index": idx,
        "raw": vector.tolist(),
        "num_labels": len(labels)
    }

import os
import shutil

SRC = "/app/sagacoin_full_model.onnx"
DST_DIR = "/models"
DST = "/models/sagacoin_full_model.onnx"

print("🔧 BOOTSTRAP MODEL COPY")

os.makedirs(DST_DIR, exist_ok=True)

if not os.path.exists(DST):
    print("➡️ Copying model to volume...")
    shutil.copy2(SRC, DST)
    print("✅ Model copied to /models")
else:
    print("ℹ️ Model already exists in /models")

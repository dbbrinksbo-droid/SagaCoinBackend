import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from modules.unified_analyzer_v16 import analyze_image_v16

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "server": "SagaMoent - V16 AI Engine",
        "status": "online",
        "model": "unified_analyzer_v16",
        "version": "16.0",
        "time": int(time.time())
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files["image"]

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)

        print("⚙️ Running V16 analyzer on:", filepath)

        result = analyze_image_v16(filepath)

        return jsonify({
            "success": True,
            "engine": "V16",
            "result": result
        }), 200

    except Exception as e:
        print("🔥 SERVER ERROR:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

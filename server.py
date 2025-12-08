import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO

from modules.unified_analyzer_v16 import analyze_coin_full

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "server": "SagaMoent Backend V16",
        "status": "online",
        "model": "sagacoin_full_model.onnx"
    })


@app.route("/analyze", methods=["POST"])
def analyze_route():
    try:
        if "front" not in request.files:
            return jsonify({"success": False, "error": "Missing front image"}), 400
        if "back" not in request.files:
            return jsonify({"success": False, "error": "Missing back image"}), 400

        front_bytes = request.files["front"].read()
        back_bytes = request.files["back"].read()

        front_img = Image.open(BytesIO(front_bytes)).convert("RGB")
        back_img = Image.open(BytesIO(back_bytes)).convert("RGB")

        user_input = request.form.get("userInput", "{}")

        result = analyze_coin_full(front_img, back_img, user_input)

        return jsonify({
            "success": True,
            "engine": "SagaMoent V16",
            "result": result
        })

    except Exception as e:
        print("🔥 SERVER ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

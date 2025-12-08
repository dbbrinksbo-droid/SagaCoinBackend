import os
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from modules.analyzer_v3 import analyze_full_coin_v3

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "server": "SagaMoent Backend",
        "status": "online",
        "endpoint": "/full-analyze-v3"
    }), 200


# ---------------------------------------------------------
# 🔥 FULL ANALYZE V3 (Forside + Bagside + OCR + GPT + Metadata)
# ---------------------------------------------------------
@app.route("/full-analyze-v3", methods=["POST"])
def full_analyze_v3():
    try:
        if "front" not in request.files:
            return jsonify({"success": False, "error": "Missing front image"}), 400
        if "back" not in request.files:
            return jsonify({"success": False, "error": "Missing back image"}), 400

        # Read image bytes
        front_bytes = request.files["front"].read()
        back_bytes = request.files["back"].read()

        # Optional user input JSON
        user_input_raw = request.form.get("userInput", "{}")

        # Run analyzer
        result = analyze_full_coin_v3(front_bytes, back_bytes, user_input_raw)

        return jsonify({
            "success": True,
            "engine": "SagaMoent-V12",
            "result": result
        }), 200

    except Exception as e:
        print("🔥 SERVER ERROR:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

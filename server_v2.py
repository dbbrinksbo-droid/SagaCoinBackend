from flask import Flask, request, jsonify
from flask_cors import CORS

from modules.analyzer_v3 import analyze_full_coin_v3

print("🔥🔥🔥 NEW SERVER_V2 ACTIVE — VISION ONLY 🔥🔥🔥")

app = Flask(__name__)
CORS(app)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "front" not in request.files:
        return jsonify({"success": False, "error": "Missing front image"}), 400

    front_bytes = request.files["front"].read()
    back_bytes = request.files["back"].read() if "back" in request.files else None

    result = analyze_full_coin_v3(front_bytes, back_bytes)

    return jsonify({
        "engine": "SagaMoent V12 — VISION ONLY",
        "success": True,
        "result": result
    })


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

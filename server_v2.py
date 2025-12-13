from flask import Flask, request, jsonify
from flask_cors import CORS

from modules.analyzer_v3 import analyze_full_coin_v3

print("🔥 SagaMoent Backend — FULL COIN ANALYSIS ACTIVE")

app = Flask(__name__)
CORS(app)


@app.route("/full-analyze-v3", methods=["POST"])
def full_analyze_v3():
    front_file = request.files.get("front")
    back_file = request.files.get("back")
    user_input = request.form.get("user_input")

    if not front_file:
        return jsonify({"success": False, "error": "Missing front image"}), 400

    front_bytes = front_file.read()
    back_bytes = back_file.read() if back_file else None

    result = analyze_full_coin_v3(
        front_bytes=front_bytes,
        back_bytes=back_bytes,
        user_input=user_input,
    )

    return jsonify({
        "success": True,
        "engine": "SagaMoent V13",
        "result": result
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


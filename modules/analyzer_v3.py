from io import BytesIO
from PIL import Image

from modules.model_loader import predict_image
from modules.vision_gpt_reader import read_coin_from_image


def analyze_full_coin_v3(front_bytes, back_bytes=None):
    # 🔥 VISION FIRST – HARD RULE
    visual = read_coin_from_image(front_bytes)

    # 🔎 Model prediction (classification only)
    front_img = Image.open(BytesIO(front_bytes)).convert("RGB")
    front_pred = predict_image(front_img)

    result = {
        "visual": visual,
        "front_prediction": front_pred
    }

    if back_bytes:
        back_img = Image.open(BytesIO(back_bytes)).convert("RGB")
        result["back_prediction"] = predict_image(back_img)

    return result

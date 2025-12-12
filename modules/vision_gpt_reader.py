# modules/vision_gpt_reader.py
# SagaMoent Vision GPT Reader — IMAGE → TEXT (STRICT)

import base64
from openai import OpenAI

client = OpenAI()


def read_coin_from_image(image_bytes: bytes):
    """
    Uses GPT-4 Vision to read all visible text, symbols and details
    directly from the coin image.
    """

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional numismatist. "
                    "Read ONLY what is visually present on the coin image. "
                    "Do NOT guess. Do NOT hallucinate. "
                    "If something is not visible, say NOT VISIBLE."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read the coin carefully. "
                            "Extract front text, back text, symbols, numbers, "
                            "dates, mint marks and any visible details."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}"
                        },
                    },
                ],
            },
        ],
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()

    return {
        "front_text": text,
        "back_text": text,
        "symbols": [],
    }

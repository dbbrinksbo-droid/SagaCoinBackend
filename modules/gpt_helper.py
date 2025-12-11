import os
from openai import OpenAI

def gpt_enhance(prediction_text, ocr_text):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "Ingen GPT-forklaring (mangler API-nøgle)."

    try:
        client = OpenAI(api_key=key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du er møntekspert."},
                {"role": "user", "content":
                    f"Analyse dette fund:\n"
                    f"- Prediction: {prediction_text}\n"
                    f"- OCR: {ocr_text}\n"
                    f"Giv en kort dansk forklaring om mønten."}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"GPT-fejl: {e}"

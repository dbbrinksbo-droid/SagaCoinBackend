FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Lav models-folder (i tilfælde den mangler)
RUN mkdir -p /app/models

# Kopiér ONNX fil ind i volume mount
COPY sagacoin_full_model.onnx /app/models/sagacoin_full_model.onnx

COPY . .

ENV PORT=8080

CMD ["python3", "server_v2.py"]

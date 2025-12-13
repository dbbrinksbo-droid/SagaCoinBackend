FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 👇 COPY MODEL INTO IMAGE (KUN til bootstrap)
COPY sagacoin_full_model.onnx /app/sagacoin_full_model.onnx

COPY . .

RUN chmod +x /app/start.sh

ENV PYTHONUNBUFFERED=1

CMD ["/app/start.sh"]

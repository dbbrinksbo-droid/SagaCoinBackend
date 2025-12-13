FROM python:3.10-slim

WORKDIR /app

# System deps (curl + gdown needs this)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files (NO ONNX HERE)
COPY . .

# Ensure start script is executable
RUN chmod +x /app/start.sh

ENV PYTHONUNBUFFERED=1

CMD ["/app/start.sh"]

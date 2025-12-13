#!/bin/sh
set -e

echo "🚨🚨🚨 HARD RESET start.sh 🚨🚨🚨"
echo "PWD=$(pwd)"
echo "FILES IN /:"
ls -la /

MODEL_DST="/tmp/sagacoin_full_model.onnx"

if [ ! -f "$MODEL_DST" ]; then
  echo "⬇️ Downloading ONNX model with curl..."
  curl -L "$MODEL_URL" -o "$MODEL_DST"
  echo "✅ Model downloaded to /tmp"
else
  echo "✅ Model already exists in /tmp"
fi

exec python server_v2.py

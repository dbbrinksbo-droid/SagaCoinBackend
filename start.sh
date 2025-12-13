#!/usr/bin/env bash
set -e

echo "🚀 SagaMoent Backend starting..."
echo "📂 Working dir: $(pwd)"
echo "📂 Files:"
ls -la

python server_v2.py

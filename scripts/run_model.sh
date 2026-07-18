#!/usr/bin/env bash
# Start llama-server with a vision-capable Gemma GGUF, CPU-only.
#
# Preferred model: Gemma "4" E2B Q4_K_M multimodal GGUF — as of this writing no
# such GGUF is published, so the DEFAULT below is the spec's fallback:
# Gemma 3 4B-it (QAT Q4_0/Q4_K_M) from ggml-org, which ships with its vision
# projector (mmproj) and is auto-downloaded by llama.cpp's -hf flag on first
# run (~3.2 GB into ~/Library/Caches/llama.cpp).
#
# When a Gemma 4 E2B GGUF lands, point MODEL_HF at it and nothing else changes:
#   MODEL_HF=ggml-org/gemma-4-e2b-it-GGUF ./scripts/run_model.sh
#
# Manual download URL (documented for the README):
#   https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF
set -euo pipefail

MODEL_HF="${MODEL_HF:-ggml-org/gemma-3-4b-it-GGUF}"
PORT="${MODEL_PORT:-8080}"
CTX="${MODEL_CTX:-4096}"
THREADS="${MODEL_THREADS:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server not found."
  echo "Install with:  brew install llama.cpp"
  echo "or build from source: https://github.com/ggml-org/llama.cpp"
  exit 1
fi

echo "Starting llama-server on :${PORT} with ${MODEL_HF} (ctx ${CTX}, ${THREADS} threads)"
echo "First run downloads the GGUF + mmproj automatically (a few GB)."

# -hf pulls the GGUF (and its mmproj for vision) from Hugging Face if missing.
# --no-webui: we only need the OpenAI-compatible API.
exec llama-server \
  -hf "${MODEL_HF}" \
  --host 127.0.0.1 --port "${PORT}" \
  -c "${CTX}" \
  -t "${THREADS}" \
  --no-webui

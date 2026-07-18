#!/usr/bin/env bash
# Start llama-server with a vision-capable Gemma GGUF, CPU-only.
#
# DEFAULT: Gemma 4 E2B instruction-tuned, dynamic 4-bit GGUF from Unsloth
# (https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF), auto-downloaded with
# its vision projector (mmproj) by llama.cpp's -hf flag on first run.
# Requires a recent llama.cpp (brew upgrade llama.cpp if vision doesn't load).
#
# Fallback if E2B multimodal misbehaves on your llama.cpp build:
#   MODEL_HF=ggml-org/gemma-3-4b-it-GGUF ./scripts/run_model.sh
set -euo pipefail

MODEL_HF="${MODEL_HF:-unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL}"
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

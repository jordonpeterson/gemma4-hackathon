#!/usr/bin/env bash
# Start llama-server with a vision-capable Gemma GGUF, CPU-only.
#
# DEFAULT: Gemma 4 E4B instruction-tuned, dynamic 4-bit GGUF from Unsloth
# (https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF), auto-downloaded with
# its vision projector (mmproj) by llama.cpp's -hf flag on first run.
# Requires a recent llama.cpp (brew upgrade llama.cpp if vision doesn't load).
#
# E4B (not the smaller E2B) is required for the empty-bin vision task: E2B
# consistently hallucinates contents into empty containers and cannot tell an
# empty bin from a full one. E4B perceives it correctly. Tradeoff: ~2x slower
# on CPU (~60s/image vs ~30s). To try the smaller/faster model anyway:
#   MODEL_HF=unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL ./scripts/run_model.sh
set -euo pipefail

MODEL_HF="${MODEL_HF:-unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL}"
PORT="${MODEL_PORT:-8080}"
CTX="${MODEL_CTX:-4096}"
# Physical cores, not logical: CPU inference is memory-bound and
# hyperthread oversubscription usually slows it down.
THREADS="${MODEL_THREADS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 4)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
# --parallel 1: Sentinel serializes requests anyway; a single slot means the
#   long constant system prompt stays cached between calls (big speedup on CPU).
# --mlock: keep the model resident in RAM between poll cycles.
# --cache-reuse 256: allow prefix-cache reuse across slightly-shifted prompts.
exec llama-server \
  -hf "${MODEL_HF}" \
  --host 127.0.0.1 --port "${PORT}" \
  -c "${CTX}" \
  -t "${THREADS}" \
  --parallel 1 --mlock --cache-reuse 256 \
  --no-webui ${EXTRA_ARGS}

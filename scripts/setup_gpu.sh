#!/usr/bin/env bash
# Set up a fresh Linux GPU machine (RunPod or Lambda) from the repo root:
#   bash scripts/setup_gpu.sh
set -euo pipefail

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

uv sync --locked --extra gpu

if [ ! -f .env ]; then
  echo "warning: no .env file; copy .env.example to .env and fill in the tokens"
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run python - <<'PY'
import torch, transformers, trl, vllm

print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}, bf16: {torch.cuda.is_bf16_supported()}")
print(f"vllm {vllm.__version__}, trl {trl.__version__}, transformers {transformers.__version__}")
PY

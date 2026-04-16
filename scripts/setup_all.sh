#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
fi

PYTHON_BIN="$VENV_PATH/bin/python"

"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install torch --index-url "$TORCH_INDEX_URL"
"$PYTHON_BIN" -m pip install -r requirements-dev.txt
USE_CUDA=1 FORCE_CUDA=1 "$PYTHON_BIN" -m pip install --no-build-isolation -e .

echo "Setup complete."
echo "Activate with: source $VENV_PATH/bin/activate"

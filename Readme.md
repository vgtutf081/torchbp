# Torchbp

![SAR image](https://github.com/Ttl/torchbp/blob/master/docs/img/07_19_1_autofocus_sigma0_pol_cal_pauli.png?raw=true)

Fast C++ Pytorch extension for differentiable synthetic aperture radar image formation and autofocus library on GPU.

Only Nvidia GPUs are supported.

This project is configured for GPU-first workflows. Example scripts in `examples/`
require a CUDA-capable PyTorch environment and run on GPU.

On RTX 3090 Ti backprojection on polar grid achieves 370 billion
backprojections/s and fast factorized backprojection (ffbp) is ten times faster on
moderate size images.

## Installation

### Compatibility (important)

Torchbp is tested with:

- NVIDIA driver with CUDA support
- CUDA Toolkit `13.0` (`nvcc --version` should report 13.0)
- PyTorch CUDA wheels for `cu130`

Quick check:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

Expected:

- `torch.version.cuda` shows `13.0`
- `cuda available: True`

If your CUDA Toolkit version differs, install a matching PyTorch CUDA build.

### From source

```bash
git clone https://github.com/Ttl/torchbp.git
cd torchbp
pip install --no-build-isolation -e .
```

### Linux (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu130
USE_CUDA=1 FORCE_CUDA=1 pip install --no-build-isolation -e .
```

### Windows (PowerShell)

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
$env:USE_CUDA='1'
$env:FORCE_CUDA='1'
python -m pip install --no-build-isolation -e .
```

### One-shot setup scripts

Windows (PowerShell):

```powershell
./scripts/setup_all.ps1
```

Linux:

```bash
bash ./scripts/setup_all.sh
```

These scripts create a venv (if needed), install CUDA PyTorch wheels,
install all runtime/dev/API dependencies, and install `torchbp` in editable mode.

## GPU-only runtime checks

The helper module `torchbp.gpu` provides:

- `require_cuda()`
- `require_cuda_kernels([...])`

Examples require CUDA and run on GPU. If a specific native CUDA kernel is not
available, scripts print a warning and use a GPU fallback path where implemented.

## Example pipeline

1. Download `sar.safetensors` from: https://hforsten.com/sar.safetensors.zip
2. Place it in `examples/`.

If the file is already present in `examples/`, skip steps 1-2.

### Use the correct environment (important)

Run examples only from an environment where PyTorch sees CUDA.

Windows (PowerShell):

```powershell
& .\.venv311\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print('cuda build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

Expected: `cuda available: True`.

If it prints `False`, you activated the wrong environment (for example `.venv` with `torch+cpu`).

### Run

```bash
python examples/sar_process_safetensor.py sar.safetensors
python examples/sar_polar_to_cart.py sar_img.p
```

Expected outputs:

- `sar_img.png`
- `sar_img.p`
- `sar_img_cart.png`

## Documentation

API documentation and examples can be built with sphinx.

```bash
pip install .[docs]
cd docs
make html
```

Open `docs/build/html/index.html`.

## Testing

Run the new fast end-to-end smoke test (generates synthetic safetensors,
processes it, and verifies output artifacts):

```bash
python -m unittest tests.test_examples_smoke -v
```

Run the full operator test suite:

```bash
python -m unittest tests.test_torchbp -v
```

Notes:

- `tests.test_examples_smoke` is CUDA-only and auto-skips if CUDA is unavailable.
- The smoke test is intended for quick validation in development and CI.

## API

Minimal API server is provided in `api/app.py`.

### Start API

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### Process SAR file

Endpoint: `POST /process`

Form fields:

- `file`: input `.safetensors` file (required)
- `nsweeps`: number of sweeps (default: `10000`)
- `fft_oversample`: FFT oversampling factor (default: `1.5`)
- `dpi`: output PNG DPI (default: `700`)
- `max_side`: optional max output side in pixels

The endpoint returns `sar_img_cart.png` directly as binary response.

Windows (PowerShell) example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/process" ^
	-F "file=@examples/sar.safetensors" ^
	-F "nsweeps=10000" ^
	-F "fft_oversample=1.5" ^
	-F "dpi=300" ^
	--output sar_img_cart_api.png
```

Linux/macOS example:

```bash
curl -X POST "http://127.0.0.1:8000/process" \
	-F "file=@examples/sar.safetensors" \
	-F "nsweeps=10000" \
	-F "fft_oversample=1.5" \
	-F "dpi=300" \
	--output sar_img_cart_api.png
```

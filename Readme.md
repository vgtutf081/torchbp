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

### Run with JSON config

You can keep processing parameters in separate JSON files:

```bash
python examples/sar_process_safetensor.py --config examples/sar_process_config.json
python examples/sar_polar_to_cart.py --config examples/sar_polar_to_cart_config.json
```

Provided templates:

- `examples/sar_process_config.json`
- `examples/sar_polar_to_cart_config.json`

CLI flags still work and can override JSON values when needed.

Compatibility note: JSON config only controls runtime parameters. It does not
change the safetensors format, so existing/old `sar.safetensors` files continue
to work as before.

Expected outputs:

- `sar_img.png`
- `sar_img.p`
- `sar_img_cart.png`
- `sar_img.tif` (GeoTIFF)
- `sar_img_cart.pgw` (optional companion world file, only with `--write-world-file`)

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

Run core API/pipeline unit tests:

```bash
python -m unittest \
	tests.test_api_jobs \
	tests.test_job_store \
	tests.test_jobs_prepare \
	tests.test_pipeline_progress \
	tests.test_profiles \
	tests.test_output_formats \
	tests.test_validation \
	tests.test_telemetry_logging -v
```

Run the full operator test suite:

```bash
python -m unittest tests.test_torchbp -v
```

Notes:

- `tests.test_examples_smoke` is CUDA-only and auto-skips if CUDA is unavailable.
- The smoke test is intended for quick validation in development and CI.

## API

Production-style API server is provided in `api/app.py`.

### Start API

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### Submit processing job

Endpoint: `POST /jobs` (back-compat alias: `POST /process`)

Form fields:

- `file`: input `.safetensors` file (required)
- `profile`: `fast_preview | standard | high_quality` (default: `standard`)
- `nsweeps`: number of sweeps (CLI/JSON override profile default)
- `fft_oversample`: FFT oversampling factor (CLI/JSON override profile default)
- `dpi`: output PNG DPI (CLI/JSON override profile default)
- `max_side`: optional max output side in pixels
- `write_world_file`: optional boolean (default: `false`), exports `.pgw`

### Job status model

Job statuses:

- `queued`
- `validating`
- `running`
- `canceling`
- `canceled`
- `success`
- `failed`

`GET /jobs/{job_id}` returns:

- `stage` (current stage name)
- `stage_progress` (0..100 within stage)
- `overall_progress` (0..100 weighted total)
- `cancel_requested` (bool)
- `error_class` (`validation_error | transient_infra_error | resource_error | algorithm_error`)

Progress is weighted by stage (not guessed from wall-clock):

- ingest = 5%
- validation = 5%
- range compression = 15%
- backprojection = 35%
- autofocus = 30%
- export = 10%

### Cancellation semantics

Endpoint: `POST /jobs/{job_id}/cancel`

- Cancellation is accepted in `queued | validating | running`
- Job switches to `canceling`, worker checks cancel flag between long stages
- Final state becomes `canceled`

### Input validation before enqueue

Validation is executed before queueing and rejects invalid payloads with HTTP `422`.

Checks include:

- mission metadata fields and numeric sanity
- trajectory shape, monotonic timestamps, speed/acceleration sanity
- signal sanity (finite values, minimum sweeps/samples, saturation/DC-bias hints)
- dimensional consistency across signal/trajectory/timestamps

On success API returns `job_id` and queue `task_id`.

Status endpoints:

- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/manifest`
- `POST /jobs/{job_id}/cancel`

### Idempotency and versioning

Request deduplication uses a fingerprint composed of:

- input fingerprint (filename + payload hash)
- processing fingerprint (parameters + pipeline/version metadata)

Version fields included in fingerprint:

- `pipeline_version`
- `algorithm_version`
- `profile_version`
- `schema_version`
- optional `calibration_version`
- optional `dem_version`

These are configured by env vars:

- `TORCHBP_PIPELINE_VERSION`
- `TORCHBP_ALGORITHM_VERSION`
- `TORCHBP_PROFILE_VERSION`
- `TORCHBP_SCHEMA_VERSION`
- `TORCHBP_CALIBRATION_VERSION`
- `TORCHBP_DEM_VERSION`

### Artifacts per job

Storage prefix:

- `jobs/{job_id}/...`

Layout:

```text
jobs/{job_id}/
	inputs/
	manifest.json
	metrics/
	logs/
	intermediates/
	outputs/
	previews/
	debug/
```

Typical artifacts:

- `inputs/input.safetensors`
- `intermediates/sar_img.p`
- `previews/sar_img_cart.png`
- `outputs/sar_img.tif`
- `outputs/sar_img_cart.pgw` (optional)
- `logs/*.log`
- `metrics/timings.json`
- `manifest.json`

Manifest contains pipeline passport metadata:

- `job_id`
- `pipeline_version`, `algorithm_version`, `schema_version`
- `profile`
- `processing_parameters`
- `input_artifacts`, `output_artifacts` with `sha256`
- `timings`, `stage_weights`, `warnings`, `quality_metrics`

Storage backend is configurable:

- Local filesystem (`TORCHBP_STORAGE_BACKEND=local`)
- S3/MinIO (`TORCHBP_STORAGE_BACKEND=s3` + S3 env vars)

Windows (PowerShell) example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/jobs" ^
	-F "file=@examples/sar.safetensors" ^
	-F "profile=standard" ^
	-F "dpi=300" ^
	-F "max_side=2048" ^
	-F "write_world_file=false"
```

Linux/macOS example:

```bash
curl -X POST "http://127.0.0.1:8000/jobs" \
	-F "file=@examples/sar.safetensors" \
	-F "profile=fast_preview" \
	-F "dpi=300" \
	-F "max_side=1024" \
	-F "write_world_file=false"
```

Check status:

```bash
curl "http://127.0.0.1:8000/jobs/<job_id>"
```

Cancel running job:

```bash
curl -X POST "http://127.0.0.1:8000/jobs/<job_id>/cancel"
```

## CI

GitHub Actions workflows are included:

- `.github/workflows/ci.yml` with separate `smoke` and `regression` jobs
- `.github/workflows/nightly-benchmark.yml` for scheduled benchmarks

CI split:

- `smoke`: tiny/synthetic quick health checks
- `regression`: broader operator regression suite
- `nightly benchmark`: throughput/latency benchmark on schedule

All workflows upload logs as artifacts.

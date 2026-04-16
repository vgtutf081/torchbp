param(
    [string]$VenvPath = ".venv311",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
}

$python = Join-Path $VenvPath "Scripts\python.exe"

& $python -m pip install -U pip
& $python -m pip install torch --index-url $TorchIndexUrl
& $python -m pip install -r requirements-dev.txt

$env:USE_CUDA = "1"
$env:FORCE_CUDA = "1"
& $python -m pip install --no-build-isolation -e .

Write-Host "Setup complete."
Write-Host "Activate with: & .\$VenvPath\Scripts\Activate.ps1"

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESS_SCRIPT = REPO_ROOT / "examples" / "sar_process_safetensor.py"
CART_SCRIPT = REPO_ROOT / "examples" / "sar_polar_to_cart.py"
RUNS_DIR = REPO_ROOT / "api_runs"

app = FastAPI(title="torchbp SAR API", version="0.1.0")


def _run_command(cmd: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Processing command failed",
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/process")
async def process_sar(
    file: UploadFile = File(...),
    nsweeps: int = Form(10000),
    fft_oversample: float = Form(1.5),
    dpi: int = Form(700),
    max_side: int | None = Form(None),
) -> FileResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Input filename is missing")
    if not file.filename.lower().endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="Only .safetensors files are supported")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / f"run_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)

    input_path = run_dir / "input.safetensors"
    with input_path.open("wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)

    process_cmd = [
        sys.executable,
        str(PROCESS_SCRIPT),
        str(input_path),
        "--nsweeps",
        str(nsweeps),
        "--fft-oversample",
        str(fft_oversample),
        "--skip-png",
    ]
    _run_command(process_cmd, cwd=run_dir)

    pkl_path = run_dir / "sar_img.p"
    if not pkl_path.exists():
        raise HTTPException(status_code=500, detail="sar_img.p was not generated")

    cart_cmd = [
        sys.executable,
        str(CART_SCRIPT),
        str(pkl_path),
        "--dpi",
        str(dpi),
    ]
    if max_side is not None:
        cart_cmd.extend(["--max-side", str(max_side)])
    _run_command(cart_cmd, cwd=run_dir)

    output_image = run_dir / "sar_img_cart.png"
    if not output_image.exists():
        raise HTTPException(status_code=500, detail="sar_img_cart.png was not generated")

    return FileResponse(
        path=str(output_image),
        media_type="image/png",
        filename="sar_img_cart.png",
        background=BackgroundTask(shutil.rmtree, run_dir, True),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)

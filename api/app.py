import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .job_store import JobStore
from .jobs import prepare_job
from .models import ProcessParams
from .queueing import InlineQueueBackend, RQQueueBackend
from .settings import load_settings
from .worker import run_job


SETTINGS = load_settings()
STORE = JobStore(SETTINGS.jobs_db_path)


def _build_queue_backend():
    if SETTINGS.queue_backend == "rq":
        try:
            return RQQueueBackend(SETTINGS.redis_url)
        except Exception:
            return InlineQueueBackend()
    return InlineQueueBackend()


QUEUE = _build_queue_backend()

app = FastAPI(title="torchbp SAR API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "queue_backend": SETTINGS.queue_backend,
        "storage_backend": SETTINGS.storage_backend,
    }


@app.post("/jobs")
async def submit_job(
    file: UploadFile = File(...),
    nsweeps: int = Form(10000),
    fft_oversample: float = Form(1.5),
    dpi: int = Form(700),
    max_side: int | None = Form(None),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Input filename is missing")
    if not file.filename.lower().endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="Only .safetensors files are supported")
    if nsweeps <= 0:
        raise HTTPException(status_code=400, detail="nsweeps must be > 0")
    if fft_oversample <= 0:
        raise HTTPException(status_code=400, detail="fft_oversample must be > 0")
    if dpi <= 0:
        raise HTTPException(status_code=400, detail="dpi must be > 0")

    payload = await file.read()
    params = ProcessParams(
        nsweeps=nsweeps,
        fft_oversample=fft_oversample,
        dpi=dpi,
        max_side=max_side,
        profile="standard",
    )
    job_id, _, reused = prepare_job(
        filename=file.filename,
        payload=payload,
        params=params,
        settings=SETTINGS,
        store=STORE,
    )
    if not reused:
        task = QUEUE.enqueue(
            run_job,
            job_id,
            store_path=str(SETTINGS.jobs_db_path),
            settings_dict=SETTINGS.to_worker_dict(),
        )
        return {"job_id": job_id, "status": "queued", "task_id": task.external_id}

    existing = STORE.get_job(job_id)
    return {
        "job_id": job_id,
        "status": existing.status if existing else "queued",
        "reused": True,
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = STORE.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    manifest = json.loads(job.result_manifest_json) if job.result_manifest_json else {}
    params = json.loads(job.params_json) if job.params_json else {}
    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "error": job.error_message,
        "profile": job.profile,
        "params": params,
        "result_manifest": manifest,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


@app.get("/jobs/{job_id}/manifest")
def job_manifest(job_id: str) -> dict:
    job = STORE.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(job.result_manifest_json) if job.result_manifest_json else {}


@app.post("/process")
async def process_back_compat(
    file: UploadFile = File(...),
    nsweeps: int = Form(10000),
    fft_oversample: float = Form(1.5),
    dpi: int = Form(700),
    max_side: int | None = Form(None),
) -> dict:
    return await submit_job(
        file=file,
        nsweeps=nsweeps,
        fft_oversample=fft_oversample,
        dpi=dpi,
        max_side=max_side,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)

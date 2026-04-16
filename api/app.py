import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .job_store import JobStore
from .jobs import prepare_job
from .models import ProcessParams
from .pipeline import STATUS_CANCELING, STATUS_QUEUED, STATUS_RUNNING, STATUS_VALIDATING
from .queueing import InlineQueueBackend, RQQueueBackend
from .settings import load_settings
from .validation import validate_safetensors_payload
from .worker import run_job
from torchbp.profiles import normalize_profile


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
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _resolve_local_artifact_path(job_id: str, object_name: str) -> Path:
    if SETTINGS.storage_backend != "local":
        raise HTTPException(status_code=400, detail="Local artifact serving is available only for local storage backend")
    base_dir = (SETTINGS.artifacts_dir / job_id).resolve()
    candidate = (base_dir / object_name).resolve()
    if not str(candidate).startswith(str(base_dir)):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return candidate


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="UI assets are missing")
    return FileResponse(str(index_path), media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


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
    profile: str = Form("standard"),
    write_world_file: bool = Form(False),
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
    try:
        profile_name = normalize_profile(profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = await file.read()
    validation_report = validate_safetensors_payload(payload)
    if not validation_report["ok"]:
        raise HTTPException(status_code=422, detail=validation_report)

    params = ProcessParams(
        nsweeps=nsweeps,
        fft_oversample=fft_oversample,
        dpi=dpi,
        max_side=max_side,
        profile=profile_name,
        write_world_file=bool(write_world_file),
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
        return {"job_id": job_id, "status": STATUS_QUEUED, "task_id": task.external_id}

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
        "stage_progress": job.stage_progress,
        "overall_progress": job.overall_progress,
        "cancel_requested": job.cancel_requested,
        "error": job.error_message,
        "error_class": job.error_class,
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


@app.get("/jobs/{job_id}/artifact")
def job_artifact(job_id: str, object_name: str) -> FileResponse:
    artifact_path = _resolve_local_artifact_path(job_id, object_name)
    return FileResponse(path=str(artifact_path), filename=artifact_path.name)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = STORE.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in {STATUS_QUEUED, STATUS_VALIDATING, STATUS_RUNNING}:
        raise HTTPException(status_code=409, detail=f"Job cannot be canceled from status '{job.status}'")

    STORE.request_cancel(job_id)
    STORE.update_status(job_id, status=STATUS_CANCELING)
    return {"job_id": job_id, "status": STATUS_CANCELING, "cancel_requested": True}


@app.post("/process")
async def process_back_compat(
    file: UploadFile = File(...),
    nsweeps: int = Form(10000),
    fft_oversample: float = Form(1.5),
    dpi: int = Form(700),
    max_side: int | None = Form(None),
    profile: str = Form("standard"),
    write_world_file: bool = Form(False),
) -> dict:
    return await submit_job(
        file=file,
        nsweeps=nsweeps,
        fft_oversample=fft_oversample,
        dpi=dpi,
        max_side=max_side,
        profile=profile,
        write_world_file=write_world_file,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)

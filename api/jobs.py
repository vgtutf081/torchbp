from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

from .job_store import JobStore
from .logging_utils import get_structured_logger
from .models import ProcessParams
from .settings import Settings
from .storage import LocalArtifactStorage, S3ArtifactStorage
from .telemetry import collect_gpu_metrics, stage_timer


PROCESS_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "sar_process_safetensor.py"
CART_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "sar_polar_to_cart.py"
LOGGER = get_structured_logger("torchbp.api.jobs")


def request_hash(filename: str, payload_bytes: bytes, params: ProcessParams) -> str:
    h = hashlib.sha256()
    h.update(filename.encode("utf-8"))
    h.update(payload_bytes)
    h.update(json.dumps(params.to_dict(), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def prepare_job(
    *,
    filename: str,
    payload: bytes,
    params: ProcessParams,
    settings: Settings,
    store: JobStore,
) -> tuple[str, Path, bool]:
    req_hash = request_hash(filename, payload, params)
    existing = store.find_by_request_hash(req_hash)
    if existing and existing.status in {"queued", "running", "success"}:
        return existing.job_id, Path(existing.input_path), True

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"job_{uuid.uuid4().hex}"
    input_path = settings.uploads_dir / f"{job_id}.safetensors"
    input_path.write_bytes(payload)

    store.create_job(
        job_id=job_id,
        input_path=str(input_path),
        request_hash=req_hash,
        profile=params.profile,
        params=params.to_dict(),
    )
    return job_id, input_path, False


def _run_command(cmd: list[str], cwd: Path) -> tuple[str, str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed",
            {
                "command": cmd,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            },
        )
    return completed.stdout, completed.stderr


def run_job(
    job_id: str,
    *,
    store_path: str,
    settings_dict: dict[str, Any],
) -> None:
    settings = Settings(
        repo_root=Path(settings_dict["repo_root"]),
        runs_dir=Path(settings_dict["runs_dir"]),
        uploads_dir=Path(settings_dict["uploads_dir"]),
        artifacts_dir=Path(settings_dict["artifacts_dir"]),
        jobs_db_path=Path(settings_dict["jobs_db_path"]),
        queue_backend=str(settings_dict["queue_backend"]),
        redis_url=str(settings_dict["redis_url"]),
        storage_backend=str(settings_dict["storage_backend"]),
        s3_bucket=str(settings_dict["s3_bucket"]),
        s3_region=str(settings_dict["s3_region"]),
        s3_endpoint_url=settings_dict.get("s3_endpoint_url"),
        s3_access_key_id=settings_dict.get("s3_access_key_id"),
        s3_secret_access_key=settings_dict.get("s3_secret_access_key"),
    )
    store = JobStore(Path(store_path))
    if settings.storage_backend == "s3":
        storage = S3ArtifactStorage(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
        )
    else:
        storage = LocalArtifactStorage(settings.artifacts_dir)

    record = store.get_job(job_id)
    if record is None:
        return

    work_dir = settings.runs_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(record.input_path)

    params = json.loads(record.params_json)
    stage_metrics: dict[str, dict[str, Any]] = {}
    LOGGER.info("job_started", extra={"job_id": job_id, "stage": "queued"})
    try:
        with stage_timer() as timer:
            store.update_status(job_id, status="running", stage="backprojection", progress=0.05)
            process_cmd = [
                sys.executable,
                str(PROCESS_SCRIPT),
                str(input_path),
                "--nsweeps",
                str(int(params["nsweeps"])),
                "--fft-oversample",
                str(float(params["fft_oversample"])),
                "--skip-png",
                "--profile",
                str(params.get("profile", "standard")),
            ]
            _run_command(process_cmd, work_dir)
        stage_metrics["backprojection"] = {
            "duration_ms": timer.get("duration_ms", 0.0),
            **collect_gpu_metrics(),
        }
        LOGGER.info(
            "stage_completed",
            extra={
                "job_id": job_id,
                "stage": "backprojection",
                "duration_ms": stage_metrics["backprojection"]["duration_ms"],
                "extra_fields": stage_metrics["backprojection"],
            },
        )

        pkl_path = work_dir / "sar_img.p"
        if not pkl_path.exists():
            raise RuntimeError("sar_img.p was not generated")

        with stage_timer() as timer:
            store.update_status(job_id, stage="export", progress=0.6)
            cart_cmd = [
                sys.executable,
                str(CART_SCRIPT),
                str(pkl_path),
                "--dpi",
                str(int(params["dpi"])),
                "--profile",
                str(params.get("profile", "standard")),
                "--output-prefix",
                str(params.get("output_prefix", "sar_img")),
            ]
            max_side = params.get("max_side")
            if max_side is not None:
                cart_cmd.extend(["--max-side", str(int(max_side))])
            _run_command(cart_cmd, work_dir)
        stage_metrics["export"] = {
            "duration_ms": timer.get("duration_ms", 0.0),
            **collect_gpu_metrics(),
        }
        LOGGER.info(
            "stage_completed",
            extra={
                "job_id": job_id,
                "stage": "export",
                "duration_ms": stage_metrics["export"]["duration_ms"],
                "extra_fields": stage_metrics["export"],
            },
        )

        manifest: dict[str, Any] = {"job_id": job_id, "files": [], "metrics": stage_metrics}
        for file_name in [
            "sar_img.p",
            f"{params.get('output_prefix', 'sar_img')}_cart.png",
            f"{params.get('output_prefix', 'sar_img')}.tif",
            f"{params.get('output_prefix', 'sar_img')}_cart.pgw",
        ]:
            local_file = work_dir / file_name
            if local_file.exists():
                uri = storage.store_file(job_id=job_id, local_path=local_file, object_name=file_name)
                manifest["files"].append({"name": file_name, "uri": uri})

        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        uri = storage.store_file(job_id=job_id, local_path=manifest_path, object_name="manifest.json")
        manifest["manifest_uri"] = uri

        store.set_result_manifest(job_id, manifest)
        store.update_status(job_id, status="success", stage="done", progress=1.0)
        LOGGER.info("job_finished", extra={"job_id": job_id, "stage": "done"})
    except Exception as exc:
        error_trace = traceback.format_exc()
        LOGGER.exception(
            "job_failed",
            extra={
                "job_id": job_id,
                "stage": "failed",
                "extra_fields": {"traceback": error_trace},
            },
        )
        store.update_status(job_id, status="failed", stage="failed", progress=1.0, error_message=str(exc))
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

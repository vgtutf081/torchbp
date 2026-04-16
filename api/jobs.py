from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .job_store import JobStore
from .logging_utils import get_structured_logger
from .models import ProcessParams
from .pipeline import (
    FingerprintContext,
    STATUS_CANCELED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_VALIDATING,
    STAGE_WEIGHTS,
    classify_error,
    input_fingerprint,
    overall_progress,
    processing_fingerprint,
    request_fingerprint,
)
from .settings import Settings
from .storage import LocalArtifactStorage, S3ArtifactStorage
from .telemetry import collect_gpu_metrics, stage_timer


PROCESS_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "sar_process_safetensor.py"
CART_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "sar_polar_to_cart.py"
LOGGER = get_structured_logger("torchbp.api.jobs")


def request_hash(filename: str, payload_bytes: bytes, params: ProcessParams, settings: Settings) -> str:
    input_fp = input_fingerprint(filename, payload_bytes)
    processing_fp = processing_fingerprint(
        params.to_dict(),
        FingerprintContext(
            pipeline_version=settings.pipeline_version,
            algorithm_version=settings.algorithm_version,
            profile_version=settings.profile_version,
            schema_version=settings.schema_version,
            calibration_version=settings.calibration_version,
            dem_version=settings.dem_version,
        ),
    )
    return request_fingerprint(input_fp, processing_fp)


def prepare_job(
    *,
    filename: str,
    payload: bytes,
    params: ProcessParams,
    settings: Settings,
    store: JobStore,
) -> tuple[str, Path, bool]:
    req_hash = request_hash(filename, payload, params, settings)
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_stage_progress(
    store: JobStore,
    job_id: str,
    *,
    stage: str,
    stage_progress: float,
    status: str = STATUS_RUNNING,
) -> None:
    store.update_status(
        job_id,
        status=status,
        stage=stage,
        stage_progress=stage_progress,
        overall_progress=overall_progress(stage, stage_progress),
    )


def _is_cancel_requested(store: JobStore, job_id: str) -> bool:
    record = store.get_job(job_id)
    return bool(record and record.cancel_requested)


def _run_export_with_retry(cmd: list[str], cwd: Path) -> tuple[str, str, bool]:
    try:
        stdout, stderr = _run_command(cmd, cwd)
        return stdout, stderr, False
    except Exception as exc:
        error_class, retryable = classify_error(exc)
        if not retryable or error_class != "resource_error" or "--max-side" in cmd:
            raise

        retry_cmd = [*cmd, "--max-side", "2048"]
        stdout, stderr = _run_command(retry_cmd, cwd)
        return stdout, stderr, True


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
        pipeline_version=str(settings_dict["pipeline_version"]),
        algorithm_version=str(settings_dict["algorithm_version"]),
        profile_version=str(settings_dict["profile_version"]),
        schema_version=str(settings_dict["schema_version"]),
        calibration_version=settings_dict.get("calibration_version"),
        dem_version=settings_dict.get("dem_version"),
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
    logs_dir = work_dir / "logs"
    metrics_dir = work_dir / "metrics"
    intermediates_dir = work_dir / "intermediates"
    outputs_dir = work_dir / "outputs"
    previews_dir = work_dir / "previews"
    debug_dir = work_dir / "debug"
    for directory in [logs_dir, metrics_dir, intermediates_dir, outputs_dir, previews_dir, debug_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    request_path = work_dir / "request.json"
    request_path.write_text(json.dumps({"job_id": job_id, "params": params}, indent=2), encoding="utf-8")

    normalized_request_path = work_dir / "normalized_request.json"
    normalized_request_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "profile": params.get("profile", "standard"),
                "processing_parameters": params,
                "pipeline_version": settings.pipeline_version,
                "algorithm_version": settings.algorithm_version,
                "profile_version": settings.profile_version,
                "schema_version": settings.schema_version,
                "calibration_version": settings.calibration_version,
                "dem_version": settings.dem_version,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    LOGGER.info("job_started", extra={"job_id": job_id, "stage": "queued"})
    try:
        if _is_cancel_requested(store, job_id):
            _update_stage_progress(store, job_id, stage="ingest", stage_progress=0.0, status=STATUS_CANCELED)
            return

        _update_stage_progress(store, job_id, stage="ingest", stage_progress=100.0)
        _update_stage_progress(
            store,
            job_id,
            stage="validation",
            stage_progress=100.0,
            status=STATUS_VALIDATING,
        )

        if _is_cancel_requested(store, job_id):
            _update_stage_progress(store, job_id, stage="validation", stage_progress=100.0, status=STATUS_CANCELED)
            return

        _update_stage_progress(store, job_id, stage="range_compression", stage_progress=25.0)
        with stage_timer() as timer:
            _update_stage_progress(store, job_id, stage="backprojection", stage_progress=5.0)
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
                "--algorithm",
                str(params.get("algorithm", "backprojection")),
            ]
            process_stdout, process_stderr = _run_command(process_cmd, work_dir)
            (logs_dir / "backprojection.stdout.log").write_text(process_stdout, encoding="utf-8")
            (logs_dir / "backprojection.stderr.log").write_text(process_stderr, encoding="utf-8")
        _update_stage_progress(store, job_id, stage="backprojection", stage_progress=100.0)
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
        pkl_intermediate = intermediates_dir / "sar_img.p"
        shutil.copy2(pkl_path, pkl_intermediate)

        if _is_cancel_requested(store, job_id):
            _update_stage_progress(store, job_id, stage="autofocus", stage_progress=0.0, status=STATUS_CANCELED)
            return

        _update_stage_progress(store, job_id, stage="autofocus", stage_progress=100.0)

        with stage_timer() as timer:
            _update_stage_progress(store, job_id, stage="export", stage_progress=5.0)
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
            if bool(params.get("write_world_file", False)):
                cart_cmd.append("--write-world-file")

            cart_stdout, cart_stderr, retried = _run_export_with_retry(cart_cmd, work_dir)
            (logs_dir / "export.stdout.log").write_text(cart_stdout, encoding="utf-8")
            (logs_dir / "export.stderr.log").write_text(cart_stderr, encoding="utf-8")
            if retried:
                (debug_dir / "export_retry.txt").write_text(
                    "resource_error detected; export retried with --max-side 2048\n",
                    encoding="utf-8",
                )
        _update_stage_progress(store, job_id, stage="export", stage_progress=100.0)
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

        file_targets = [
            ("request.json", request_path, "debug"),
            ("normalized_request.json", normalized_request_path, "debug"),
            ("intermediates/sar_img.p", pkl_intermediate, "intermediates"),
            (
                f"previews/{params.get('output_prefix', 'sar_img')}_cart.png",
                work_dir / f"{params.get('output_prefix', 'sar_img')}_cart.png",
                "previews",
            ),
            (
                f"outputs/{params.get('output_prefix', 'sar_img')}.tif",
                work_dir / f"{params.get('output_prefix', 'sar_img')}.tif",
                "outputs",
            ),
            (
                f"outputs/{params.get('output_prefix', 'sar_img')}_cart.pgw",
                work_dir / f"{params.get('output_prefix', 'sar_img')}_cart.pgw",
                "outputs",
            ),
            ("logs/backprojection.stdout.log", logs_dir / "backprojection.stdout.log", "logs"),
            ("logs/backprojection.stderr.log", logs_dir / "backprojection.stderr.log", "logs"),
            ("logs/export.stdout.log", logs_dir / "export.stdout.log", "logs"),
            ("logs/export.stderr.log", logs_dir / "export.stderr.log", "logs"),
        ]

        manifest: dict[str, Any] = {
            "job_id": job_id,
            "pipeline_version": settings.pipeline_version,
            "algorithm_version": settings.algorithm_version,
            "schema_version": settings.schema_version,
            "profile": params.get("profile", "standard"),
            "processing_parameters": params,
            "stage_weights": STAGE_WEIGHTS,
            "timings": stage_metrics,
            "warnings": [],
            "input_artifacts": [],
            "output_artifacts": [],
            "quality_metrics": {
                "valid_pixels_pct": None,
                "estimated_snr_db": None,
                "entropy": None,
                "autofocus_improvement": None,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        input_uri = storage.store_file(job_id=job_id, local_path=input_path, object_name="inputs/input.safetensors")
        manifest["input_artifacts"].append(
            {
                "name": "input.safetensors",
                "object_name": "inputs/input.safetensors",
                "uri": input_uri,
                "sha256": _sha256_file(input_path),
            }
        )

        for object_name, local_file, kind in file_targets:
            if local_file.exists():
                uri = storage.store_file(job_id=job_id, local_path=local_file, object_name=object_name)
                manifest["output_artifacts"].append(
                    {
                        "name": local_file.name,
                        "kind": kind,
                        "object_name": object_name,
                        "uri": uri,
                        "sha256": _sha256_file(local_file),
                    }
                )

        metrics_path = metrics_dir / "timings.json"
        metrics_path.write_text(json.dumps(stage_metrics, indent=2), encoding="utf-8")
        uri = storage.store_file(job_id=job_id, local_path=metrics_path, object_name="metrics/timings.json")
        manifest["output_artifacts"].append(
            {
                "name": "timings.json",
                "kind": "metrics",
                "object_name": "metrics/timings.json",
                "uri": uri,
                "sha256": _sha256_file(metrics_path),
            }
        )

        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        uri = storage.store_file(job_id=job_id, local_path=manifest_path, object_name="manifest.json")
        manifest["manifest_uri"] = uri

        store.set_result_manifest(job_id, manifest)
        store.update_status(
            job_id,
            status=STATUS_SUCCESS,
            stage="export",
            stage_progress=100.0,
            overall_progress=100.0,
        )
        LOGGER.info("job_finished", extra={"job_id": job_id, "stage": "done"})
    except Exception as exc:
        if _is_cancel_requested(store, job_id):
            store.update_status(
                job_id,
                status=STATUS_CANCELED,
                stage="canceling",
                stage_progress=100.0,
                overall_progress=100.0,
                error_message="Canceled by user",
            )
            return

        error_trace = traceback.format_exc()
        error_class, retryable = classify_error(exc)
        LOGGER.exception(
            "job_failed",
            extra={
                "job_id": job_id,
                "stage": "failed",
                "extra_fields": {"traceback": error_trace, "error_class": error_class, "retryable": retryable},
            },
        )
        store.update_status(
            job_id,
            status=STATUS_FAILED,
            stage="failed",
            stage_progress=100.0,
            overall_progress=100.0,
            error_class=error_class,
            error_message=str(exc),
        )
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

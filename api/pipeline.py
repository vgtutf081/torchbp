from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


STATUS_QUEUED = "queued"
STATUS_VALIDATING = "validating"
STATUS_RUNNING = "running"
STATUS_CANCELING = "canceling"
STATUS_CANCELED = "canceled"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

STAGE_ORDER = [
    "ingest",
    "validation",
    "range_compression",
    "backprojection",
    "autofocus",
    "export",
]

STAGE_WEIGHTS = {
    "ingest": 5.0,
    "validation": 5.0,
    "range_compression": 15.0,
    "backprojection": 35.0,
    "autofocus": 30.0,
    "export": 10.0,
}


@dataclass(frozen=True)
class FingerprintContext:
    pipeline_version: str
    algorithm_version: str
    profile_version: str
    schema_version: str
    calibration_version: str | None
    dem_version: str | None


def input_fingerprint(filename: str, payload_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(filename.encode("utf-8"))
    h.update(payload_bytes)
    return h.hexdigest()


def processing_fingerprint(params: dict[str, Any], context: FingerprintContext) -> str:
    payload = {
        "params": params,
        "pipeline_version": context.pipeline_version,
        "algorithm_version": context.algorithm_version,
        "profile_version": context.profile_version,
        "schema_version": context.schema_version,
        "calibration_version": context.calibration_version,
        "dem_version": context.dem_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def request_fingerprint(input_fp: str, processing_fp: str) -> str:
    return hashlib.sha256(f"{input_fp}:{processing_fp}".encode("utf-8")).hexdigest()


def overall_progress(stage: str, stage_progress: float) -> float:
    if stage not in STAGE_WEIGHTS:
        return min(max(stage_progress, 0.0), 100.0)

    done = 0.0
    for item in STAGE_ORDER:
        if item == stage:
            break
        done += STAGE_WEIGHTS.get(item, 0.0)

    weight = STAGE_WEIGHTS.get(stage, 0.0)
    sp = min(max(stage_progress, 0.0), 100.0)
    return min(done + weight * (sp / 100.0), 100.0)


def classify_error(error: Exception) -> tuple[str, bool]:
    message = str(error).lower()
    if "validation" in message or "invalid" in message:
        return "validation_error", False
    if "timed out" in message or "connection" in message or "temporary" in message:
        return "transient_infra_error", True
    if "oom" in message or "out of memory" in message or "resource" in message:
        return "resource_error", True
    return "algorithm_error", False

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from safetensors.torch import safe_open


REQUIRED_MISSION_FIELDS = {
    "fsample",
    "fc",
    "bw",
    "origin_angle",
    "pri",
}


def validate_mission_metadata(metadata: dict[str, str | float | int]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_MISSION_FIELDS:
        if key not in metadata:
            errors.append(f"missing metadata field: {key}")

    for key in ["fsample", "fc", "bw", "pri"]:
        if key in metadata:
            try:
                value = float(metadata[key])
                if value <= 0:
                    errors.append(f"metadata {key} must be > 0")
            except Exception:
                errors.append(f"metadata {key} must be numeric")

    return errors


def validate_trajectory(pos: torch.Tensor, att: torch.Tensor, counts: torch.Tensor) -> list[str]:
    errors: list[str] = []

    if pos.ndim != 2 or pos.shape[1] != 3:
        errors.append("pos must have shape [N, 3]")
    if att.ndim != 2 or att.shape[1] != 3:
        errors.append("att must have shape [N, 3]")
    if counts.ndim != 1:
        errors.append("counts must have shape [N]")

    n = pos.shape[0]
    if att.shape[0] != n or counts.shape[0] != n:
        errors.append("pos, att, counts must have matching N")
        return errors

    if n < 3:
        errors.append("trajectory length must be at least 3")
        return errors

    if not torch.isfinite(pos).all().item():
        errors.append("pos contains non-finite values")
    if not torch.isfinite(att).all().item():
        errors.append("att contains non-finite values")
    if not torch.isfinite(counts).all().item():
        errors.append("counts contains non-finite values")

    dcounts = torch.diff(counts)
    if torch.any(dcounts <= 0).item():
        errors.append("counts must be strictly increasing")

    dt = torch.median(dcounts).item()
    if dt <= 0:
        errors.append("counts delta must be positive")
        return errors

    velocity = torch.diff(pos, dim=0) / dt
    speed = torch.linalg.norm(velocity, dim=1)
    if torch.max(speed).item() > 1000.0:
        errors.append("trajectory speed sanity check failed (>1000 m/s)")

    if velocity.shape[0] >= 2:
        accel = torch.diff(velocity, dim=0) / dt
        accel_norm = torch.linalg.norm(accel, dim=1)
        if torch.max(accel_norm).item() > 200.0:
            errors.append("trajectory acceleration sanity check failed (>200 m/s^2)")

    return errors


def validate_signal(
    data: torch.Tensor,
    *,
    pos: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if data.ndim < 2:
        errors.append("data must have at least 2 dimensions [N, ..., samples]")
        return errors, warnings

    nsweeps = int(data.shape[0])
    nsamples = int(data.shape[-1])
    if nsweeps < 64:
        errors.append("minimum number of sweeps is 64")
    if nsamples < 64:
        errors.append("minimum number of samples per sweep is 64")

    if pos.shape[0] != nsweeps or counts.shape[0] != nsweeps:
        errors.append("inconsistent sizes: data/pos/counts first dimension must match")

    if not torch.isfinite(data).all().item():
        errors.append("data contains non-finite values")
        return errors, warnings

    sample_view = data
    if data.ndim > 2:
        sample_view = data[:, 0, :]

    max_abs = torch.max(torch.abs(sample_view)).item()
    if max_abs > 0:
        sat_ratio = float((torch.abs(sample_view) >= (0.99 * max_abs)).float().mean().item())
        if sat_ratio > 0.05:
            warnings.append(f"high saturated sample ratio: {sat_ratio:.3f}")

    mean_abs = float(torch.mean(torch.abs(sample_view)).item())
    std_abs = float(torch.std(torch.abs(sample_view)).item())
    if std_abs > 1e-9:
        dc_bias_ratio = mean_abs / std_abs
        if dc_bias_ratio > 5.0:
            warnings.append(f"possible DC bias/leakage detected: ratio={dc_bias_ratio:.2f}")

    if pos.shape[0] >= 2:
        path_len = float(torch.sum(torch.linalg.norm(torch.diff(pos, dim=0), dim=1)).item())
        if path_len < 5.0:
            warnings.append("trajectory coverage is very small (<5m), image grid coverage may be poor")

    return errors, warnings


def validate_safetensors_file(path: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for required in ["data", "pos", "att", "counts"]:
                if required not in keys:
                    errors.append(f"missing tensor: {required}")

            metadata = f.metadata() or {}
            errors.extend(validate_mission_metadata(metadata))

            if not errors:
                pos = f.get_tensor("pos")
                att = f.get_tensor("att")
                counts = f.get_tensor("counts")
                data = f.get_tensor("data")
                errors.extend(validate_trajectory(pos, att, counts))
                signal_errors, signal_warnings = validate_signal(data, pos=pos, counts=counts)
                errors.extend(signal_errors)
                warnings.extend(signal_warnings)
    except Exception as exc:
        errors.append(f"unable to read safetensors payload: {exc}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_safetensors_payload(payload: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as temp_file:
        temp_file.write(payload)
        temp_path = Path(temp_file.name)

    try:
        return validate_safetensors_file(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

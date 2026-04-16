from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def collect_gpu_metrics() -> dict[str, float | int | bool | None]:
    try:
        import torch
    except Exception:
        return {
            "gpu_available": False,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
            "gpu_max_memory_allocated_mb": None,
        }

    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
            "gpu_max_memory_allocated_mb": None,
        }

    device = torch.cuda.current_device()
    return {
        "gpu_available": True,
        "gpu_device": int(device),
        "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024 * 1024)),
        "gpu_memory_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024 * 1024)),
        "gpu_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
    }


@contextmanager
def stage_timer() -> Iterator[dict[str, float]]:
    timings = {"started_at": time.perf_counter()}
    try:
        yield timings
    finally:
        elapsed = (time.perf_counter() - timings["started_at"]) * 1000.0
        timings["duration_ms"] = float(elapsed)

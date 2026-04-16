from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class JobRecord:
    job_id: str
    status: str
    stage: str
    stage_progress: float
    overall_progress: float
    created_at: datetime
    updated_at: datetime
    input_path: str
    request_hash: str
    profile: str
    params_json: str
    result_manifest_json: str
    error_class: str | None = None
    cancel_requested: bool = False
    error_message: str | None = None


@dataclass
class ProcessParams:
    nsweeps: int
    fft_oversample: float
    dpi: int
    max_side: int | None
    profile: str = "standard"
    output_prefix: str = "sar_img"
    write_world_file: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "nsweeps": self.nsweeps,
            "fft_oversample": self.fft_oversample,
            "dpi": self.dpi,
            "max_side": self.max_side,
            "profile": self.profile,
            "output_prefix": self.output_prefix,
            "write_world_file": self.write_world_file,
        }

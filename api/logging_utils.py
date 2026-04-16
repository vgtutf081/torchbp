from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JobJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "job_id"):
            payload["job_id"] = getattr(record, "job_id")
        if hasattr(record, "stage"):
            payload["stage"] = getattr(record, "stage")
        if hasattr(record, "duration_ms"):
            payload["duration_ms"] = getattr(record, "duration_ms")
        if hasattr(record, "extra_fields"):
            payload.update(getattr(record, "extra_fields"))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_structured_logger(name: str = "torchbp.api") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JobJSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger

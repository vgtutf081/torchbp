from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .models import JobRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    result_manifest_json TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_request_hash ON jobs(request_hash)"
            )
            conn.commit()

    def create_job(
        self,
        *,
        job_id: str,
        input_path: str,
        request_hash: str,
        profile: str,
        params: dict,
    ) -> None:
        now = _utc_now_iso()
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, stage, progress, created_at, updated_at,
                    input_path, request_hash, profile, params_json, result_manifest_json, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "queued",
                    "queued",
                    0.0,
                    now,
                    now,
                    input_path,
                    request_hash,
                    profile,
                    json.dumps(params, sort_keys=True),
                    json.dumps({}, sort_keys=True),
                    None,
                ),
            )
            conn.commit()

    def find_by_request_hash(self, request_hash: str) -> JobRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE request_hash = ? ORDER BY created_at DESC LIMIT 1",
                (request_hash,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def update_status(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        error_message: str | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[object] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if stage is not None:
            fields.append("stage = ?")
            values.append(stage)
        if progress is not None:
            fields.append("progress = ?")
            values.append(float(progress))
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)

        fields.append("updated_at = ?")
        values.append(_utc_now_iso())
        values.append(job_id)
        with closing(self._connect()) as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?",
                tuple(values),
            )
            conn.commit()

    def set_result_manifest(self, job_id: str, manifest: dict) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE jobs
                SET result_manifest_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(manifest, sort_keys=True), _utc_now_iso(), job_id),
            )
            conn.commit()

    def get_job(self, job_id: str) -> JobRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            stage=row["stage"],
            progress=float(row["progress"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            input_path=row["input_path"],
            request_hash=row["request_hash"],
            profile=row["profile"],
            params_json=row["params_json"],
            result_manifest_json=row["result_manifest_json"],
            error_message=row["error_message"],
        )

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    runs_dir: Path
    uploads_dir: Path
    artifacts_dir: Path
    jobs_db_path: Path
    queue_backend: str
    redis_url: str
    storage_backend: str
    s3_bucket: str
    s3_region: str
    s3_endpoint_url: str | None
    s3_access_key_id: str | None
    s3_secret_access_key: str | None

    def to_worker_dict(self) -> dict:
        return {
            "repo_root": str(self.repo_root),
            "runs_dir": str(self.runs_dir),
            "uploads_dir": str(self.uploads_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "jobs_db_path": str(self.jobs_db_path),
            "queue_backend": self.queue_backend,
            "redis_url": self.redis_url,
            "storage_backend": self.storage_backend,
            "s3_bucket": self.s3_bucket,
            "s3_region": self.s3_region,
            "s3_endpoint_url": self.s3_endpoint_url,
            "s3_access_key_id": self.s3_access_key_id,
            "s3_secret_access_key": self.s3_secret_access_key,
        }


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "api_runs"
    uploads_dir = runs_dir / "uploads"
    artifacts_dir = runs_dir / "artifacts"
    jobs_db_path = runs_dir / "jobs.db"

    queue_backend = os.getenv("TORCHBP_QUEUE_BACKEND", "rq").strip().lower()
    if queue_backend not in {"rq", "inline"}:
        queue_backend = "rq"

    storage_backend = os.getenv("TORCHBP_STORAGE_BACKEND", "local").strip().lower()
    if storage_backend not in {"local", "s3"}:
        storage_backend = "local"

    return Settings(
        repo_root=repo_root,
        runs_dir=runs_dir,
        uploads_dir=uploads_dir,
        artifacts_dir=artifacts_dir,
        jobs_db_path=jobs_db_path,
        queue_backend=queue_backend,
        redis_url=os.getenv("TORCHBP_REDIS_URL", "redis://127.0.0.1:6379/0"),
        storage_backend=storage_backend,
        s3_bucket=os.getenv("TORCHBP_S3_BUCKET", "torchbp-results"),
        s3_region=os.getenv("TORCHBP_S3_REGION", "us-east-1"),
        s3_endpoint_url=os.getenv("TORCHBP_S3_ENDPOINT_URL") or None,
        s3_access_key_id=os.getenv("TORCHBP_S3_ACCESS_KEY_ID") or None,
        s3_secret_access_key=os.getenv("TORCHBP_S3_SECRET_ACCESS_KEY") or None,
    )

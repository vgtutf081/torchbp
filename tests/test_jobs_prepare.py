import tempfile
import unittest
from pathlib import Path

from api.job_store import JobStore
from api.jobs import prepare_job
from api.models import ProcessParams
from api.settings import Settings


class TestPrepareJob(unittest.TestCase):
    def test_prepare_job_deduplicates_by_request_hash(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                repo_root=root,
                runs_dir=root / "runs",
                uploads_dir=root / "uploads",
                artifacts_dir=root / "artifacts",
                jobs_db_path=root / "jobs.db",
                queue_backend="inline",
                redis_url="redis://127.0.0.1:6379/0",
                storage_backend="local",
                s3_bucket="bucket",
                s3_region="us-east-1",
                s3_endpoint_url=None,
                s3_access_key_id=None,
                s3_secret_access_key=None,
            )
            store = JobStore(settings.jobs_db_path)
            params = ProcessParams(nsweeps=100, fft_oversample=1.5, dpi=300, max_side=1024)
            payload = b"dummy-safetensors-data"

            job_id1, _, reused1 = prepare_job(
                filename="input.safetensors",
                payload=payload,
                params=params,
                settings=settings,
                store=store,
            )
            self.assertFalse(reused1)

            store.update_status(job_id1, status="success", stage="done", progress=1.0)

            job_id2, _, reused2 = prepare_job(
                filename="input.safetensors",
                payload=payload,
                params=params,
                settings=settings,
                store=store,
            )
            self.assertTrue(reused2)
            self.assertEqual(job_id1, job_id2)


if __name__ == "__main__":
    unittest.main()

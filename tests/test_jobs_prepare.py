import tempfile
import unittest
from pathlib import Path

from api.job_store import JobStore
from api.jobs import prepare_job, request_hash
from api.models import ProcessParams
from api.settings import Settings


class TestPrepareJob(unittest.TestCase):
    def _make_settings(self, root: Path) -> Settings:
        return Settings(
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
            pipeline_version="1.0.0",
            algorithm_version="1.0.0",
            profile_version="1",
            schema_version="1",
            calibration_version=None,
            dem_version=None,
        )

    def test_prepare_job_deduplicates_by_request_hash(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            settings = self._make_settings(root)
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

            store.update_status(
                job_id1,
                status="success",
                stage="done",
                stage_progress=100.0,
                overall_progress=100.0,
            )

            job_id2, _, reused2 = prepare_job(
                filename="input.safetensors",
                payload=payload,
                params=params,
                settings=settings,
                store=store,
            )
            self.assertTrue(reused2)
            self.assertEqual(job_id1, job_id2)

    def test_request_hash_changes_with_pipeline_version(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            settings_v1 = self._make_settings(root)
            settings_v2 = Settings(
                **{**settings_v1.__dict__, "pipeline_version": "2.0.0"}
            )
            params = ProcessParams(nsweeps=100, fft_oversample=1.5, dpi=300, max_side=1024)
            payload = b"dummy-safetensors-data"

            h1 = request_hash("input.safetensors", payload, params, settings_v1)
            h2 = request_hash("input.safetensors", payload, params, settings_v2)

            self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()

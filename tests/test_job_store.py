import tempfile
import unittest
from pathlib import Path

from api.job_store import JobStore


class TestJobStore(unittest.TestCase):
    def test_create_update_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            db_path = Path(temp_dir) / "jobs.db"
            store = JobStore(db_path)

            store.create_job(
                job_id="job_1",
                input_path="/tmp/input.safetensors",
                request_hash="abc",
                profile="standard",
                params={"nsweeps": 10},
            )
            job = store.get_job("job_1")
            self.assertIsNotNone(job)
            self.assertEqual(job.status, "queued")
            self.assertEqual(job.stage, "queued")

            store.update_status("job_1", status="running", stage="backprojection", progress=0.5)
            job = store.get_job("job_1")
            self.assertIsNotNone(job)
            self.assertEqual(job.status, "running")
            self.assertEqual(job.stage, "backprojection")
            self.assertAlmostEqual(job.progress, 0.5)

            store.set_result_manifest("job_1", {"files": [{"name": "sar_img.p"}]})
            job = store.get_job("job_1")
            self.assertIsNotNone(job)
            self.assertIn("sar_img.p", job.result_manifest_json)


if __name__ == "__main__":
    unittest.main()

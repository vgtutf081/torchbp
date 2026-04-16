import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app
from api.settings import Settings


class TestApiUi(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_ui_route_serves_html(self) -> None:
        response = self.client.get("/ui")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Torchbp SAR Processor", response.text)

    def test_favicon_route(self) -> None:
        response = self.client.get("/favicon.ico")
        self.assertEqual(response.status_code, 204)

    def test_root_redirects_to_ui(self) -> None:
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers.get("location"), "/ui")

    def test_local_artifact_download(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            base = Path(temp_dir)
            artifacts = base / "artifacts"
            artifact_file = artifacts / "job_123" / "previews" / "result.png"
            artifact_file.parent.mkdir(parents=True, exist_ok=True)
            artifact_file.write_bytes(b"PNG")

            current = app.dependency_overrides
            self.assertIsNotNone(current)

            patched_settings = Settings(
                repo_root=base,
                runs_dir=base / "runs",
                uploads_dir=base / "uploads",
                artifacts_dir=artifacts,
                jobs_db_path=base / "jobs.db",
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

            with patch("api.app.SETTINGS", patched_settings):
                response = self.client.get(
                    "/jobs/job_123/artifact",
                    params={"object_name": "previews/result.png"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b"PNG")


if __name__ == "__main__":
    unittest.main()

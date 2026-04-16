import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app


class TestApiJobs(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("api.app.prepare_job")
    @patch("api.app.QUEUE")
    def test_submit_job_queues_task(self, mock_queue, mock_prepare_job) -> None:
        mock_prepare_job.return_value = ("job_123", None, False)
        mock_queue.enqueue.return_value.external_id = "task_1"

        files = {"file": ("sample.safetensors", b"abc", "application/octet-stream")}
        response = self.client.post("/jobs", files=files)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["job_id"], "job_123")
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["task_id"], "task_1")


if __name__ == "__main__":
    unittest.main()

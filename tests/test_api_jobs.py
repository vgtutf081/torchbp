import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app


class TestApiJobs(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("api.app.prepare_job")
    @patch("api.app.QUEUE")
    @patch("api.app.validate_safetensors_payload")
    def test_submit_job_queues_task(self, mock_validation, mock_queue, mock_prepare_job) -> None:
        mock_validation.return_value = {"ok": True, "errors": [], "warnings": []}
        mock_prepare_job.return_value = ("job_123", None, False)
        mock_queue.enqueue.return_value.external_id = "task_1"

        files = {"file": ("sample.safetensors", b"abc", "application/octet-stream")}
        response = self.client.post("/jobs", files=files)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["job_id"], "job_123")
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["task_id"], "task_1")

    @patch("api.app.prepare_job")
    def test_submit_job_rejects_invalid_payload(self, mock_prepare_job) -> None:
        files = {"file": ("sample.safetensors", b"not-a-safetensors", "application/octet-stream")}
        response = self.client.post("/jobs", files=files)

        self.assertEqual(response.status_code, 422)
        mock_prepare_job.assert_not_called()

    @patch("api.app.STORE")
    def test_cancel_job_sets_canceling_status(self, mock_store) -> None:
        class _Job:
            status = "running"

        mock_store.get_job.return_value = _Job()

        response = self.client.post("/jobs/job_123/cancel")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "canceling")
        mock_store.request_cancel.assert_called_once_with("job_123")


if __name__ == "__main__":
    unittest.main()

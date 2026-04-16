import json
import logging
import unittest

from api.logging_utils import JobJSONFormatter
from api.telemetry import collect_gpu_metrics, stage_timer


class TestTelemetryAndLogging(unittest.TestCase):
    def test_json_formatter_includes_job_fields(self) -> None:
        formatter = JobJSONFormatter()
        record = logging.LogRecord(
            name="torchbp.api",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="stage completed",
            args=(),
            exc_info=None,
        )
        record.job_id = "job_123"
        record.stage = "backprojection"
        record.duration_ms = 12.3
        line = formatter.format(record)
        payload = json.loads(line)

        self.assertEqual(payload["job_id"], "job_123")
        self.assertEqual(payload["stage"], "backprojection")
        self.assertAlmostEqual(payload["duration_ms"], 12.3)

    def test_stage_timer_records_duration(self) -> None:
        with stage_timer() as metrics:
            value = 1 + 1
            self.assertEqual(value, 2)
        self.assertIn("duration_ms", metrics)
        self.assertGreaterEqual(metrics["duration_ms"], 0.0)

    def test_collect_gpu_metrics_shape(self) -> None:
        metrics = collect_gpu_metrics()
        self.assertIn("gpu_available", metrics)
        self.assertIn("gpu_memory_allocated_mb", metrics)


if __name__ == "__main__":
    unittest.main()

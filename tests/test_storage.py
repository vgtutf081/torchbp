import tempfile
import unittest
from pathlib import Path

from api.storage import LocalArtifactStorage


class TestStorage(unittest.TestCase):
    def test_local_storage_supports_nested_object_name(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            local_path = root / "sample.txt"
            local_path.write_text("hello", encoding="utf-8")

            storage = LocalArtifactStorage(root / "artifacts")
            uri = storage.store_file(
                job_id="job_1",
                local_path=local_path,
                object_name="logs/export/stdout.log",
            )

            self.assertTrue(uri.endswith("/job_1/logs/export/stdout.log"))
            saved = root / "artifacts" / "job_1" / "logs" / "export" / "stdout.log"
            self.assertTrue(saved.exists())
            self.assertEqual(saved.read_text(encoding="utf-8"), "hello")


if __name__ == "__main__":
    unittest.main()

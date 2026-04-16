import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from api.validation import validate_mission_metadata, validate_safetensors_file, validate_trajectory


class TestValidation(unittest.TestCase):
    def test_mission_metadata_missing_fields(self) -> None:
        errors = validate_mission_metadata({"fc": "1.0"})
        self.assertTrue(any("missing metadata field" in e for e in errors))

    def test_trajectory_detects_non_monotonic_counts(self) -> None:
        pos = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
        att = torch.zeros((3, 3), dtype=torch.float32)
        counts = torch.tensor([0.0, 1.0, 1.0])

        errors = validate_trajectory(pos, att, counts)
        self.assertTrue(any("strictly increasing" in e for e in errors))

    def test_validate_safetensors_file_ok(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            file_path = Path(temp_dir) / "sample.safetensors"
            tensors = {
                "data": torch.randn(64, 1, 128),
                "pos": torch.tensor(
                    [[float(i), 0.0, 1.0] for i in range(64)],
                    dtype=torch.float32,
                ),
                "att": torch.zeros((64, 3), dtype=torch.float32),
                "counts": torch.arange(64, dtype=torch.float32),
            }
            metadata = {
                "fsample": "50000000",
                "fc": "5800000000",
                "bw": "20000000",
                "origin_angle": "0.0",
                "pri": "0.001",
            }
            save_file(tensors, str(file_path), metadata=metadata)

            report = validate_safetensors_file(file_path)
            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], [])

    def test_validate_safetensors_file_ok_for_int16_data(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            file_path = Path(temp_dir) / "sample_int16.safetensors"
            tensors = {
                "data": torch.randint(-1000, 1000, (64, 1, 128), dtype=torch.int16),
                "pos": torch.tensor(
                    [[float(i), 0.0, 1.0] for i in range(64)],
                    dtype=torch.float32,
                ),
                "att": torch.zeros((64, 3), dtype=torch.float32),
                "counts": torch.arange(64, dtype=torch.float32),
            }
            metadata = {
                "fsample": "50000000",
                "fc": "5800000000",
                "bw": "20000000",
                "origin_angle": "0.0",
                "pri": "0.001",
            }
            save_file(tensors, str(file_path), metadata=metadata)

            report = validate_safetensors_file(file_path)
            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from torchbp.output import write_geotiff, write_world_file


class TestOutputFormats(unittest.TestCase):
    def test_write_geotiff(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            output = Path(temp_dir) / "test.tif"
            image = np.arange(12, dtype=np.float32).reshape(3, 4)
            write_geotiff(output, image, metadata={"profile": "test"})

            self.assertTrue(output.exists())
            loaded = tifffile.imread(str(output))
            self.assertEqual(loaded.shape, (3, 4))
            self.assertAlmostEqual(float(loaded[2, 3]), 11.0)

    def test_write_world_file(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            output = Path(temp_dir) / "test.pgw"
            write_world_file(
                output,
                xmin=0.0,
                xmax=100.0,
                ymin=0.0,
                ymax=50.0,
                width=10,
                height=5,
            )
            self.assertTrue(output.exists())
            lines = output.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 6)
            self.assertEqual(lines[1], "0.0")
            self.assertEqual(lines[2], "0.0")


if __name__ == "__main__":
    unittest.main()

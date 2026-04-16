import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESS_SCRIPT = REPO_ROOT / "examples" / "sar_process_safetensor.py"
CART_SCRIPT = REPO_ROOT / "examples" / "sar_polar_to_cart.py"


def _make_synthetic_safetensors(path: Path, nsweeps: int = 128, nsamples: int = 512) -> None:
    torch.manual_seed(0)

    data = torch.randn(nsweeps, 1, nsamples, dtype=torch.float32)

    x = torch.linspace(0.0, 12.0, nsweeps)
    y = 0.2 * torch.sin(torch.linspace(0.0, 4.0, nsweeps))
    z = torch.full((nsweeps,), 1.5)
    pos = torch.stack((x, y, z), dim=1).to(torch.float32)

    roll = torch.zeros(nsweeps, dtype=torch.float32)
    pitch = torch.zeros(nsweeps, dtype=torch.float32)
    yaw = torch.linspace(-0.2, 0.2, nsweeps, dtype=torch.float32)
    att = torch.stack((roll, pitch, yaw), dim=1)

    counts = torch.arange(nsweeps, dtype=torch.float32)

    tensors = {
        "data": data,
        "pos": pos,
        "att": att,
        "counts": counts,
    }

    metadata = {
        "fsample": "50000000",
        "fc": "5800000000",
        "sweeps": str(float(nsweeps)),
        "bw": "20000000",
        "origin_angle": "0.0",
        "samples": str(float(nsamples)),
        "pri": "0.001",
        "channels": "1",
    }

    save_file(tensors, str(path), metadata=metadata)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA runtime")
class TestExamplesSmoke(unittest.TestCase):
    def test_synthetic_pipeline_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            input_file = work_dir / "synthetic.safetensors"
            _make_synthetic_safetensors(input_file)

            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else str(REPO_ROOT)
            )

            process_cmd = [
                sys.executable,
                str(PROCESS_SCRIPT),
                str(input_file),
                "--nsweeps",
                "128",
                "--fft-oversample",
                "1.0",
                "--skip-png",
            ]
            process_result = subprocess.run(
                process_cmd,
                cwd=work_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )
            self.assertEqual(
                process_result.returncode,
                0,
                msg=f"sar_process_safetensor failed:\nSTDOUT:\n{process_result.stdout}\nSTDERR:\n{process_result.stderr}",
            )

            pickle_path = work_dir / "sar_img.p"
            self.assertTrue(pickle_path.exists(), "sar_img.p was not created")
            self.assertGreater(pickle_path.stat().st_size, 0, "sar_img.p is empty")

            with open(pickle_path, "rb") as handle:
                sar_img, mission, grid_dict, grid_polar_dict, origin, origin_angle = pickle.load(handle)
            self.assertEqual(sar_img.ndim, 2)
            self.assertGreater(sar_img.shape[0], 0)
            self.assertGreater(sar_img.shape[1], 0)
            self.assertTrue(torch.isfinite(torch.from_numpy(sar_img)).all().item())
            self.assertIn("fc", mission)
            self.assertIn("x", grid_dict)
            self.assertIn("r", grid_polar_dict)
            self.assertEqual(len(origin), 3)
            self.assertIsInstance(origin_angle, float)

            cart_cmd = [
                sys.executable,
                str(CART_SCRIPT),
                str(pickle_path),
                "--dpi",
                "120",
                "--max-side",
                "1024",
            ]
            cart_result = subprocess.run(
                cart_cmd,
                cwd=work_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )
            self.assertEqual(
                cart_result.returncode,
                0,
                msg=f"sar_polar_to_cart failed:\nSTDOUT:\n{cart_result.stdout}\nSTDERR:\n{cart_result.stderr}",
            )

            image_path = work_dir / "sar_img_cart.png"
            self.assertTrue(image_path.exists(), "sar_img_cart.png was not created")
            self.assertGreater(image_path.stat().st_size, 0, "sar_img_cart.png is empty")


if __name__ == "__main__":
    unittest.main()

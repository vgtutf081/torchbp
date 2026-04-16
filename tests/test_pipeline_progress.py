import unittest

from api.pipeline import overall_progress


class TestPipelineProgress(unittest.TestCase):
    def test_weighted_overall_progress(self) -> None:
        value = overall_progress("backprojection", 65.0)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 100.0)

    def test_caps_progress_to_100(self) -> None:
        value = overall_progress("export", 150.0)
        self.assertEqual(value, 100.0)


if __name__ == "__main__":
    unittest.main()

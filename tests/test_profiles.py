import unittest

from torchbp.profiles import cart_profile_defaults, normalize_profile, process_profile_defaults


class TestProfiles(unittest.TestCase):
    def test_normalize_profile_default(self) -> None:
        self.assertEqual(normalize_profile(None), "standard")

    def test_normalize_profile_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            normalize_profile("ultra")

    def test_process_profile_defaults(self) -> None:
        fast = process_profile_defaults("fast_preview")
        high = process_profile_defaults("high_quality")
        self.assertLess(fast["nsweeps"], high["nsweeps"])
        self.assertLessEqual(fast["max_steps"], high["max_steps"])

    def test_cart_profile_defaults(self) -> None:
        fast = cart_profile_defaults("fast_preview")
        standard = cart_profile_defaults("standard")
        self.assertLessEqual(fast["dpi"], standard["dpi"])


if __name__ == "__main__":
    unittest.main()

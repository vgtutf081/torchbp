from __future__ import annotations

PROFILE_NAMES = {"fast_preview", "standard", "high_quality"}

PROCESS_PROFILE_DEFAULTS = {
    "fast_preview": {
        "nsweeps": 1024,
        "fft_oversample": 1.0,
        "max_steps": 4,
    },
    "standard": {
        "nsweeps": 10000,
        "fft_oversample": 1.5,
        "max_steps": 15,
    },
    "high_quality": {
        "nsweeps": 30000,
        "fft_oversample": 2.0,
        "max_steps": 30,
    },
}

CART_PROFILE_DEFAULTS = {
    "fast_preview": {
        "dpi": 180,
        "max_side": 1024,
        "oversample": 1,
        "multilook": 1,
    },
    "standard": {
        "dpi": 700,
        "max_side": None,
        "oversample": 1,
        "multilook": 2,
    },
    "high_quality": {
        "dpi": 900,
        "max_side": None,
        "oversample": 2,
        "multilook": 3,
    },
}


def normalize_profile(profile: str | None) -> str:
    if profile is None:
        return "standard"
    profile_name = str(profile).strip().lower()
    if profile_name not in PROFILE_NAMES:
        raise ValueError(
            f"Unknown profile '{profile}'. Expected one of: fast_preview, standard, high_quality"
        )
    return profile_name


def process_profile_defaults(profile: str | None) -> dict:
    profile_name = normalize_profile(profile)
    return dict(PROCESS_PROFILE_DEFAULTS[profile_name])


def cart_profile_defaults(profile: str | None) -> dict:
    profile_name = normalize_profile(profile)
    return dict(CART_PROFILE_DEFAULTS[profile_name])

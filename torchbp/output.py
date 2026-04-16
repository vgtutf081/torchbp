from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def write_geotiff(path: Path, image: np.ndarray, metadata: dict | None = None) -> None:
    image32 = np.asarray(image, dtype=np.float32)
    tifffile.imwrite(str(path), image32, metadata=metadata or {})


def write_world_file(
    path: Path,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    pixel_size_x = (xmax - xmin) / max(width, 1)
    pixel_size_y = (ymax - ymin) / max(height, 1)
    x_center = xmin + pixel_size_x / 2.0
    y_center = ymax - pixel_size_y / 2.0

    content = "\n".join(
        [
            f"{pixel_size_x:.12f}",
            "0.0",
            "0.0",
            f"{-pixel_size_y:.12f}",
            f"{x_center:.12f}",
            f"{y_center:.12f}",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")

#!/usr/bin/env python
# Visualize pickled radar image
import pickle
import os
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torchbp
from torchbp.profiles import cart_profile_defaults, normalize_profile
from torchbp.util import entropy
from torchbp.grid import PolarGrid, CartesianGrid
from torchbp.output import write_geotiff, write_world_file
from torchbp.gpu import require_cuda, has_cuda_kernel
import torch
import sys
import torch.nn.functional as F


def _load_json_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object")
    return data


def _resolve_value(cli_value, config: dict, key: str, default=None):
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def polar_to_cart_fallback_gpu(
    img_abs: torch.Tensor,
    origin: torch.Tensor,
    grid_polar: PolarGrid,
    grid: CartesianGrid,
    origin_angle: float,
) -> torch.Tensor:
    device = img_abs.device
    x = torch.linspace(grid.x0, grid.x1, grid.nx, device=device)
    y = torch.linspace(grid.y0, grid.y1, grid.ny, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    ox, oy = origin[0], origin[1]
    xr = xx - ox
    yr = yy - oy

    c = torch.cos(torch.tensor(-origin_angle, device=device, dtype=torch.float32))
    s = torch.sin(torch.tensor(-origin_angle, device=device, dtype=torch.float32))
    x_rot = c * xr - s * yr
    y_rot = s * xr + c * yr

    r = torch.sqrt(x_rot * x_rot + y_rot * y_rot + 1e-12)
    theta = y_rot / torch.clamp(r, min=1e-6)

    gx = 2 * (theta - grid_polar.theta0) / (grid_polar.theta1 - grid_polar.theta0) - 1
    gy = 2 * (r - grid_polar.r0) / (grid_polar.r1 - grid_polar.r0) - 1
    sample_grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)

    src = img_abs.unsqueeze(0).unsqueeze(0)
    out = F.grid_sample(src, sample_grid, mode="bilinear", align_corners=True)
    return out[0, 0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SAR polar image to cartesian")
    parser.add_argument("filename", nargs="?", default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Processing profile: fast_preview|standard|high_quality",
    )
    parser.add_argument("--dpi", type=int, default=None, help="Output PNG DPI")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output filename prefix (default: sar_img)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="Maximum output side in pixels (keeps aspect ratio).",
    )
    args = parser.parse_args()
    config = _load_json_config(args.config)
    profile = normalize_profile(_resolve_value(args.profile, config, "profile", "standard"))
    profile_defaults = cart_profile_defaults(profile)
    output_prefix = str(_resolve_value(args.output_prefix, config, "output_prefix", "sar_img"))

    filename = _resolve_value(args.filename, config, "filename", "sar_img.p")
    # Check in examples directory if file not found in current directory
    if not os.path.exists(filename) and os.path.exists(os.path.join("examples", filename)):
        filename = os.path.join("examples", filename)
    with open(filename, "rb") as f:
        sar_img, mission, grid_dict, grid_polar_dict, origin, origin_angle = pickle.load(f)

    # Convert dicts to Grid objects
    grid = CartesianGrid.from_dict(grid_dict)
    grid_polar = PolarGrid.from_dict(grid_polar_dict)

    dev = require_cuda().type
    sar_img = torch.from_numpy(sar_img).to(dtype=torch.complex64, device=dev)
    fc = mission["fc"]
    print("Entropy", entropy(sar_img).item())

    # Increase Cartesian image size
    oversample = int(config.get("oversample", profile_defaults["oversample"]))
    # Increases image size, but then resamples it down by the same amount
    # Can be used for multilook processing, when the input polar format data
    # resolution is higher than can fit into the Cartesian grid
    multilook = int(config.get("multilook", profile_defaults["multilook"]))
    grid = grid.resize(
        nx=int(oversample * grid.nx * multilook),
        ny=int(oversample * grid.ny * multilook)
    )

    plt.figure()
    origin = torch.from_numpy(origin).to(dtype=torch.float32, device=dev)
    # Amplitude scaling in image
    m = 20 * torch.log10(torch.median(torch.abs(sar_img))) - 3
    m = m.cpu().numpy()
    m2 = m + 40

    if has_cuda_kernel("torchbp::polar_to_cart_linear"):
        sar_img_cart = torchbp.ops.polar_to_cart(
            torch.abs(sar_img),
            origin,
            grid_polar,
            grid,
            fc,
            origin_angle
        )
    else:
        print("CUDA kernel torchbp::polar_to_cart_linear missing; using GPU interpolation fallback.")
        sar_img_cart = polar_to_cart_fallback_gpu(torch.abs(sar_img), origin, grid_polar, grid, origin_angle)
    extent = [grid.x0, grid.x1, grid.y0, grid.y1]
    img_db = torch.abs(sar_img_cart) + 1e-10
    out_shape = [img_db.shape[-2] // multilook, img_db.shape[-1] // multilook]
    max_side_value = _resolve_value(args.max_side, config, "max_side", profile_defaults["max_side"])
    if max_side_value is not None:
        max_side = max(1, int(max_side_value))
        current_max = max(out_shape)
        if current_max > max_side:
            scale = max_side / current_max
            out_shape = [max(1, int(out_shape[0] * scale)), max(1, int(out_shape[1] * scale))]
    if img_db.ndim == 2:
        img_db = img_db.unsqueeze(0).unsqueeze(0)
    elif img_db.ndim == 3:
        img_db = img_db.unsqueeze(0)
    img_db = F.interpolate(
        img_db,
        size=out_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    img_db = 20 * torch.log10(img_db)
    img_db = img_db.cpu().numpy()
    img_linear = np.power(10.0, img_db / 20.0)
    raster = img_linear.T.astype("float32")

    geotiff_path = f"{output_prefix}.tif"
    world_file_path = f"{output_prefix}_cart.pgw"
    write_geotiff(
        Path(geotiff_path),
        raster,
        metadata={"format": "torchbp", "source": "sar_polar_to_cart"},
    )
    write_world_file(
        Path(world_file_path),
        xmin=grid.x0,
        xmax=grid.x1,
        ymin=grid.y0,
        ymax=grid.y1,
        width=raster.shape[1],
        height=raster.shape[0],
    )

    plt.imshow(img_db.T, origin="lower", aspect="equal", vmin=m, vmax=m2, extent=extent)
    plt.grid(False)
    dpi = int(_resolve_value(args.dpi, config, "dpi", profile_defaults["dpi"]))
    plt.savefig(f"{output_prefix}_cart.png", dpi=dpi)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show(block=True)

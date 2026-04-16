#!/usr/bin/env python
# Visualize pickled radar image
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import torchbp
from torchbp.util import entropy
from torchbp.grid import PolarGrid, CartesianGrid
from torchbp.gpu import require_cuda, has_cuda_kernel
import torch
import sys
import torch.nn.functional as F


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
    parser.add_argument("filename", nargs="?", default="sar_img.p")
    parser.add_argument("--dpi", type=int, default=700, help="Output PNG DPI")
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="Maximum output side in pixels (keeps aspect ratio).",
    )
    args = parser.parse_args()

    filename = args.filename
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
    oversample = 1
    # Increases image size, but then resamples it down by the same amount
    # Can be used for multilook processing, when the input polar format data
    # resolution is higher than can fit into the Cartesian grid
    multilook = 2
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
    if args.max_side is not None:
        max_side = max(1, int(args.max_side))
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

    plt.imshow(img_db.T, origin="lower", aspect="equal", vmin=m, vmax=m2, extent=extent)
    plt.grid(False)
    plt.savefig("sar_img_cart.png", dpi=args.dpi)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show(block=True)

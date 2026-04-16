import pathlib
import sys
import torch

try:
	from . import _C
except ImportError:
	ext_suffix = ".pyd" if sys.platform.startswith("win") else ".so"
	ext_path = pathlib.Path(__file__).with_name(f"_C{ext_suffix}")
	if ext_path.exists():
		torch.ops.load_library(str(ext_path))
		_C = None
	else:
		raise

from . import ops, autofocus, util, polarimetry, interferometry, grid, gpu
from .grid import Grid, PolarGrid, CartesianGrid

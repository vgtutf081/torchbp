# Example SAR data processing script

Instructions:

 1. Download the sample data from: https://hforsten.com/sar.safetensors.zip
 2. Unzip the file to this directory.
 3. Run `sar_process_safetensor.py` (optimization based minimum entropy
    autofocus) or `sar_process_safetensor_gpga.py` (generalized phase gradient
    autofocus). It will process the file and display polar formatted image.
    Processed image is also saved to disk for next step.
 4. Run `sar_polar_to_cart.py` to display the previously saved image in Cartesian grid.

Notes:

- Scripts run in GPU-only mode and will fail if CUDA is unavailable.
- Scripts also validate required torchbp CUDA kernels and fail fast if kernels are missing.
- Run from repository root or from this directory; scripts resolve `examples/` paths automatically.

Some processing parameters can be modified in the `sar_process_safetensor.py`
file. For example, for higher resolution image set `nsweeps = 51200` and
increase `max_step_limit` to 5.

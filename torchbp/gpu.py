import torch


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU-only mode: CUDA is not available. "
            "Install CUDA-enabled PyTorch and use an NVIDIA GPU."
        )
    return torch.device("cuda")


def has_cuda_kernel(op_name: str) -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "CUDA")
    except Exception:
        return False


def require_cuda_kernels(op_names: list[str]) -> None:
    missing = [name for name in op_names if not has_cuda_kernel(name)]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "GPU-only mode: missing CUDA kernels: "
            f"{missing_list}. Rebuild torchbp with CUDA support "
            "(USE_CUDA=1 FORCE_CUDA=1)."
        )

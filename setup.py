import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "torchbp"

py_limited_api = False

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    is_windows = os.name == "nt"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and (torch.cuda.is_available() or force_cuda) and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    if use_cuda:
        print("Compiling with cuda support")
    else:
        print("No cuda support")

    if is_windows:
        extra_link_args = []
        extra_compile_args = {
            "cxx": [
                "/O2" if not debug_mode else "/Od",
                "/openmp",
            ],
            "nvcc": [
                "-O3" if not debug_mode else "-O0",
                "-DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS",
                "--use_fast_math",
                "-lineinfo",
                "-Xcompiler=/openmp",
            ],
        }
    else:
        extra_link_args = ["-fopenmp"]
        extra_compile_args = {
            "cxx": [
                "-O3" if not debug_mode else "-O0",
                "-fdiagnostics-color=always",
                "-fopenmp",
            ],
            "nvcc": [
                "-O3" if not debug_mode else "-O0",
                "-DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS",
                "--use_fast_math",
                "-lineinfo"
            ],
        }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].extend(["-g", "-G"])
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch", "numpy", "scipy"],
    extras_require = {
        'docs':  [
            "matplotlib >=3.5",
            "nbval >=0.9",
            "jupyter-client >=7.3.5",
            "sphinx-rtd-theme >=1.0",
            "sphinx >=4",
            "nbsphinx >= 0.8.9",
            "openpyxl >= 3",
            "lxml-html-clean >= 0.4.1",
            "numpydoc"]
    },
    description="Differentiable synthetic aperture radar library",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
    #url="",
    cmdclass={"build_ext": BuildExtension},
)

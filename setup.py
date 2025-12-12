import ast
import os
import re
import shutil
import setuptools
import subprocess
import sys
import torch
import platform
import urllib
import urllib.error
import urllib.request
from setuptools import find_packages

# from setuptools.command.build_py import build_py
from packaging.version import parse
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME, BuildExtension
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


# Compiler flags
cxx_flags = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
]

nvcc_flags = [
    "-forward-unknown-to-host-compiler",
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90a,code=sm_90a",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

# Sources
current_dir = os.path.dirname(os.path.realpath(__file__))
sources = ["csrc/python_api.cu"]
build_include_dirs = [
    f"{CUDA_HOME}/include",
    f"{CUDA_HOME}/include/cccl",
    f"{current_dir}/third_party/cutlass/include",
    f"{current_dir}/third_party/cutlass/tools/util/include",
]
build_libraries = ["cuda", "cudart", "nvrtc"]
build_library_dirs = [f"{CUDA_HOME}/lib64", f"{CUDA_HOME}/lib64/stubs"]


def get_ext_modules():
    return [
        CUDAExtension(
            name="moe_test",
            sources=sources,
            include_dirs=build_include_dirs,
            libraries=build_libraries,
            library_dirs=build_library_dirs,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ]


if __name__ == "__main__":
    # noinspection PyTypeChecker
    setuptools.setup(
        name="moe_test",
        packages=find_packages("."),
        ext_modules=get_ext_modules(),
        zip_safe=False,
        cmdclass={
            "build_ext": BuildExtension,
        },
    )

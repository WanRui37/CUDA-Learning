from setuptools import setup, Extension, find_packages
import torch
import os
import subprocess
import sys
from pathlib import Path

from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)

torch.utils.cpp_extension.CUDA_HOME = '/usr/local/cuda-12.2/'
CUDA_HOME = torch.utils.cpp_extension.CUDA_HOME
# CUDA_HOME = '/usr/local/cuda-12.2/'

SKIP_CUDA_BUILD = False
BUILD_TARGET = "cuda"
IS_ROCM = False
this_dir = os.path.dirname(os.path.abspath(__file__))

cuda_lib_path = "/usr/local/cuda-12.2/lib64"
cuda_lib_path2 = "/usr/local/cuda-12.2/targets/x86_64-linux/include"
cuda_bin_path = "/usr/local/cuda-12.2/bin"

os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["CUDA_TOOLKIT_ROOT_DIR"] = CUDA_HOME
os.environ["PATH"] = cuda_bin_path + os.pathsep + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = cuda_lib_path + cuda_lib_path2 + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

print(f"====================")
print(f"CUDA_HOME:{CUDA_HOME}")
print(f"ROCM_HOME:{ROCM_HOME}")
print(f"IS_HIP_EXTENSION:{IS_HIP_EXTENSION}")
print(f"this file:{this_dir}")
print(f"====================")

# cxx_args = ['-std=c++11']
# cxx_args = ['-std=c++14']
# cxx_args = ['-std=c++17']
cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_89,code=sm_89")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90,code=sm_90")

ext_modules=[]

def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]

if not SKIP_CUDA_BUILD and not IS_ROCM:
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    ext_modules.append(        
        CUDAExtension(
            name="add2",
            sources=["pytorch/add2_ops.cpp", "kernel/add2_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            include_dirs=[
                Path(this_dir) / ".." / "include",
            ],
        )
)

setup(
    name="add2",
    packages=find_packages(
        exclude=(
            "build",
            "kernel",
            "include",
            "add2.egg-info",
        )
    ),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension
    }
)
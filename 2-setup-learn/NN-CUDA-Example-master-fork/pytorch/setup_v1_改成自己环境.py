from setuptools import setup, Extension
import torch
import os
import subprocess
import sys

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

cuda_lib_path = "/usr/local/cuda-12.2/lib64"
cuda_lib_path2 = "/usr/local/cuda-12.2/targets/x86_64-linux/include"
cuda_bin_path = "/usr/local/cuda-12.2/bin"

os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["CUDA_TOOLKIT_ROOT_DIR"] = CUDA_HOME
os.environ["PATH"] = cuda_bin_path + os.pathsep + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = cuda_lib_path + cuda_lib_path2 + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

print(f"====================")
print(f"CUDA_HOME:{torch.utils.cpp_extension.CUDA_HOME}")
print(f"ROCM_HOME:{ROCM_HOME}")
print(f"IS_HIP_EXTENSION:{IS_HIP_EXTENSION}")
print(f"====================")

# cxx_args = ['-std=c++11']
# cxx_args = ['-std=c++14']
cxx_args = ['-std=c++17']

ext_modules=[]
ext_modules.append(        
    CUDAExtension(
        name="add2",
        sources=["pytorch/add2_ops.cpp", "kernel/add2_kernel.cu"],
        extra_compile_args={'cxx': cxx_args}
    )
)

setup(
    name="add2",
    include_dirs=["./include"],
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension
    }
)
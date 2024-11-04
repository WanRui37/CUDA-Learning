import torch
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline
import os

# 设置PATH环境变量
cuda_bin_path = "/usr/local/cuda-12.2/bin"
os.environ["PATH"] = cuda_bin_path + os.pathsep + os.environ.get("PATH", "")
print(os.environ["PATH"])

# 设置LD_LIBRARY_PATH环境变量
cuda_lib_path = "/usr/local/cuda-12.2/lib64"
cuda_lib_path2 = "/usr/local/cuda-12.2/targets/x86_64-linux/include"
os.environ["LD_LIBRARY_PATH"] = cuda_lib_path + cuda_lib_path2 + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
print(os.environ["LD_LIBRARY_PATH"])

os.environ['CXX'] = '/usr/lib/ccache/g++-11'
os.environ['CC'] = '/usr/lib/ccache/gcc-11'

def show_img(x, figsize=(4,3), **kwargs):
    "Display HW or CHW format image `x`"
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape)==3: x = x.permute(1,2,0)  # CHW -> HWC
    plt.imshow(x.cpu(), **kwargs)

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b;}
'''

def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False, name=None):
    "Simple wrapper for torch.utils.cpp_extension.load_inline"
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if opt else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                        extra_cuda_cflags=[flags],
                        extra_ldflags = [f'-L{cuda_lib_path}'],
                        extra_include_paths = [cuda_lib_path2, cuda_lib_path],
                        verbose=verbose, 
                        name=name)

def cdiv(a,b):
    "Int ceiling division of `a` over `b`"
    return (a+b-1)//b

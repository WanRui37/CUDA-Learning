import time
import argparse
import numpy as np
import torch
import element_wise

B = 64
C = 32
H = 112
W = 112
N = B*C*H*W

# a = torch.rand((B,C,H,W), device="cuda:0", dtype=float)
# b = torch.rand((B,C,H,W), device="cuda:0", dtype=float)
# cuda_c = torch.rand((B,C,H,W), device="cuda:0", dtype=float)

a = torch.rand((B,C,H,W)).cuda().float().contiguous()
b = torch.rand((B,C,H,W)).cuda().float().contiguous()
cuda_c = torch.rand((B,C,H,W)).cuda().float().contiguous()

ntest = 10

def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
    return times, res

def run_cuda():
    element_wise.elementwise_add_f32(a, b, cuda_c)
    return cuda_c

def run_torch():
    c = a + b
    return c.contiguous()

if __name__ == "__main__":
    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time, torch_res = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    torch.allclose(cuda_res, torch_res)
    print("Kernel test passed.")

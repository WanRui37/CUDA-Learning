#include <cuda_runtime.h>
#include <stdio.h>

__global__ void print_cord_kernel(){
    int index = threadIdx.z * blockDim.x * blockDim.y + \
              threadIdx.y * blockDim.x + \
              threadIdx.x;

    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int z  = blockIdx.z * blockDim.z + threadIdx.z;

    printf("block idx: (%3d, %3d, %3d), thread idx: %3d, cord: (%3d, %3d, %3d)\n",
         blockIdx.z, blockIdx.y, blockIdx.x,
         index, x, y, z);
}

void print_cord(){
    dim3 block(2, 2, 2);
    dim3 grid(2, 2, 2);

    print_cord_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();
}

int main() {
     print_cord();
    return 0;
}

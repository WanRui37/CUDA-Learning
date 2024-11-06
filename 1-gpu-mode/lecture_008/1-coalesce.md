# Cuda 代码解析

这段 C++ 代码使用了 CUDA 进行数据的复制操作，分别实现了非合并访问和合并访问的两种方式，并对两种方式进行性能对比。

## 一、函数定义

1. `copyDataNonCoalesced`函数：
   - **功能**：这个 CUDA 内核函数实现了非合并访问的数据复制。每个线程从输入数组`in`中以非连续的方式读取数据，然后写入到输出数组`out`中。具体来说，输出数组`out`的第`index`个元素来自输入数组`in`的`(index * 2) % n`位置。
   - **参数**：
     - `float *in`：输入数组的指针。
     - `float *out`：输出数组的指针。
     - `int n`：数组的长度。
   - **执行条件**：只有当线程索引`index`小于数组长度`n`时，才进行数据复制操作。

2. `copyDataCoalesced`函数：
   - **功能**：这个 CUDA 内核函数实现了合并访问的数据复制。每个线程从输入数组`in`中以连续的方式读取数据，然后写入到输出数组`out`中。即输出数组`out`的第`index`个元素直接来自输入数组`in`的第`index`个位置。
   - **参数**：
     - `float *in`：输入数组的指针。
     - `float *out`：输出数组的指针。
     - `int n`：数组的长度。
   - **执行条件**：只有当线程索引`index`小于数组长度`n`时，才进行数据复制操作。

3. `initializeArray`函数：
   - **功能**：用于初始化给定的数组。将数组中的每个元素设置为其索引值。
   - **参数**：
     - `float *arr`：要初始化的数组指针。
     - `int n`：数组的长度。

## 二、主函数`main`

1. **变量初始化**：
   - 定义了数组长度`n`为`1 << 24`，表示数组的大小为$2^{24}$。这个值可以根据需要进行调整以增加或减小工作量。
   - 定义了两个指针`in`和`out`，分别用于指向输入和输出数组。

2. **内存分配**：
   - 使用`cudaMallocManaged`函数在统一内存空间中分配了输入数组`in`和输出数组`out`的内存空间，大小均为`n * sizeof(float)`。

3. **数组初始化**：
   - 调用`initializeArray`函数初始化输入数组`in`，将其每个元素设置为其索引值。

4. **非合并访问数据复制**：
   - 定义了线程块大小`blockSize`为 128。可以根据需要调整这个值，不同的块大小可能会影响性能。
   - 计算需要的线程块数量`numBlocks`，确保有足够的线程块来覆盖整个数组。计算公式为`(n + blockSize - 1) / blockSize`。
   - 使用`copyDataNonCoalesced<<<numBlocks, blockSize>>>(in, out, n)`启动非合并访问的 CUDA 内核函数，将输入数组`in`中的数据以非合并访问的方式复制到输出数组`out`中。
   - 使用`cudaDeviceSynchronize`函数等待 GPU 上的所有任务完成。

5. **重置输出数组并进行合并访问数据复制**：
   - 再次调用`initializeArray(out, n)`重置输出数组，以便进行下一次数据复制操作。
   - 使用`copyDataCoalesced<<<numBlocks, blockSize>>>(in, out, n)`启动合并访问的 CUDA 内核函数，将输入数组`in`中的数据以合并访问的方式复制到输出数组`out`中。
   - 使用`cudaDeviceSynchronize`函数等待 GPU 上的所有任务完成。

6. **内存释放**：
   - 使用`cudaFree`函数释放输入数组`in`和输出数组`out`所占用的内存空间。

7. **返回值**：
   - 主函数返回 0，表示程序正常结束。
 <br/>  <br/>  <br/> 

# 两个实验结果的详细分析：

# 一、实验概述
进行了两个实验，分别是`copyDataNonCoalesced`（非合并访问数据实验）和`copyDataCoalesced`（合并访问数据实验）。每个实验都给出了多个方面的性能指标。

# 二、性能指标分析

## （一）GPU Speed Of Light Throughput 部分
1. **DRAM Frequency（内存频率）**：
   - `copyDataNonCoalesced`：10.12 cycle/nsecond。
   - `copyDataCoalesced`：10.00 cycle/nsecond。两者较为接近，说明内存的时钟周期在两个实验中差别不大。
2. **SM Frequency（流多处理器频率）**：
   - `copyDataNonCoalesced`：2.20 cycle/nsecond。
   - `copyDataCoalesced`：2.18 cycle/nsecond。同样差别较小。
3. **Elapsed Cycles（经过的时钟周期数）**：
   - `copyDataNonCoalesced`：399,799 cycle。
   - `copyDataCoalesced`：215,278 cycle。合并访问数据的实验经过的时钟周期数明显更少，说明该实验可能执行得更快。
4. **Memory Throughput（内存吞吐量）和 DRAM Throughput（直接内存访问吞吐量）**：
   - `copyDataNonCoalesced`：95.63%。
   - `copyDataCoalesced`：94.02%。两者都较高，表明内存的使用效率不错，但非合并访问数据实验略高一些。
5. **Duration（执行时间）**：
   - `copyDataNonCoalesced`：181.25 usecond（微秒）。
   - `copyDataCoalesced`：98.85 usecond。合并访问数据实验的执行时间更短，性能更好。
6. **L1/TEX Cache Throughput（一级纹理缓存吞吐量）**：
   - `copyDataNonCoalesced`：10.26%。
   - `copyDataCoalesced`：17.28%。合并访问数据实验的一级纹理缓存吞吐量更高，可能意味着更好地利用了缓存。
7. **L2 Cache Throughput（二级缓存吞吐量）**：
   - `copyDataNonCoalesced`：32.90%。
   - `copyDataCoalesced`：35.43%。同样，合并访问数据实验的二级缓存吞吐量更高。
8. **SM Active Cycles（流多处理器活跃时钟周期数）**：
   - `copyDataNonCoalesced`：393,214.05 cycle。
   - `copyDataCoalesced`：208,912.30 cycle。合并访问数据实验的流多处理器活跃时钟周期数更少，说明其在流多处理器上的计算效率可能更高。
9. **Compute (SM) Throughput（流多处理器计算吞吐量）**：
   - `copyDataNonCoalesced`：8.98%。
   - `copyDataCoalesced`：11.43%。合并访问数据实验的流多处理器计算吞吐量更高。

## （二）Launch Statistics 部分
两个实验在这部分的指标完全相同：
1. **Block Size（块大小）**：128。
2. **Function Cache Configuration（函数缓存配置）**：CachePreferNone。
3. **Grid Size（网格大小）**：131,072。
4. **Registers Per Thread（每个线程的寄存器数量）**：16 register/thread。
5. **Shared Memory Configuration Size（共享内存配置大小）**：32.77 Kbyte。
6. **Driver Shared Memory Per Block（每个块的驱动共享内存）**：1.02 Kbyte/block。
7. **Dynamic Shared Memory Per Block（每个块的动态共享内存）**：0 byte/block。
8. **Static Shared Memory Per Block（每个块的静态共享内存）**：0 byte/block。
9. **Threads（线程数量）**：16,777,216。
10. **Waves Per SM（每个流多处理器的波数量）**：85.33。

## （三）Occupancy 部分
1. **Block Limit SM（流多处理器上的块限制）、Block Limit Registers（寄存器的块限制）、Block Limit Shared Mem（共享内存的块限制）、Block Limit Warps（波的块限制）、Theoretical Active Warps per SM（理论上每个流多处理器的活跃波数量）、Theoretical Occupancy（理论占用率）**：
   - 两个实验在这些指标上的值完全相同，表明理论上它们在流多处理器的资源限制和理论占用率方面没有差异。
2. **Achieved Occupancy（实际占用率）和 Achieved Active Warps Per SM（实际每个流多处理器的活跃波数量）**：
   - `copyDataNonCoalesced`：Achieved Occupancy 为 89.02%，Achieved Active Warps Per SM 为 42.73 warp。
   - `copyDataCoalesced`：Achieved Occupancy 为 67.85%，Achieved Active Warps Per SM 为 32.57 warp。非合并访问数据实验的实际占用率和实际活跃波数量更高一些。

# 三、总结
总体而言，合并访问数据的实验在执行时间、内存吞吐量、缓存吞吐量和流多处理器计算吞吐量等方面表现更好。虽然非合并访问数据实验在某些指标上（如实际占用率和实际活跃波数量）略高一些，但综合来看，合并访问数据的方式在性能上更具优势。两个实验在启动统计信息方面完全相同，说明它们在启动参数设置上没有差异。对于进一步优化，可以参考优化建议，如分析内存使用情况、减少 warp 调度开销和解决负载不平衡问题等，以提高性能。
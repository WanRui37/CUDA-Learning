cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# modify to your own nvcc path, or delete it if ok

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc")
set(CUDNN_INCLUDE_DIR "/home/ic611/anaconda3/envs/cuda12_1/include")
set(CUDNN_INCLUDE_PATH "/home/ic611/anaconda3/envs/cuda12_1/include")
set(CUDNN_LIBRARY "/home/ic611/anaconda3/envs/cuda12_1/lib/libcudnn.so")
set(CUDNN_LIBRARY_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/libcudnn.so")
set(Python_EXECUTABLE "/home/ic611/anaconda3/envs/cuda12_1/bin/python")

set(TORCH_INSTALL_PREFIX "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/")
set(TORCH_DIR "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/;${TORCH_DIR}")
set(TORCH_DIR "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/share/cmake/;${TORCH_DIR}")
set(TORCH_DIR "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/share/cmake/Torch/;${TORCH_DIR}")
set(TORCH_DIR "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/;${TORCH_DIR}")
set(TORCH_DIR "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/lib/;${TORCH_DIR}")

# set(DCMAKE_PREFIX_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/;${DCMAKE_PREFIX_PATH}")
# set(DCMAKE_PREFIX_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/share/cmake/;${DCMAKE_PREFIX_PATH}")
# set(DCMAKE_PREFIX_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/share/cmake/Torch/;${DCMAKE_PREFIX_PATH}")
# set(DCMAKE_PREFIX_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/;${DCMAKE_PREFIX_PATH}")
# set(DCMAKE_PREFIX_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/lib/;${DCMAKE_PREFIX_PATH}")

# set(CMAKE_MODULE_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/;${CMAKE_MODULE_PATH}")
# set(CMAKE_MODULE_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/share/cmake/;${CMAKE_MODULE_PATH}")
# set(CMAKE_MODULE_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/share/cmake/Torch/;${CMAKE_MODULE_PATH}")
# set(CMAKE_MODULE_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/;${CMAKE_MODULE_PATH}")
# set(CMAKE_MODULE_PATH "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/lib/;${CMAKE_MODULE_PATH}")

# set(TORCH_PYTHON_LIBRARY "/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/lib/libtorch_python.so;${TORCH_PYTHON_LIBRARY}")

project(add2 LANGUAGES CXX CUDA)

find_package(Python REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-std=c++17" )

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

message(STATUS "PATH: ${PATH}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_FRAMEWORK_PATH: ${CMAKE_FRAMEWORK_PATH}")
message(STATUS "CMAKE_APPBUNDLE_PATH: ${CMAKE_APPBUNDLE_PATH}")
message(STATUS "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}")

find_package(Torch REQUIRED 
        PATHS "${TORCH_DIR}")
message(STATUS "TORCH_INSTALL_PREFIX_4:${TORCH_INSTALL_PREFIX}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

message(STATUS  "DCMAKE_PREFIX_PATH:${DCMAKE_PREFIX_PATH}")
message(STATUS "TORCH_PYTHON_LIBRARY:${TORCH_PYTHON_LIBRARY}")
message(STATUS "TORCH_DIR:${TORCH_DIR}")
message(STATUS "TORCH_LIBRARIES:${TORCH_LIBRARIES}")

# modify to your own python path, or delete it if ok
# include_directories("/usr/local/cuda-12.2/targets/x86_64-linux/lib/stubs/libcuda.so")
# set(CUDA_SO_LIBRARIES "/usr/local/cuda-12.2/targets/x86_64-linux/lib/stubs/libcuda.so")
# message(STATUS "CUDA_SO_LIBRARIES:${CUDA_SO_LIBRARIES}")

# set(SRCS add2_ops.cpp ../kernel/add2_kernel.cu)
# add_library(add2 SHARED ${SRCS})

# target_link_libraries(add2 "${CUDA_SO_LIBRARIES}" "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")


set(TORCH_LIBRARIES "torch;torch_library;/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/lib/libc10.so;/usr/local/cuda-12.2/lib64/libnvrtc.so;/usr/local/cuda-12.2/lib64/libnvToolsExt.so;/usr/local/cuda-12.2/lib64/libcudart.so;/home/ic611/anaconda3/envs/cuda12_1/lib/python3.12/site-packages/torch/lib/libc10_cuda.so")

include_directories("/usr/local/cuda-12.2/targets/x86_64-linux/lib/stubs/libcuda.so")
set(CUDA_SO_LIBRARIES "/usr/local/cuda-12.2/targets/x86_64-linux/lib/stubs/libcuda.so")
message(STATUS "CUDA_SO_LIBRARIES:${CUDA_SO_LIBRARIES}")

set(SRCS add2_ops.cpp ../kernel/add2_kernel.cu)
# message(STATUS "SRCS:${SRCS}")
# message(STATUS "CUDA_NVCC_EXECUTABLE:${CUDA_NVCC_EXECUTABLE}")
include_directories(/home/ic611/anaconda3/envs/cuda12_1/include/python3.12/)
include_directories(../include)

add_library(add2 SHARED ${SRCS})

target_link_libraries(add2 "${CUDA_SO_LIBRARIES}" "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
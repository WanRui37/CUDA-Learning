# 定义CUDA的安装目录变量
CUDA_HOME := /usr/local/cuda-12.2

# 指定nvcc编译器的路径
NVCC := $(CUDA_HOME)/bin/nvcc

# 指定编译器，这里假设宿主编译器是g++
CC := g++

# 定义编译选项，这里包括CUDA架构的编码
GENCODE_FLAGS := -gencode arch=compute_50,code=sm_50 \
				 -gencode arch=compute_52,code=sm_52 \
				 -gencode arch=compute_60,code=sm_60 \
				 -gencode arch=compute_61,code=sm_61 \
				 -gencode arch=compute_70,code=sm_70 \
				 -gencode arch=compute_75,code=sm_75 \
				 -gencode arch=compute_80,code=sm_80 \
				 -gencode arch=compute_86,code=sm_86 \
				 -gencode arch=compute_90,code=sm_90

# 定义编译器使用的其他选项
NVCC_FLAGS := -O2 $(GENCODE_FLAGS) --expt-extended-lambda

# 定义目标文件和源文件
TARGET := ./bin/print_theardId.exe
SOURCE := print_theardId.cu

# 创建目标目录，如果不存在的话
$(shell mkdir -p ./bin)

# 声明伪目标
.PHONY: all clean

# 默认目标
all: $(TARGET)

# 目标依赖于源文件
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# 清理编译生成的文件
clean:
	rm -f $(TARGET)


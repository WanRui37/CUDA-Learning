# Neural Network CUDA Example

原工程地址如后面所示[https://github.com/godweiyang/NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example)

想在自己电脑部署，需要做那些修改，只需要关注./pytorch/setup.py和./pytorch/setup_old.py以及./pytorch/CMakeLists.txt和./pytorch/CMakeLists_old.txt的区别即可。

## Linux Environments
* 显卡 GeForce RTX 4090
* Ubuntu 20.04.2
* NVIDIA Driver: 535.183.01
* CUDA: 12.2
* CMake: 3.16.3
* GCC: 9.4.0

## anaconda Environments
* Python: 3.12
* PyTorch: 2.4.1 py3.12_cuda12.1_cudnn9.1.0_0
* pytorch-cuda 12.1
* Ninja: 1.12.1

## Code structure
```shell
├── include
│   └── add2.h
├── kernel
│   └── add2_kernel.cu
├── pytorch
│   ├── add2_ops.cpp
│   ├── time.py
│   ├── train.py
│   ├── setup.py
│   ├── setup_old.py
│   └── CMakeLists.txt
│   └── CMakeLists_old.txt
└── README.md
```

## PyTorch
### Compile cpp and cuda
**JIT**  
直接跑它会自动编译好

**Setuptools**
使用下面的代码编译setup
```shell
python3 pytorch/setup.py install
```

**CMake**
使用下面的代码编译cmake
```shell
mkdir build
cd build
cmake ../pytorch
make
```

### Run python
**Compare kernel running time**

|&nbsp;&nbsp;cuda  time: 19.717us &nbsp;&nbsp;|&nbsp;&nbsp;torch time: 23.127us  &nbsp;&nbsp;|
```shell
python3 pytorch/time.py --compiler jit
```

|&nbsp;&nbsp;cuda  time: 10.514us &nbsp;&nbsp;|&nbsp;&nbsp;torch time: 15.473us  &nbsp;&nbsp;|
```shell
python3 pytorch/time.py --compiler setup
```

|&nbsp;&nbsp;cuda  time: 14.186us &nbsp;&nbsp;|&nbsp;&nbsp;torch time: 14.853us  &nbsp;&nbsp;|
```shell
python3 pytorch/time.py --compiler cmake
```

**Train model**  
```shell
python3 pytorch/train.py --compiler jit
python3 pytorch/train.py --compiler setup
python3 pytorch/train.py --compiler cmake
```
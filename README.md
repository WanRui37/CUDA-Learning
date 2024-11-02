# CUDA-Learning
&#8195;&#8195;本工程记录我的CUDA学习过程

## 1-gpu-mode
&#8195;&#8195;第1部分基于[gpu-mode](https://github.com/gpu-mode/lectures)这个讲座进行，因为原有的代码的环境依赖并没有详细讲解，我将在这个工程当中详细阐述。

&#8195;&#8195;下面是我的环境：
<style>
table
{
    margin: auto;
}
</style>

| <div style="width:100px">设备/环境</div> | 安装操作/版本 | 
| :-------- |  :-------- |
| Ubuntu | 20.04.6 |
| 显卡 | RTX4090 |
| CUDA | 12.2 |
| pytorch | conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia |
| ninja | conda install conda-forge::ninja |

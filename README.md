# CUDA-Learning
&#8195;&#8195;本工程记录我的CUDA学习过程

## 1-gpu-mode
&#8195;&#8195;第1部分基于[gpu-mode](https://github.com/gpu-mode/lectures)这个讲座进行，因为原有的代码的环境依赖并没有详细讲解，我将在这个工程当中详细阐述。

&#8195;&#8195;下面是我的环境：

<table align="center">
  <tr>
    <td><div style="width:100px"><b>设备/环境</b></div></td><td><b>安装操作/版本</b></td>
  </tr>
  <tr>
    <td>Ubuntu</td><td>20.04.6</td>
  </tr>
  <tr>
    <td>显卡</td><td>RTX4090</td>
  </tr>
  <tr>
    <td>CUDA</td><td>12.2</td>
  </tr>
  <tr>
    <td>pytorch</td><td>conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia</td>
  </tr>
  <tr>
    <td>ninja</td><td>conda install conda-forge::ninja</td>
  </tr>
  <tr>
    <td>numba</td><td>conda install numba::numba</td>
  </tr>
  <tr>
  <td>matplotlib</td><td>conda install matplotlib</td>
  </tr>
</table>

## 2-setup-learn
第二部分是学习setup编译、cmake编译的一些小例子

1、Neural Network CUDA Example原工程地址如后面所示[https://github.com/godweiyang/NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example)
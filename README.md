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
    <td>pytorch</td><td>conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia</td>
  </tr>
  <tr>
    <td>ninja</td><td>conda install conda-forge::ninja</td>
  </tr>
  <tr>
    <td>numba</td><td>conda install numba::numba</td>
  </tr>
</table>
---
title: MobileNet Note
layout: default
tags: [machine learning,]
---


# MobileNet Note

MobileNet 是 Google 2017年的一个工作，提出了一个轻量级的神经网络模型架构。

arXiv链接：<https://arxiv.org/abs/1704.04861>

MobileNet 的主要工作可以概况为以下三个部分：

- 深度可分离卷积，将一层卷积分解为两层计算，以更少的参数和计算量产生同样格式的输出
- 引入超参 $\alpha$ ，减少模型中每一层的通道数
- 引入超参 $\rho$，缩小原始图片尺寸，则整个模型中产生的 feature map size 都会缩小

## 1 深度可分离卷积（depthwise separable convolution）

假设有一个卷积核 kernel_size = $F\times F$ ，输入通道数 = $C_{in}$ ，输出通道数 = $C_{out}$ 的卷积层 conv。

那么深度可分离卷积以如下两个卷积层来代替原始卷积。

**第一个卷积层conv1**：

kernel_size = $F\times F$，输入通道数=输出通道数 = $C_{in}$，以**分组卷积**的形式产生 $C_{in}$ 个 feature map，分组数也为 $C_{in}$ ，即输入的每个通道单独做卷积；

**第二个卷积层conv2**：

kernel_size = $1\times1$ ，输入通道数 = $C_{in}$ ，输出通道数 = $C_{out}$，将上一层得到的 $C_{in}$ 个 feature map 再 combine 为 $C_{out}$ 个。

![此处应有图](/img/dw_conv.jpg)

*1. 参数减少量*

before: 

$$
F\times F\times C_{in}\times C_{out}
$$

after: 

$$
F\times F\times C_{in}+ C_{in}\times C_{out}
$$

则参数量减少为原来的 $\cfrac{1}{C_{out}}+\cfrac{1}{F^2}$，若 $F=3$，则参数量减少为原来的 $\cfrac{1}{9}$。

*2. FLOPs 减少量*

假设卷积层输出的 feature map 的size 为 $W_{out}\times H_{out}$，那么卷积层所需要的计算量 FLOPs 分别如下。

before: 

$$
FLOPs_{conv}=2F^2\times W_{out}\times H_{out} \times C_{in}\times C_{out}
$$

after: 

$$
\begin{aligned}
FLOPs_{conv1+2}=(2F^2\times W_{out}\times H_{out} \times C_{in}\times 1) +
(2\times W_{out}\times H_{out} \times C_{in}\times C_{out})
\end{aligned}
$$

则计算量减少为原来的 $\cfrac{1}{C_{out}}+\cfrac{1}{F^2}$，若 $F=3$，则参数量减少为原来的 $\cfrac{1}{9}$。



## 2 整体架构

**网络层数**

![](/img/dw_conv_block.PNG)

MobileNet 由1个普通的卷积层+13个可分离卷积层+1个全连接层构成，因此若将可分离卷积中的每一层都作为独立的一层，则MobileNet共有28层。

MobileNet 95%的计算量都用在 $1\times1$ 卷积上，卷积的本质是矩阵乘法，而 $1\times1$ 卷积的矩阵乘法是极其稀疏的，通过高度优化的GEMM来实现可以加速计算。

**两个超参**

取 $\alpha\in \{1,0.75,0.5,0.25\},\rho\in\{224,192,160,128\}$，在 ImageNet 上模型精度下降如下图所示。

![](/img/dw_hyper.png)

**训练细节**

因为深度可分离卷积对比原始卷积已经大幅度减少了参数数量，极大地缓和了过拟合问题，所以不再使用L2正则项（即weight decay）。
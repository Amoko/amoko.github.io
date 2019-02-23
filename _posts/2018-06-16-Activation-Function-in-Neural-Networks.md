---
title: Deep learning その 1、Activation Function
layout: default
tags: [machine learning,]
---



# Deep learning その 1: Activation Function

> 神经网络中，一个节点的**激活函数**定义了此节点输入与输出之间的映射关系。

### 0 Perception

神经网络（Neural Networks）与感知机（Perceptron）很大的区别，或者说改进就在于**激活函数**的不同。

简单介绍一下，感知机由![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Flag_of_the_United_States.svg/38px-Flag_of_the_United_States.svg.png)美国心理学家Frank Rosenblatt 于1958年提出，是一种处理二分类问题的线性模型，同时也是Logistics回归、SVM和神经网络的基础。模型如下：

![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Perceptron.svg/750px-Perceptron.svg.png)

对于线性不可分的数据，感知机模型无法收敛。因为感知机的激活函数$f(x)$是Sign函数是线性的。Sign函数，也称单位跃迁函数，常用形式如下：
$$
f(x)={\rm sign}(x)=\begin{cases}1,x>0\\0,x\leq0\end{cases}\tag{1}
$$
而神经网络常使用非线性函数（如Sigmoid、Tanh、ReLU等）作为神经元的激活函数。

为什么要使用非线性的激活函数，因为**若激活函数为线性，则无论神经网络有多少层，输出都是输入的线性组合，多层没有任何意义**。因此引入非线性的激活函数，神经网络才有理论上拟合任意函数的能力。

### 1 Sigmoid函数

Sigmoid函数，也称S型函数，常用形式如下：

$$
f(x)=\frac{1}{1+e^{-x}}\tag{2}
$$

Sigmoid函数将输出映射在[0, 1]，优点是梯度下降快，缺点是在输入非常大或非常小的时候容易出现梯度消失的情况。

### 2 Tanh函数

Tanh函数，即双曲正切函数，常用形式如下：
$$
f(x)=\tanh(x)=\frac{\sinh(x)}{\cosh(x)}=\frac{e^x-e^{-x}}{e^x+e^{-x}}\tag{3}
$$

Tanh函数与Sigmoid函数的区别是将输出映射在[-1, 1]。

### 3 ReLU函数

线性整流函数（Rectified Linear Unit）,也称斜坡函数，常用形式如下：
$$
f(x)=\max(0,x)\tag{4}
$$

现在广泛使用的是ReLU函数，其优点是收敛速度更快，避免了梯度消失，计算简单，缺点是比较脆弱。

### 4 Softmax函数

Softmax函数是Sigmoid函数的泛化，常用于多分类神经网络的最后一层。

Softmax函数的作用是将输出向量归一化，并且让归一化后的值大的更大、小的更小。

将一个元素为任意实数的$n$维向量$\vec{z}$压缩为一个同为$n$维的向量$\vec{a}$，向量$\vec{a}$中每一个元素范围为(0, 1)，且所有元素和为1。常用形式如下：

$$
a_i(\vec{z})=\cfrac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}，{\rm for}\quad i=1,\dots,n\tag{5}
$$

### Reference

\[1] [Wikipedia Activation Function](https://en.wikipedia.org/wiki/Activation_function)

\[2] [从感知机到深度神经网络-机器之心](https://www.jiqizhixin.com/articles/2018-01-15-2)

\[3] [深度学习-从线性到非线性-徐阿衡](http://www.shuang0420.com/2017/01/21/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-%E4%BB%8E%E7%BA%BF%E6%80%A7%E5%88%B0%E9%9D%9E%E7%BA%BF%E6%80%A7/)
---
title: NumPy Notes
layout: default
tags: [machine learning,]
---

# NumPy Notes

> 一些NumPy笔记 ，Updated on Nov 29, 2018

## 1 特征值&特征向量

对一个$n$阶方阵$A$，其一组特征值$\lambda$和特征向量$v$满足以下等式：$Av=\lambda v$。


在NumPy中计算方阵的特征值、特征向量的函数为:

``` python
# linear algebra eigen
numpy.linalg.eig(a)
```

返回值是两个array，依次为特征值，特征向量（**注意特征向量在返回值中以列存储**）。参见下面这段代码。

``` python
>>> a = np.arange(16).reshape(4,4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
>>> values, vectors = np.linalg.eig(a)
>>> values
array([ 3.24642492e+01, -2.46424920e+00,  1.92979794e-15, -4.09576009e-16])
>>> vectors
array([[-0.11417645, -0.7327781 ,  0.54500164,  0.00135151],
       [-0.3300046 , -0.28974835, -0.68602671,  0.40644504],
       [-0.54583275,  0.15328139, -0.2629515 , -0.8169446 ],
       [-0.76166089,  0.59631113,  0.40397657,  0.40914805]])
# check equaion
>>> A_Ve = np.dot(a,vectors[:,0])
>>> A_Ve
array([ -3.70665277, -10.71335153, -17.72005028, -24.72674904])
>>> lam_Ve = np.dot(values[0], vectors[:,0])
>>> lam_Ve
array([ -3.70665277, -10.71335153, -17.72005028, -24.72674904])
```



## 2 协方差矩阵

### 2.1 协方差

协方差（Covariance）用于衡量两个变量之间的总体误差，反映两变量间线性相关程度。

对随机变量$X=\{x_1,x_2,\cdots,x_n\}$和$Y=\{y_1,y_2,\cdots,y_n\}$，协方差${\rm Cov}(X,Y)$定义如下：

$$
\begin{aligned}
{\rm Cov}(X,Y)&={\rm E}[(X-{\rm E}[X])(Y-{\rm E}[Y])]=E[XY]-{\rm E}[X]{\rm E}[Y]\\
&=\frac{1}{n}\sum_{i=1}^n(x_i-\bar x)(y_i-\bar y)
\end{aligned}\tag{1}
$$
### 2.2 协方差矩阵

协方差矩阵（Covariance matrix）记录的是是多个变量，两两之间的协方差。以符号$\sum$表示。

其$i,j$位置的元素，即是第$i$个变量与第$j$个变量之间的协方差。公式定义如下：$\sum_{ij}=Cov(X_i,X_j)$。

以上就是协方差矩阵。

一般我们对于含有M个样本、N个 属性的数据，矩阵存储形状为$M\times N$。但在计算协方差矩阵时，**我们所关注的变量是属性而不是样本**，因此需要对矩阵进行转置。

在NumPy中计算协方差矩阵的函数为：

``` python
numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)
```

**注意参数ddof**，delta degree of freedom，这里默认值使用无偏估计，即自由度为$n-1$；若设定ddof=0，则计算的是样本平均，即自由度为$n$。参见下面这段代码。

``` python
>>> a = np.array([[1,2,3],[2,4,6]])
>>> a
array([[1, 2, 3],
       [2, 4, 6]])
>>> np.cov(a)
array([[1., 2.],
       [2., 4.]])
>>> np.cov(a, ddof=0)
array([[0.66666667, 1.33333333],
       [1.33333333, 2.66666667]])
```



## 3 矩阵的转置&行列式&逆

### 3.1 矩阵的转置（Transpose）

``` python
>>> a=np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> aT=a.T
>>> aT
array([[0, 3],
       [1, 4],
       [2, 5]])
```

### 3.2 矩阵的行列式（Determinant）

``` python
>>> a=np.arange(4).reshape(2,2)
>>> a
array([[0, 1],
       [2, 3]])
>>> np.linalg.det(a)
-2.0
```

### 3.3 矩阵的逆（Inverse）

对一个$n$阶方阵$A$，其逆矩阵$A^{-1}$满足以下等式：$AA^{-1}=A^{-1}A=I_n$，$I_n$为$n$阶单位矩阵。

``` python
>>> a=np.arange(4).reshape(2,2)
>>> a
array([[0, 1],
       [2, 3]])
>>> aI=np.linalg.inv(a)
>>> aI
array([[-1.5,  0.5],
       [ 1. ,  0. ]])
>>> np.dot(a,aI)
array([[1., 0.],
       [0., 1.]])
```



## 4 计算两点距离

两点构成一个向量，可以通过向量的范数来计算两点距离。

在NumPy中计算矩阵、向量范数的函数为：

```python
numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)
```

**默认值** ord=2，对应L2范数，即常用的欧式距离；ord=1 对应L1范数，即曼哈顿距离；ord=0 对应L0范数。

参见下面示例。

```python
>>> a = np.array([1, 1])
>>> b = np.array([4, 5])
>>> np.linalg.norm(a-b)
5.0
>>> np.linalg.norm(a-b, ord=0)
2.0
>>> np.linalg.norm(a-b, ord=1)
7.0
>>> np.linalg.norm(a-b, ord=2)
5.0
```



## 5 TBD

つづく
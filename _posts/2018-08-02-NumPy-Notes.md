---
title: NumPy Notes
layout: default
tags: [machine learning,]
---

# NumPy Notes
{:.no_toc}

> 一些NumPy笔记 ，Updated on Apr 27, 2019

* toc
{:toc}

## 1 切片&变形

取 numpy 矩阵的子行列

```python
>>> a = np.arange(16).reshape(4, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
>>> b = a[1:3][:,range(1, 3)]
>>> b
array([[ 5,  6],
       [ 9, 10]])
```

改变 numpy 矩阵 shape

```python
>>> a = np.arange(3).reshape(-1, 1)
>>> a
array([[0],
       [1],
       [2]])
>>> a.shape
(3, 1)
>>> b = a.transpose(1, 0)
>>> b
array([[0, 1, 2]])
>>> b.shape
(1, 3)
>>> c = a.squeeze()
>>> c
array([0, 1, 2])
>>> c.shape
(3,)
```



## 2 特征值&特征向量

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



## 3 均值&标准差

计算均值 mean 与标准差 standard deviation。

均值，$\mu=\cfrac{1}{n}\sum_{i=1}^nx_i$

标准差，$\sigma = \sqrt{\cfrac{1}{n-1}\sum_{i=1}^n(x_i-\mu)^2}$

$\star$ 注意 np.std() **默认使用有偏估计 ，ddof=0，即自由度为 $n$**；

若需要计算无偏估计，即自由度为 $n-1$，则要设定参数 ddof=1。

$\star$ 注意方差/标准差衡量的是变量的离散程度，所以会受到数据 scale 的影响。

```python
>>> np.mean([1,2,3])
2.0
>>> np.std([0,1])
0.5
>>> np.std([0,4])
2.0
>>> np.std([0,4], ddof=1)
2.8284271247461903
```



## 4 协方差矩阵

### 4.1 协方差

协方差（Covariance）用于衡量两个变量之间的总体误差，反映两变量间线性相关程度。

对随机变量$X=\{x_1,x_2,\cdots,x_n\}$ 和 $Y=\{y_1,y_2,\cdots,y_n\}$，协方差 ${\rm Cov}(X,Y)$ 定义如下：

$$
\begin{aligned}
{\rm Cov}(X,Y)&={\rm E}[(X-{\rm E}[X])(Y-{\rm E}[Y])]=E[XY]-{\rm E}[X]{\rm E}[Y]\\
&=\frac{1}{n}\sum_{i=1}^n(x_i-\bar x)(y_i-\bar y)
\end{aligned}
$$

### 4.2 协方差矩阵

协方差矩阵（Covariance matrix）记录的是是多个变量，两两之间的协方差。协方差矩阵以符号 $\sum$ 表示。

其$i,j$位置的元素，即是第 $i$ 个变量与第 $j$ 个变量之间的协方差。公式定义如下：$\sum_{ij}=Cov(X_i,X_j)$。

因此**对角线上的元素是方差**，对角线以外的元素才是狭义上的协方差。

在计算协方差矩阵时，**如果你所关注的变量是属性而不是样本，则需要对矩阵进行转置**。

在NumPy中计算协方差矩阵的函数为：

``` python
numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)
```

参数 ddof 是 delta degree of freedom的缩写，即自由度差。

$\star$ 注意 np.cov() **默认使用无偏估计 ，ddof=1，即自由度为 $n-1$**；

若需要计算无偏估计，即自由度为 $n-1$，则要设定参数 ddof=1。

参见下面这段代码。

``` python
>>> np.cov([[0,1], [0,4]])
array([[0.5, 2. ],
       [2. , 8. ]])
>>> np.cov([[0,1], [0,4]], ddof=0)
array([[0.25, 1.  ],
       [1.  , 4.  ]])
```



## 5 矩阵性质

### 5.1 矩阵的转置（Transpose）

计算矩阵矩阵 $A$ 的转置矩阵 $A^T$。

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

### 5.2 矩阵的秩（Rank）

矩阵 $A$ 的秩 $rank(A)$ 代表矩阵中线性无关向量的个数，行秩与列秩相等。

```python
a1 = np.array([[1, 2], [1, 2], [1, 2]])
r1 = np.linalg.matrix_rank(a1)
print(r1)
# 1
a2 = np.array([[1, 2], [1, 3], [1, 4]])
r2 = np.linalg.matrix_rank(a2)
print(r2)
# 2
```

### 5.3 矩阵的逆（Inverse）

可逆矩阵是针对**方阵**而言的。

对一个 $n$ 阶方阵 $A$，其逆矩阵 $A^{-1}$ 满足以下等式：$AA^{-1}=A^{-1}A=I_n$，$I_n$ 为 $n$ 阶单位矩阵。

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

### 5.4 矩阵的行列式（Determinant）

矩阵的行列式是针对**方阵**而言的。

对一个 $n$ 阶方阵 $A$，以下叙述等价：

- 矩阵 $A$ 的行列式 $\vert A \vert\neq0$
- 矩阵 $A$ 可逆
- 矩阵 $A$ 满秩，即 $rank(A)=n$
- 矩阵 $A$ 为非奇异矩阵（nonsingular matrix）

```python
>>> a=np.arange(4).reshape(2,2)
>>> a
array([[0, 1],
       [2, 3]])
>>> np.linalg.det(a)
-2.0
```



## 6 解线性方程组

使用Numpy解线性方程组 $AX=b$。

限制条件是矩阵 $A$ 必须为方阵，且为非奇异矩阵。

```python
a = np.array([[1, 2], [1, 3]])
b = np.array([1, 2])
x = np.linalg.solve(a,b)
print(x)
#[-1.  1.]
```



## 7 计算两点距离

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



## 8 TBD

つづく
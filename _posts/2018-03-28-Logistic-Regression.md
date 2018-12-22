---
title: Logistic Regression
layout: default
tags: [machine learning,]
---

# Logistic Regression



逻辑回归（LR）是机器学习中的经典分类方法。一句话介绍就是在线性回归的基础上套用 logistic 函数模型。



### 1 Logistic 回归模型

首先介绍Logistic 函数，由![Belgium](https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Flag_of_Belgium_%28civil%29.svg/38px-Flag_of_Belgium_%28civil%29.svg.png)比利时人 Pierre Verhulst 于1845年发现命名，用于模仿人口增长的曲线。其数学形式如下
$$
y=\frac{L}{1+e^{-k(x-x_0)}}\tag{1}
$$


Sigmoid 单词意为S型。机器学习中常用的激活函数 sigmoid 函数即泛指S型函数，一般特指 logistic 函数的特殊形式，下式$(2)$。其函数值在[0, 1]之间，数学形式及函数曲线如下
$$
y=\frac{1}{1+e^{-x}}\tag{2}
$$

![sigmod函数曲线](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png)



进入正题，Logistic 回归模型的数学形式如下

$$
p(y=1|x,\theta)=\frac{1}{1+e^{-\theta^Tx}}\tag{3}
$$
函数具有对称性质，$p(y=0|x,\theta)=1-p(y=1|x,\theta)$
Logistic 回归的输出是一个概率值，根据此概率值与阈值$μ$的大小进行分类，若 $p(y=1|x,\theta)>μ$，则$y^*=1$，$μ$一般取值0.5

因为线性回归残差服从高斯分布，所以可以使用最小二乘法进行参数求解；而 Logistic 回归的因变量、残差均为二项分布，不满足正态性，所以使用MLE为目标函数来进行参数$\theta$的求解。



### 2 最大似然估计（MLE）

最大似然估计Maximum Likelihood Estimation，是一个用来估计概率模型参数的方法，由![Britain](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Flag_of_the_United_Kingdom.svg/38px-Flag_of_the_United_Kingdom.svg.png)英国人 Ronald Fisher 于上世纪20年代推广。MLE的两个条件是已知数据和概率分布假设（似然函数），其思想是求得这个分布具有什么样的参数值才能使我们观测到这组数据的可能性最大。

假设有一组独立同分(i.i.d.)布的随机变量$X$，给定一个概率分布$D$，假设其概率密度函数为$f$，从这组样本中抽出$x_1,x_2,\cdots,x_n$，那么通过参数为$\theta$的模型$f$产生上面样本的概率为
$$
f(x_1,x_2,\cdots,x_n|\theta)=f(x_1|\theta)\times f(x_2|\theta)\times f(x_n|\theta)\tag{4}
$$

MLE寻找$\theta$最可能的取值使得这组样本出现的概率最大。根据样本估计参数$\theta$，定义**似然函数**为
$$
L(\theta|x_1,x_2,\cdots,x_n)=f(x_1,x_2,\cdots,x_n|\theta)=\prod_{i=1}^n f(x_i|\theta)\tag{5}
$$
取对数后，**对数似然函数**如下

$$
l(\theta)=\ln L(\theta|x_1,x_2,\cdots,x_n)=\sum_{i=1}^n f(x_i|\theta)\tag{6}
$$
取平均后，**平均对数似然函数**为
$$
\hat l=\frac{1}{n}\ln L\tag{7}
$$
最大似然估计即最大化$\hat l​$
$$
\hat \theta_{MLE}=\mathop{\arg\max}_\theta\hat l(\theta|x_1,x_2,\cdots,x_n)\tag{8}
$$

以上是MLE的通用范式，具体到 Logistic 回归，令$h(\theta^Tx)=\cfrac{1}{1+e^{-\theta^Tx}}$，则似然函数为
$$
L(\theta|x_1,x_2,\cdots,x_n)=\prod_{i=1}^n h(\theta^Tx_i)^{y_i}(1-h(\theta^Tx_i))^{1-y_i}\tag{9}
$$
取对数后，对数似然函数为
$$
\begin{aligned}
l(\theta)&= \sum_{i=1}^n y_i\ln h(\theta^Tx_i)+(1-y_i)\ln (1-h(\theta^Tx_i))\\
&= \sum_{i=1}^n y_i\ln\frac{h(\theta^Tx_i)}{1-h(\theta^Tx_i)}+\ln(1-h(\theta^Tx_i))\\
&= \sum_{i=1}^n -y_i\theta^Tx_i+\ln \frac{e^{\theta^Tx_i}}{1+e^{\theta^Tx_i}}
\end{aligned}\tag{10}
$$

在不引入正则项的情况下，Logistic 回归的**目标函数**是最小化负的似然函数，如下
$$
J(\theta)=-\frac{1}{n}l(\theta)\tag{11}
$$

$$
\theta^*=\mathop{\arg \min}_\theta J(\theta)\tag{12}
$$

由于多元变量很难求得解析解，使用梯度下降法逼近$\theta$的最优值。




### 3 梯度下降法

梯度下降法 Gradient Descent 是最优化算法的一种，据说由![France](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Flag_of_France.svg/38px-Flag_of_France.svg.png)法国人 Augustin Cauchy 于1847年提出。从函数当前点对应梯度的反方向按一定的步长$\alpha$进行迭代搜索，寻找函数$J(\theta)$的一个局部极小值点。若目标函数为凸函数，则得到的一定是全局最小值点。

GD 算法如下：

1）随机指定初始值$\theta^k$，$k=0$

2) 以全部数据为样本计算当前点$\theta^k$的梯度，即对每个$\theta_i$求偏导
$$
g^k_i=\frac{\partial J(\theta)}{\partial\theta_i}\tag{13}
$$
3) 更新$\theta$值，迭代公式如下
$$
\theta^{k+1}_i=\theta^k_i-\alpha g^k_i\tag{14}
$$

4) 若函数$J(\theta)$未收敛到极小值点，则令$k=k+1$，重复步骤2、3；若已收敛，则停止迭代。




### 4 调包淆

```python
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

def LR_demo():
    # load datasets
    iris = datasets.load_iris()
	X = iris.data[:, :2]  # use the first two features
	Y = iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    # fit model
	lr_model = linear_model.LogisticRegression(C=1e3)
	lr_model.fit(X_train, Y_train)
    
    # predict
	acc = lr_model.score(X_train,Y_train)
	prepro = lr_model.predict_proba(X_train)
    print acc, prepro
```





### Reference

\[1] 李航 (2012) 统计学习方法. 清华大学出版社, 北京.

\[2] [美团点评技术团队 logistic regression](https://tech.meituan.com/intro_to_logistic_regression.html)

\[3] [Wikipedia Logistic function](https://en.wikipedia.org/wiki/Logistic_function)

\[4] [Wikipedia 最尤推定](https://ja.wikipedia.org/wiki/%E6%9C%80%E5%B0%A4%E6%8E%A8%E5%AE%9A)

\[5] [最大似然估计与贝叶斯估计](https://blog.csdn.net/bitcarmanlee/article/details/52201858)

\[6] [梯度下降算法 理论基础](http://www.hanlongfei.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2015/07/29/gradient/)

\[7] [梯度下降算法 python实现](https://ctmakro.github.io/site/on_learning/gd.html)

\[8] [scikit-learn Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)


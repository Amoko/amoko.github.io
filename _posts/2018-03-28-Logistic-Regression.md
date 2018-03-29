# Logistic Regression



逻辑回归（LR）是机器学习中的经典分类方法。一句话介绍就是在线性回归的基础上套用 logistic 函数。



### 1 sigmoid 函数

logistic 函数由Pierre 于1845年发现命名，用于模仿人口增长的曲线。其数学形式如下
$$
y=\frac{L}{1+e^{-k(x-x_0)}}\tag{1}
$$


sigmoid 意为S型，sigmoid function 即为S型函数，一般指 logistic 函数的特殊形式，其数学形式如下
$$
y=\frac{1}{1+e^{-x}}\tag{2}
$$


公式$(2)$对应的 sigmoid 函数曲线如下：
<div align="center">
  ![sigmod函数曲线](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png)
</div>

sigmoid 函数的取值在[0, 1]之间，根据这个取值与阈值*θ*的大小关系进行分类，logistic回归的决策函数如下
$$
y^*=1, if P(y=1|x)>0.5\tag{3}
$$

### 2 最大似然估计

参数求解



### 3 梯度下降法

最优化问题



### Reference

\[1] [美团点评技术团队 logistic regression](https://tech.meituan.com/intro_to_logistic_regression.html)

\[2] [Wikipedia Logistic function](https://en.wikipedia.org/wiki/Logistic_function)

\[3] [hello kitty]()


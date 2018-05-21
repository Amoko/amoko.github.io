# Pareto Distribution

高斯分布 Gaussian Distribution 与帕累托分布 Pareto Distribution 并列为两大主导自然和人类世界的概率分布。



### 高斯分布

根据中心极限定理，大量独立随机变量的均值收敛于高斯分布。

比如男性（或女性）的身高分布、等等。高斯分布的概率密度函数及钟形函数曲线如下
$$
f(x|\mu,\sigma^2)=\cfrac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/525px-Normal_Distribution_PDF.svg.png)



### 帕累托分布

高斯分布的本质是独立性，而对于以社会性为本质的人类，某些属性在个体之间的独立性是不存在的。

所以在人类群体中，出现最多的往往是帕累托分布，比如个人财富、期刊引用量、社交媒体上的KOL，图书销售情况等。

帕累托分布，也称幂律分布，由 ![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Flag_of_Italy.svg/38px-Flag_of_Italy.svg.png) 意大利人Vilfredo Pareto于19世纪末提出。

其概率密度函数及函数曲线（J型）如下，$x_m$是$x$所能取到的最小值
$$
f(x|\alpha)=\frac {\alpha x_m^\alpha}{x^{\alpha +1}}
$$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Probability_density_function_of_Pareto_distribution.svg/488px-Probability_density_function_of_Pareto_distribution.svg.png)



对于帕累托分布的出现，美国社会学家 Robert Merton 于1968年提出了马太效应 Matthew Effect进行解释。

Robert认为，在正反馈机制作用下，个体所获得的初始优势会不断滚雪球，最终强者愈强，弱者愈弱，出现帕累托分布。

马太效应的名称源自圣经新约。

> For to every one who has will more be given, and he will have abundance; but from him who has not, even what he has will be taken away.
>
> —Matthew 25:29
>
> 因为凡有的，还要加给他，叫他有余；没有的，连他所有的也要夺过来。
>
> —马太福音 25:29



### 结合LiNGAM中的非高斯性?

LiNGAM中的非高斯性指的是变量还是噪声？体现在哪里。



### 路径依赖 v.s. 马尔科夫链

以社畜の作息为例，我们有三个变量 $S$: sleep early，$G$: get up early，$A$: arrive at office on time。

马尔科夫链如下
$$
S\rightarrow G\rightarrow A
$$

虽然$A$的状态是由$G$直接决定的，但从路径依赖的考虑，从$S$开始干预所付出的成本是最小的，更容易达成目标。

如果说做不到从$S$开始干预，也只能从$G$下手了。
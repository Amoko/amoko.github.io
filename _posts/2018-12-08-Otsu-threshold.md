---
title: Otsu の method
layout: default
tags: [machine learning, ]
---

# Otsu の method

大津（Otsu）算法是图像领域一个基础的二值化方法，由![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Flag_of_Japan.svg/38px-Flag_of_Japan.svg.png)日本人 <ruby>大津展之<rt>おおつのぶゆき</rt></ruby> 于1979年提出。

大津算法的输入为灰度图，在灰度图上求得一个自适应阈值，以此阈值为界将灰度图像二值化。



## 1 灰度图 & 二值图

对于灰度图，一个像素的存储空间为 8 bit，即取值空间为256，从黑到白共有256种亮度变化（0~255）。

对于二值图，一个像素的存储空间为 1 bit，即取值空间为2，只有黑、白两种取值。



下图为灰度图、二值图对比：

![](/img/8bit_1bit.png)

所以将灰度图二值化，也就是在灰色地带选一个阈值，将低于此值的归入黑色类，将高于此值的归入白色类。

Id est, from quantity to quality.

那么，这个阈值要怎么取才能更加公允呢？



## 2 大津算法

算法步骤很简单，**遍历所有灰度值，找到使类间方差最大的灰度值作为二值化的阈值**，完了。

现在问题只有一个，类间方差是什么？



大津算法所使用的类间方差定义，与Fisher线性判别（LDA）相同。

**类间方差**

阈值为 $t$ 时，类间方差 $\delta_t$ 定义如下：

$$
\begin{aligned}
\delta_t^2 &= w_0(\mu_0-\mu)^2 + w_1(\mu_1-\mu)^2 \\
&= w_0w_1(\mu_0-\mu_1)^2
\end{aligned}\tag{1}
$$

$p_i$，灰度值 $i$ 占比；$\mu=\sum_0^{255} ip_i$，全图灰度均值。

$w_0=\sum_0^{t}p_i$，黑色像素占比；$\mu_0=\cfrac{\sum_0^tip_i}{w_0}$，黑色像素灰度均值。

$w_1=\sum_{t+1}^{255}p_i$，白色像素占比；$\mu_1=\cfrac{\sum_{t+1}^{255}ip_i}{w_1}$，白色像素灰度均值。



大津法类间方差的计算，基于每个灰度值的占比。

因此计算灰度直方图后，对灰度直方图进行查表统计，就可以得到每个阈值下的类间方差。

![](/img/gray_histogram.png)



## 3 NumPy实现

笔者基于 NumPy 对大津法的一个简单实现：

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def get_otsu_value(grayImg):
    # get hist
    ravel = grayImg.ravel()
    hist, _ = np.histogram(ravel, range(0,257))
    sum_value = hist.sum()
    hist_p = [e/sum_value for e in hist]
    weight_p = []
    for i in range(256):
        weight_p.append(i*hist_p[i])
    hist_p = np.array(hist_p)
    weight_p = np.array(weight_p)
    
    # search
    the_max = [0, 0]
    mu = weight_p.sum()
    print("weighted mean:", mu)
    for i in range(0, 255):
        w0 = hist_p[:i].sum()
        w1 = hist_p[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = weight_p[:i].sum() / w0
        mu1 = weight_p[i:].sum() / w1
     
        delta = w0*(mu0-mu)**2 + w1*(mu1-mu)**2
        #delta2 = w0*w1*(mu0-mu1)**2
        #print(delta, delta2)
        if delta > the_max[1]:
            the_max = [i, delta]
    print("adjust thres, loss:", the_max)
    return the_max[0]

img  = cv.imread("bear.jpg")
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
thres = get_otsu_value(grayImg)
ret, thresImg = cv.threshold(grayImg, thres, 255, cv.THRESH_BINARY)
```



##  4 OpenCV中直接调用

在 OpenCV 中使用大津法进行二值化，只需在二值化函数中加入 <code>THRESH_OTSU</code> 参数即可。

可以查看 OpenCV 源代码，在二值化函数中加入 <code>THRESH_OTSU</code> 参数，实质为在二值化前调用大津算法求自适应阈值，**替换参数原阈值后**再进行二值化。

### 4.1 Python

```python
import cv2 as cv
import matplotlib.pyplot as plt

img  = cv.imread("bear.jpg")
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# threshold with otsu
ret, thresImg = cv.threshold(grayImg, 0, 255, cv.THRESH_OTSU)
print(ret)
plt.imshow(thresImg, cmap=plt.cm.gray)
plt.show()
```

### 4.2 C++

```c++
#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
	Mat img;
	img = imread("bear.jpg");
	Mat grayImg;
	cvtColor(img, grayImg, CV_BGR2GRAY);
	Mat biImg;
	// threshold with otsu
	threshold(grayImg, biImg, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("biImg", biImg);
	waitKey(0);
	return 0;
}
```

**tips**

据笔者在工程中的体验，大津算法的时间瓶颈在于统计灰度直方图步骤，而非阈值的遍历搜索步骤。



## Reference

\[1] [Wikipedia Grayscale](https://en.wikipedia.org/wiki/Grayscale)

\[2] [Wikipedia 大津算法](https://zh.wikipedia.org/wiki/%E5%A4%A7%E6%B4%A5%E7%AE%97%E6%B3%95)

\[3] [OpenCV thresholding turorial](https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html)


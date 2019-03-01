---
title: Integral Image
layout: default
tags: [machine learning, ]
---

## 1 积分图

对于需要多次计算图像子区域像素值之和的算法，都可以采用积分图进行加速。

积分图上每一点的值 $sum(x,y)$ 是原图对应位置左上角矩形的像素和。

而关于这个矩形是否包含边缘，并没有一个统一的标准，因此积分图的定义有两种格式。

**Wikipedia 的定义**：

$$
sum(x,y)=\sum_{x'\leq x,y'\leq y}i(x',y')
$$

inclusive，最终积分图尺寸为 $W \times H$。

**OpenCV  的定义**：

$$
sum(x,y)=\sum_{x'<x,y'<y}i(x',y')
$$

exclusive，最终积分图尺寸 $(W+1) \times (H+1)$。

## 2 OpenCV中的积分图

### 2.1 python

从skimage库中读取一张RGB图片，转化为灰度图后，求积分图。

灰度图每个像素所占存储空间为8位，而积分图需要32位。

```python
>>> from skimage import data
>>> import cv2 as cv
>>> img = data.astronaut()
>>> grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
>>> img_integral = cv.integral(grayImg)
>>> img.shape
(512, 512, 3)
>>> grayImg.shape
(512, 512)
>>> img_integral.shape
(513, 513)
>>> type(grayImg[0][0])
<class 'numpy.uint8'>
>>> type(img_integral[0][0])
<class 'numpy.int32'>
```

### 2.2 C++

```c++
integral(grayImg, inteImg);
```

输入、输出的数据类型都为Mat，Mat.data的数据类型为 unsigned char*

灰度图、积分图的每个元素的数据类型分别为CV_8U、CV_32S

**tips**

当使用一些第三方库对积分图进行加速优化时，需要特别注意 **积分图定义方式**，**步长*数据类型长度**。



## Reference

\[1] [Wikipedia Integral image](https://en.wikipedia.org/wiki/Summed-area_table)

\[2] [OpenCV docs Integral image](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#integral)

\[3] [積分画像の原理・計算式・高速化](https://algorithm.joho.info/image-processing/integral-image/)

\[4] [13行代码实现最快速最高效的积分图像算法](https://www.cnblogs.com/Imageshop/p/6219990.html)
---
title: OpenCV Configuration, Python & C++
layout: default
tags: [dev config,]
---

# OpenCV Configuration, Python & C++

## 1 Linux + Python

```shell
# 1 not using cv.imshow()
pip install opencv-python
# 2 using cv.imshow(), but can't read video files
conda remove opencv
conda install -c menpo opencv
```

## 2 Win 10 + Python

### 2.1 安装
>版本：Win 10 + Python 3.7 + OpenCV 3.4.3 
>
>时间：2018.11.17

1. 安装Anaconda3，[官网下载页面](https://www.anaconda.com/download/)；
2. 安装OpenCV，[UCI下载页面](https://www.lfd.uci.edu/~gohlke/pythonlibs/ )，选择opencv_python-3.4.3-cp37-cp37m-win_amd64.whl下载，使用pip本地安装即可。

### 2.2 Canny边缘检测demo

``` python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('tower.jpg')
edge = cv2.Canny(img,100,200)

plt.subplot(121)
plt.imshow(img)
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(edge)
plt.title('Edge Image')
plt.xticks([]), plt.yticks([])
plt.show()
```

## 3 Win 10 + C++

### 3.1 安装

>版本：Win 10 + VS 2013 + OpenCV 2.4.13
>
>时间：2018.11.17

OpenCV 发布页面，[OpenCV releases](https://opencv.org/releases.html)，选择2.4.13- winpack下载，双击解压即可。

### 3.2 配置项目

注意，我的 opencv-2.4.13.exe 解压目录为 <code>D:\opencv</code>，请根据你的目录修改路径。

1. 添加环境变量

   path中添加 <code> D:\opencv\build\x86\vc12\bin;</code>

2. 配置项目目录

   选择 Project - Properties - VC++ Directories 

   Include Directories 中添加 <code>D:\opencv\build\include;</code>

   Library Directories 中添加 <code>D:\opencv\build\x86\vc12\lib;</code>

3. 配置链接

   选择 Project - Properties - Linker - Input

   Additional Dependencies 中添加下列内容。（默认是debug模式，）

   ```
   opencv_ml2413d.lib
   opencv_calib3d2413d.lib
   opencv_contrib2413d.lib
   opencv_core2413d.lib
   opencv_features2d2413d.lib
   opencv_flann2413d.lib
   opencv_gpu2413d.lib
   opencv_highgui2413d.lib
   opencv_imgproc2413d.lib
   opencv_legacy2413d.lib
   opencv_objdetect2413d.lib
   opencv_ts2413d.lib
   opencv_video2413d.lib
   opencv_nonfree2413d.lib
   opencv_ocl2413d.lib
   opencv_photo2413d.lib
   opencv_stitching2413d.lib
   opencv_superres2413d.lib
   opencv_videostab2413d.lib
   ```


**tips**

- 项目编译通过，Debug阶段报错，“应用程序无法正常启动(0xc000007b)”。

  x86/x64平台问题，检查环境变量与项目配置中，所使用的是哪一个版本。

### 3.3 Canny边缘检测demo

```c++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

int main()
{
	Mat src, edge;
	src = imread("1.jpg");
    // Canny edge
    Canny(src, edge, 100, 200);
	imshow("Canny edge", edge);
	waitKey(0);
	return 0;
}
```



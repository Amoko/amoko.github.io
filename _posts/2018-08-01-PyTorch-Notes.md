---
title: PyTorch Notes
layout: default
tags: [machine learning,]
---

# PyTorch Notes

> Updated on Nov 19，2019

## 0 Install

国内安装需要添加清华源加速

```shell
# 添加清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
# 安装
conda install pytorch torchvision cudatoolkit=9.0
```



## 1 CrossEntropyLoss & Softmax

因为 PyTorch 在 CrossEntropyLoss 损失函数中整合了 Softmax 激活函数，所以对于多分类神经网络，最后一层不需要添加激活函数，只要设定神经元个数为类别个数即可。
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # the last layer needs no activation function
        x = self.fc3(x)
        return x
        
criterion = nn.CrossEntropyLoss()
net = Net()
outputs = net(inputs)
# calculate loss
loss = criterion(outputs, labels)
# predict
predicted = torch.max(outputs.data, 1)[1]
```

## 2 归一化

```python
from torchvision import transforms
transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
```

PyTorch在数据预处理时使用参数 transform 来定义对数据的归一化方式，将原始值域为 $[0,255]$ 的 numpy.ndarray 转换为值域为 $[0,1]$ 的Tensor。

这里面有两个操作，第一个<code>transforms.ToTensor()</code>是 Min-Max 归一化；

第二个<code>transforms.Normalize()</code>则是 Z-score 归一化，两组参数均值与标准差需要计算后赋值，因为定义的是单通道图像所以均值与标准差各只有1个。

参见下面这段代码：

```python
img = np.array([[0, 32], [128, 255]], np.uint8)
plt.imshow(img, cmap='gray')
plt.show()
# x' = x / 255
tensor = transforms.functional.to_tensor(img.reshape(2,2,1))
print(tensor)
'''
tensor([[[ 0.0000,  0.1255],
         [ 0.5020,  1.0000]]])
'''
# x' = (x - mean) / std
tensor = transforms.functional.normalize(tensor, (0.407,), (0.389,)) 
print(tensor)
'''
tensor([[[-1.0463, -0.7237],
         [ 0.2441,  1.5244]]])
'''
```


## 3 参数dim

PyTorch中的参数dim，就是NumPy中的参数axis，参考下面两个函数定义：

```python
# pytorch
torch.max(input, dim, keepdim=False, out=None)
torch.squeeze(input, dim=None, out=None)
# numpy
numpy.max(a, axis=None, out=None)
numpy.squeeze(a, axis=None)
```
对于二维矩阵，dim = 0 则以行（0维度）为轴，对各列进行“压缩”；dim = 1 则以列（1维度）为轴对各行进行“压缩”。参见下面这段代码：

```python
>>> a = torch.randint(0, 10, (2, 3))
>>> a
tensor([[ 5.,  2.,  9.],
        [ 7.,  3.,  2.]])
>>> b = torch.max(a, 0)
>>> b
(tensor([ 7.,  3.,  9.]), tensor([ 1,  1,  0]))
>>> c = torch.max(a, 1)
>>> c
(tensor([ 9.,  7.]), tensor([ 2,  0]))
```

## 4 TBC

人は運命にはさからえませんから。
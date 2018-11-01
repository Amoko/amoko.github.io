---
title：PyTorch Notes
tags: [machine learning]
---



# PyTorch Notes

> 一些PyTorch学习中的个人笔记。

updated in Nov 11，2018



## 1 CrossEntropyLoss & Softmax

因为PyTorch在CrossEntropyLoss损失函数中整合了Softmax激活函数，所以对于多分类神经网络，最后一层不需要添加激活函数，只要设定神经元个数为类别个数即可。
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




## 2 参数dim

在上一段代码中，注意到torch.max()中有一个参数值为1，这个参数是PyTorch中经常会用到dim，比如下面两个函数中：

```python
torch.max(input, dim, keepdim=False, out=None)
torch.squeeze(input, dim=None, out=None)
```
PyTorch中的参数dim，就是NumPy中的参数axis。

```python
numpy.max(a, axis=None, out=None)
numpy.squeeze(a, axis=None)
```
对于二维矩阵，dim = 0 则以行（0维度）为轴，对各列进行“压缩”；dim = 1 则以列（1维度）为轴对各行进行“压缩”。参见下面这段代码。

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



## 3 TBC

人は運命にはさからえませんから。
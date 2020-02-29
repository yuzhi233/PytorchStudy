# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:23:06 2020

@author: zhoubo
"""

# 下面的例子里我们创建一个高和宽为3的二维卷积层，
# 然后设输入高和宽两侧的填充数分别为1。给定一个高和宽为8的输入，我们发现输出的高和宽也是8
import torch
from torch import nn

# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.view((1, 1) + X.shape)#把生成的X变成 4维度  
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)#pytroch内置的卷积层

X = torch.rand(8,8)

print(comp_conv2d(conv2d, X).shape)

# 当卷积核的高和宽不同时，我们也可以通过设置高和宽上不同的填充数使输出和输入具有相同的高和宽。
# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))#在默认情况下，填充为0，步幅为1。
print(comp_conv2d(conv2d, X).shape)


#步幅/步长

# 下面我们令高和宽上的步幅均为2，从而使输入的高和宽减半。
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(conv2d.padding,conv2d.stride)
print(comp_conv2d(conv2d, X).shape)

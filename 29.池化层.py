# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:27:08 2020

@author: zhoubo
"""

#池化层
# 同卷积层一样，池化层每次对输入数据的一个固定形状窗口（又称池化窗口）中的元素计算输出。
# 不同于卷积层里计算输入和核的互相关性，池化层直接计算池化窗口内元素的最大值或者平均值。
# 该运算也分别叫做最大池化或平均池化。



# 让我们再次回到本节开始提到的物体边缘检测的例子
# 现在我们将卷积层的输出作为2×2最大池化的输入。
# 设该卷积层输入是X、池化层输出为Y。无论是X[i, j]和X[i, j+1]值不同，
# 还是X[i, j+1]和X[i, j+2]不同，池化层输出均有Y[i, j]=1。
# 也就是说，使用2×2最大池化层时，只要卷积层识别的模式在高和宽上移动不超过一个元素，我们依然可以将它检测出来。
import torch
from torch import nn

def pool2d(X,pool_size,mode='max'):#输入为二维X，元组poo_size
    X=X.float()#数据转成torch.float
    p_h,p_w =pool_size
    Y = torch.zeros(X.shape[0]-p_h+1,X.shape[1]-p_w+1)#二维数组的池化后大小计算
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode =='max':
                Y[i,j]=X[i:i+p_h,j:j+p_w].max()#最大池化
            elif mode =='avg':
                Y[i,j]=X[i:i+p_h,j:j+p_w].mean()#平均池化
    return Y


#验证一下：
X =torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
print(X)

Y_maxpool =pool2d(X,(2,2),'max')
Y_avgpool =pool2d(X,(2,2),'avg')

print(Y_maxpool)
print(Y_avgpool)

#%%填充和步幅
import torch
from torch import nn
# 同卷积层一样，池化层也可以在输入的高和宽两侧的填充并调整窗口的移动步幅来改变输出形状
# 池化层填充和步幅与卷积层填充和步幅的工作机制一样。
# 我们将通过nn模块里的二维最大池化层MaxPool2d来演示池化层填充和步幅的工作机制。
# 我们先构造一个形状为(1, 1, 4, 4)的输入数据，前两个维度分别是   批量  和  通道  。

X =torch.arange(16,dtype=torch.float).view((1,1,4,4))
print(X)# (1,1,4,4)

# 默认情况下，MaxPool2d实例里步幅和池化窗口形状相同。
# 下面使用形状为(3, 3)的池化窗口，默认获得形状为(3, 3)的步幅。

pool2d =nn.MaxPool2d(3)#3*3池化窗口 （默认步幅和窗口形状相同）
print(pool2d(X))

# 我们可以手动指定步幅和填充。

pool2d =nn.MaxPool2d(3,padding =1,stride=2)#池化窗口3*3，padding为1 步长2
print(pool2d(X))


#也可以指定非正方形池化窗口
pool2d2 = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d2(X))


#池化多通道
X=torch.cat((X,X+1),dim =1)

pool2d =nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))


# 最大池化和平均池化分别取池化窗口中输入元素的最大值和平均值作为输出。
# 池化层的一个主要作用是缓解卷积层对位置的过度敏感性。
# 可以指定池化层的填充和步幅。
# 池化层的输出通道数跟输入通道数相同。


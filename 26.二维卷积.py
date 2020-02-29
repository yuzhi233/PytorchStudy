# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:18:16 2020

@author: zhoubo
"""

#%%二维卷积 卷积操作 以及卷积层创建

import torch
from torch import nn as nn

# 接受输入数组X与卷积核数组K，并输出数组Y

def corrd2d(X,K):#X---需要进行卷积的二维数组   K------卷积核张量数组
    h,w =K.shape
    Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))#初始化y  Y是卷积后的二维tensor  这里步长应该默认是1 padding =0
    for i in range(Y.shape[0]):#遍历Y每个元素 (填经过卷积后的值)
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()#截取了一小块做卷积 这里*K 是广播运算 对于元素相乘 算出了一次卷积的的结果
            
    return Y

#验证
    
X1 =torch.tensor([[1,2,3],
                 [4,5,6],
                 [7,8,9]])

Kernel =torch.tensor([[1,1],
                      [1,1]])
Y1 = corrd2d(X1,Kernel)
print(Y1)

#下面基于corr2d函数来实现一个自定义的二维卷积层。

class Conv2D(nn.Module):
    def __init__(self,kernel_size):#初始化实例要传入卷积核大小 kernel_size=(H,W)行，列
        super(Conv2D,self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))#随机生成卷积核数组 （高斯分布）
        self.bias =nn.Parameter(torch.randn(1))#生成偏差
    def forward(self,X):
        return corrd2d(X,self.weight)+self.bias


# 卷积窗口形状为p×q的卷积层称为p×q卷积层。同样，p×q卷积或p×q卷积核说明卷积核的高和宽分别为p和q
        
    
# 图像种物体边缘检测
# 检测图像中物体的边缘，即找到像素变化的位置。
# 首先我们构造一张6×86×8的图像（即高和宽分别为6像素和8像素的图像）。它中间4列为黑（0），其余为白（1）。


X =torch.ones(6,8)
X[:,2:-2]=0
print(X)
# 然后我们构造一个高和宽分别为1和2的卷积核K。当它与输入做互相关运算时，如果横向相邻元素相同，输出为0；否则输出为非0。
K=torch.tensor([[1,-1]])#注意这里卷积核必须得是2维度的！ 不要以为这一行 是一维！
# 下面将输入X和我们设计的卷积核K做互相关运算。可以看出，我们将从白到黑的边缘和从黑到白的边缘分别检测成了1和-1。其余部分的输出全是0。

Y=corrd2d(X,K)
print(Y)

# 由此，我们可以看出，卷积层可通过重复使用卷积核有效地表征局部空间。

# --------------------------通过数据学习核数组---------------------------------

# 使用物体边缘检测中的输入数据X和输出数据Y来学习我们构造的核数组K
# 我们首先构造一个卷积层，其卷积核将被初始化成随机数组。
# 接下来在每一次迭代中，我们使用平方误差来比较Y和卷积层的输出，然后计算梯度来更新权重。



#构造一个卷积核数组形状是(1,2)的二维卷积层
conv2d =Conv2D(kernel_size=(1,2))# 这里的conv2d是我们自己创建的一个Conv2D实例 自定义的不是pytorch定义的
step =20
lr =0.01
for i in range(step):
    Y_hat =conv2d(X)
    l =((Y_hat-Y)**2).sum() # 使用平方误差来比较Y和卷积层的输出
    l.backward()
    
    #梯度下降  一定要先下降再清零
    conv2d.weight.data -= lr*conv2d.weight.grad
    conv2d.bias.data -= lr*conv2d.bias.grad
    
    
    
    #梯度清零
    conv2d.weight.grad.fill_(0)#还记得不 像这种.fill_()后面带个_的 都是pytorch中的inplace操作
    conv2d.bias.grad.fill_(0)
    
    
    
    
    if (i + 1) % 5 == 0:#每5次打印一次
        print('step %d，loss %.3f'%(i+1,l.item()))
    
print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)
    
# 可以看到，学到的卷积核的权重参数与我们之前定义的核数组K较接近，而偏置参数接近0。
    
    
    
    
    
    
    
    



































# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:04:21 2020

@author: zhoubo
"""

import torch
from torch import nn
from torch.nn import init 
import numpy as np
import sys
import d2lzh_pytorch as d2l

#仍然采用FashionMNIST 数据集  

#设置batch size
batch_size =256#一次读取256张图片

#读取数据
train_iter,test_iter= d2l.load_data_fashion_mnist(batch_size)

#定义和初始化模型

num_inputs = 784 #一张图片是28*28=784  个特征
num_outputs = 10 #共有10个分类结果 
#===============================如果自己写线性网络的话===============================
class LinearNet(nn.Module):#继承父类nn.Module
    #构造函数
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.linear =nn.Linear(num_inputs,num_outputs)
       
    #前向传播函数
    def forward(self,X): # x shape: (batch, 1, 28, 28)
        y = self.linear(X.view(x.shape[0],-1))#X矩阵转换成256行 （自动推断）列 
        return y


# # FlattenLayer层 用于改变x的形状
# class FlattenLayer(nn.Module):
#     def __init__(self):
#         super(FlattenLayer, self).__init__()
#     def forward(self, x): # x shape: (batch, *, *, ...)
#         return x.view(x.shape[0], -1)#batch行 自动推断列


#==========================用sequential容器生成的话===============================
        
    
# net =LinearNet(num_inputs,num_outputs)   #实例化一个对象 我们自己的线性网络类 
# print(net)
from collections import OrderedDict

net =nn.Sequential(OrderedDict([('flatten',d2l.FlattenLayer()),
                                ('linear',nn.Linear(num_inputs,num_outputs)
                                  )])
                              )
print(net)
   
 
# 初始化权重参数
init.normal_(net.linear.weight,mean =0,std =0.01)#均值为0 方差为0.01
init.constant_(net.linear.bias, val=0)

# 定义损失函数
# softmax 和交叉熵损失函数 分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定。
# 因此，PyTorch提供了一个包括softmax运算和交叉熵损失计算的函数。它的数值稳定性更好。
loss =nn.CrossEntropyLoss()


#定义优化算法
optimizer =torch.optim.SGD(net.parameters(),lr =0.1)#学习率为0.1的小批量随机梯度下降算法


# 模型训练

num_epochs =5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)

























    
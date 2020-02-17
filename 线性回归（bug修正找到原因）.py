# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:03:39 2020

@author: zhoubo
"""

#%%线性回归

import torch
from matplotlib import pyplot as plt
import numpy as np
import random
from IPython import display

#生成数据集

#设训练集样本数为1000  输入特征数为2
#给定随机生成的批量样本特征X是1000*2
#使用线性回归模型真实权重w=[2,-3.4](竖着)，和偏差 b =4.2和随机噪声e生成标签y =xw+b+e
#其中噪声服从均值为0 方差为0.01的正态分布 

num_inputs = 2 #输入的特征数
num_examples = 1000 #样本个数 

true_w = [2, -3.4]#真实的权值
true_b = 4.2#真实的bias

#对特征矩阵初始化
features = torch.randn((num_examples,num_inputs))


# print(features.size())
#制作标签 是真实的y值+噪音
labels = true_w[0]*features[:,0] + true_w[1]*features[:,1]+true_b#先算出真实的 y值

labels +=torch.tensor(np.random.normal(0, 0.01, size=labels.size()),dtype=torch.float32)
print(features[0], labels[0])#查看featurs的第一行 和其对应的label

#=========================图像展示部分===================================
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize =(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] =figsize

set_figsize()
plt.scatter(features[:,0].numpy(),labels.numpy())
#========================================================================



#数据读取
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices =list(range(num_examples))#生成跟样本个数一样的序列当作索引值[0-1000]
    random.shuffle(indices)#打乱这些索引值
    for i in range(0,num_examples,batch_size):#i每等于一个batch_size循环一次
        
        j =torch.LongTensor(indices[i:min(i+batch_size,num_examples)])#抽取一个batch_size的情况下，的索引个数，并转换成LongTensor 其实就是一个批次内的索引列表里面的索引是随机的数，那这个索引对feature和labels取值
        
        yield features.index_select(0,j).float(),labels.index_select(0,j)#暂时先将yeild看成return
    
batch_size = 10#定义一次读取的批量大小  为10个数据

#看一下我们的函数是不是实现了功能
# for X,y in data_iter(batch_size,features,labels):
#     print(X,y)
#     break

#模型参数初始化
w =torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype =torch.float32)   
b =torch.zeros(1,dtype =torch.float32)    

w.requires_grad_(requires_grad =True)
b.requires_grad_(requires_grad =True)


#定义模型
def linreg(X,w,b):#线性回归模型
    return torch.mm(X,w)+b#矩阵乘法
#定义损失函数 #平方损失函数 
def squared_loss(y_hat ,y):
    return(  y_hat -y.view(  y_hat.size()  )   )**2/2#这里返回的是向量！ pytorch的MESLoss没有除以2


#定义优化算法
def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size

#训练模型
lr = 0.03#学习率
num_epochs=3#3个迭代周期

net = linreg
loss = squared_loss

for epoch in range(num_epochs):
        #每个迭代周期中，会使用训练数据集中所有样本一次
    for X,y in data_iter(batch_size,features,labels):
        l =loss(net(X,w,b),y).sum()#计算损失值
        l.backward()#反向传播求梯度
        sgd([w,b],lr,batch_size)
        
        #梯度清0
        w.grad.data.zero_()
        b.grad.data.zero_()
    
    
    train_l =loss(net(features,w,b),labels)#一轮迭代后计算一家损失
    print('epoch{},loss{:4f}'.format(epoch+1,train_l.mean().item()))
    
print(w,b)#可以看到3轮迭代周期后  w的预测值 和b的   与真实对比非常接近了！



































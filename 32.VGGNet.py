# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:35:55 2020

@author: zhoubo
"""

#  使用重复元素的网络（VGG）
# AlexNet在LeNet的基础上增加了3个卷积层
# 但AlexNet作者对它们的卷积窗口、输出通道数和构造顺序均做了大量的调整
# 虽然AlexNet指明了深度卷积神经网络可以取得出色的结果，但并没有提供简单的规则以指导后来的研究者如何设计新的网络

# VGG块的组成规律是：连续使用数个相同的填充为1、窗口形状为3×3的卷积层后接上一个步幅为2、窗口形状为2×2的最大池化层。
# 卷积层保持输入的高和宽不变，而池化层则对其减半。


# 我们使用vgg_block函数来实现这个基础的VGG块，它可以指定卷积层的数量和输入输出通道数。

import time
import torch
from torch import nn, optim
import d2lzh_pytorch as d2l

#先定义一下可以用于训练的设备
device =torch.device('cuda' if torch.cuda.is_available()else 'cpu')

#定义一个VGG块，像搭积木一样 一个块有num_covns个卷积层（padding=1 kernelsize=3*3）,和一个2*2最大池化层
def vgg_block(num_convs,int_channels,out_channels):#卷积层的个数，输入的通道数，输出的通道数
    blk=[]#blanket?空白列表
    for i in range(num_convs):
        if i ==0:
            blk.append(nn.Conv2d(int_channels,out_channels,kernel_size=3,padding=1))
        else:
            blk.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2,stride=2))# 这里会使宽高减半
    return nn.Sequential(*blk)#解包放进Sequential做参数 返回的是一个block的Sequential容器


#开始构建VGG网络
# 与AlexNet和LeNet一样，VGG网络由卷积层模块后接全连接层模块构成。
# 卷积层模块串联数个vgg_block，其超参数由变量conv_arch定义。该变量指定了每个VGG块里卷积层个数和输入输出通道数。全连接模块则跟AlexNet中的一样。
    
conv_arch=((1,1,64),(1,64,128),(2,128,256),(2,256,512),(2, 512, 512))#经过5个VGG块宽高减半5次 224/32=7

fc_features =512*7*7
fc_hidden_units=4096#任意


def vgg(conv_arch,fc_features,fc_hidden_units=4096):
    #创建一个大的 sequential 
    net =nn.Sequential()
        #卷积部分
    for i ,(num_convs,int_channels,out_channels) in enumerate(conv_arch):
            #每经过一个block会使宽高减半
        net.add_module('vgg_block_'+str(i+1),vgg_block(num_convs, int_channels, out_channels))#前面是名字，后面是具体是什么module
        
        #全连接部分(大容器放 全连接层小容器)
    net.add_module('fc',nn.Sequential(
        d2l.FlattenLayer(),#这里注意了！！不要忘了加！
        nn.Linear(fc_features,fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units,fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units,10),
        nn.Softmax(dim =1)

        ))
    return net


#构造一个宽和高都是224的单通道数据样本 来测试一下我搭建的网络
    
# net =vgg(conv_arch,fc_features,fc_hidden_units)#

# X =torch.rand(1,1,224,224)#创建 批次 C H W  生成一个测试数据

# # named_children获取一级子模块及其名字
# (named_modules会返回所有子模块,包括子模块的子模块)

net = vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1, 1, 224, 224)

# named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
for name, blk in net.named_children(): 
    X = blk(X)
    print(name, 'output shape: ', X.shape)

    
# 可以看到，每次我们将输入的高和宽减半，直到最终高和宽变成7后传入全连接层。与此同时，输出通道数每次翻倍，直到变成512。因为每个卷积层的窗口大小一样，所以每层的模型参数尺寸和计算复杂度与输入高、输入宽、输入通道数和输出通道数的乘积成正比。VGG这种高和宽减半以及通道翻倍的设计使得多数卷积层都有相同的模型参数尺寸和计算复杂度。
    
  
# 因为VGG-11计算上比AlexNet更加复杂，出于测试的目的我们构造一个通道数更小，或者说更窄的网络在Fashion-MNIST数据集上进行训练。
    
ratio =8
small_conv_arch=[(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), 
                   (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]

net= vgg(small_conv_arch,fc_features//ratio,fc_hidden_units//ratio)

print(net)

batch_size = 1
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)




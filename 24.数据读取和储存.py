# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:49:03 2020

@author: zhoubo
"""

#读取和储存

# 到目前为止，我们介绍了如何处理数据以及如何构建、训练和测试深度学习模型。
# 然而在实际中，我们有时需要把训练好的模型部署到很多不同的设备。在这种情况下，我们可以把内存中训练好的模型参数存储在硬盘上供后续读取使用。

#%%读写tensor

import torch 
from torch import nn

#储存一个tensor
x=torch.ones(3)
torch.save(x,'x.pt')#储存 
x2 =torch.load('x.pt')#读取
print(x2)

#可以储存一个tensor 列表 并读回内存
y= torch.zeros(4)
torch.save([x,y],'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

# 存储并读取一个从字符串映射到Tensor的字典
torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)


#%%读写模型

# 在PyTorch中，Module的可学习参数(即权重和偏差)，模块模型包含在参数中(通过model.parameters()访问)。
# state_dict是一个从参数名称隐射到参数Tesnor的字典对象。
import torch 
from torch import nn
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print('查看一下net.state_dict()是什么:\n',net.state_dict())#state_dict是一个从参数名称隐射到参数Tesnor的字典对象


#注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目

# 优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print('优化器参数dict：\n',optimizer.state_dict())
print('*'*100)

# 保存和加载模型

# PyTorch中保存和加载训练模型有两种常见的方法:

# 仅保存和加载模型参数(state_dict)；
# 保存和加载整个模型。
#可以参考https://blog.csdn.net/qq_29893385/article/details/84644478

#------------------------------ 一. 保存和加载state_dict(推荐方式)---------------------------------------------

#1.保存

torch.save(net.state_dict(),'./net_justmodelparams.pt')  ##保存整个模型参数  推荐的文件后缀名是pt或pth

#2.加载模型参数
#先构造出架构-----------------你重新搭建的框架 名字得跟之前模型的一致才能读取！！！大坑

net2 =nn.Sequential(OrderedDict([   ('hidden',torch.nn.Linear(3,2)),
                         ('act',torch.nn.ReLU()),
                    ('output',torch.nn.Linear(2,1))
                    ]))
print(net2)
print('net没载入之前的模型参数:\n',list(net2.parameters()))
#再传参
net2.load_state_dict(torch.load('./net_justmodelparams.pt'))#加载
print('net载入之后的模型参数:\n',list(net2.parameters()))

X=torch.ones(1,3)
print('net的计算结果:\n',net(X))
print('net2计算结果:\n',net2(X))#成功!

#--------------------------------二.保存和加载整个模型---------------------------------------

# 1.保存
torch.save(net,'./net_wholeparams.pt')

# 2.加载
net3 =torch.load('./net_wholeparams.pt')
print('加载整个模型的net3:\n',net3)
print('net3计算结果:\n',net3(X))




#=================================================注意============================================
# 对于两种不同的方法定义不同函数进行模型的载入

# 对于整个模型的保存载入非常简单,直接使用torch.load(PATH),可以直接载入,但是不推荐这种方法,一是对于大量数据集训练迭代次数一般很多(50000...)
# 所以模型一般会非常大,更重要的是这种保存的模型泛化性极其之差.


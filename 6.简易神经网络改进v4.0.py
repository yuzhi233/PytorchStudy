# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:57:09 2020

@author: zhoubo
"""

#简易神经网络改进v4.0
#到目前为止，代码中的神经网络权重的参数优化和l更新还没有实现自动化
#并且目前使用的优化方法都有固定的学习速率，所以优化函数相对简单

import torch
from torch.autograd import Variable

batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

x = Variable(torch.randn(batch_n,input_data),requires_grad = False)
y = Variable(torch.randn(batch_n,output_data),requires_grad = False)

#实例化一个模型
models =torch.nn.Sequential(torch.nn.Linear(input_data,hidden_layer),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_layer,output_data))

#定义学习率，迭代次数
epoch_n =1000
learning_rate =1e-4

#定义损失函数
loss_fn =torch.nn.MSELoss()#定义损失函数为均方差误差

#定义优化器

#注意这里  使用了torch.optim 包中的torch .optim.Adam 类作为我们的模型参数的优化函数
#在torch.optim.Adam 类中输入的是被优化的参数和学习速率的初始值，如果没有输入学习
#速率的初始值，那么默认使用0.001 这个值。
optimzer = torch.optim.Adam(models.parameters(),lr =learning_rate)



#迭代循环体
for epoch in range(epoch_n):
    #计算预测值
    y_pred = models(x)
    
    #计算loss
    loss = loss_fn(y_pred,y)
    if epoch%100 == 0:
        print('Epoch{},Loss:{:.4f}'.format(epoch,loss))#打印每轮误差值
 
    #优化器先将参数置0
    optimzer.zero_grad()
    
    #误差反向传播
    loss.backward()
    
    optimzer.step()#用计算得到的梯度值对各个节点的参数进行梯度更
# 新

#在以上代码中有几处代码和之前的训练代码不同，这是因为我们引入了优化算法，所
# 以通过直接调用optimzer. zero_grad() 来完成对模型参数梯度的归零；并且在以上代码中增
# 加了optimzer.step() ，它的主要功能是使用计算得到的梯度值对各个节点的参数进行梯度更
# 新
    
    
    
    
    
#发现使用Adam法进行参数和学习率更新梯度下降更快收敛
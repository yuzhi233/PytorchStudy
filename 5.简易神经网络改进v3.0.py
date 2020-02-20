# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:40:31 2020

@author: zhoubo
"""
#简易神经网络再改进 

#针对参数矩阵的生成，损失函数计算，权值更新  改进
import torch
from torch.autograd import Variable

batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

x = Variable(torch.randn(batch_n,input_data),requires_grad = False)
y = Variable(torch.randn(batch_n,output_data),requires_grad = False)


#torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。
#另外，也可以传入一个有序模块

#默认传入嵌套  层会按0，1，2...命名
models =torch.nn.Sequential(torch.nn.Linear(input_data,hidden_layer),#输入层
                            torch.nn.ReLU(), #Relu层
                            torch.nn.Linear(hidden_layer,output_data))#输出层
print(models)

#--------------------------如果传入的是字典----------------------------------

# from collections import OrderedDict

# models2 = torch.nn.Sequential(OrderedDict([('Liner1层',torch.nn.Linear(input_data,hidden_layer)),
#                                           ('ReLU  层', torch.nn.ReLU()),
#                                           ('Liner2层',torch.nn.Linear(hidden_layer,output_data))]))
# print(models2)
#-----------------------------------------------------------------------------
epoch_n =10001
learning_rate =1e-4

#现在使用的是在torch.nn 包中己经定义好的均方误差函数类torch.nn.MSELoss 来计算损失值，而之前的代码是根据损失函数的计算公式来编写的。
loss_fn = torch.nn.MSELoss()

for epoch in range(epoch_n):
    y_pred =models(x)
    
    loss =loss_fn(y_pred,y)
    
    if epoch%1000 ==0:
        print('Epoch:{},Loss:{:.4f}'.format(epoch,loss.data))
    
    models.zero_grad()#将这轮梯度值先置为0
    loss.backward()#反向传播计算梯度

    for param in models.parameters():#参数更新
        param.data -= param.grad.data*learning_rate

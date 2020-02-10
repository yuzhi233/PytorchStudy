# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:33:49 2020

@author: zhoubo
"""
#%% 小型神经网络的改进v1.1
import torch
from torch.autograd import Variable#导入变量

torch.manual_seed(1)#设置一个随机数种子 以便于复现结果

batch_n =100#一个批次中输入的数据的数量 100个数据
hidden_layer =100#隐藏层特征数100，神经元个数
input_data = 1000#输入的样本特征是1000（对应1000个神经元）
output_data =10 #输出的样本特征10（对应10个神经元）,最后的分类结果数



# 在以上代码中还使用了一个
# requ ir es _grad 参数，这个参数的赋值类型是布尔型，如果requires_grad 的值是False ，那么
# 表示该变量在进行自'i))J 梯度计算的过程中不会保留梯度值。


x = Variable(torch.randn(batch_n,input_data),requires_grad = False)#将创建的tensor数据封装在Variable中，False不需要对x求解梯度，因为x是输入的数据
y = Variable(torch.randn(batch_n,output_data),requires_grad =False)#同理y是我们生成样本x对应的真实值，所以不需要求梯度

w1 = Variable(torch.randn(input_data,hidden_layer),requires_grad =True)
w2 = Variable(torch.randn(hidden_layer,output_data),requires_grad =True)


#定义训练轮数和学习率
epoch_n = 10000
learning_rates =0.000001

for epoch in range(epoch_n):  
    #计算y_pred  x的预测值y_pred
    h1 =torch.mm(x,w1)#将x，与w1做矩阵乘法得到h1 隐藏层矩阵  h1 应该是100*100
    h1 =h1.clamp(min =0) #将h1矩阵中小于0的变为0 #相当于RELU了有那意思
    y_pred =torch.mm(h1,w2)#计算出我们100个样本预测的值 100*10
    #y_pred = x.mm(w1).clamp(min =0).mm(w2)
    
    
    #计算损失
    loss =(y_pred-y).pow(2).sum()#求解该批次的总损失值
    print('Epoch:{},Loss:{:.4f}'.format(epoch,loss.data))
    
    loss.backward()
    # print('w1\n',w1)
    #print('wq.data:\n',w1.data)
    w1.data -= learning_rates*w1.grad.data
    w2.data -= learning_rates*w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()

























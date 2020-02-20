# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:51:55 2020

@author: zhoubo
"""

#多层感知机的简洁实现

import torch 
from torch import nn
import numpy as np
import d2lzh_pytorch as d2l
from torch.nn import init
import torch.utils.data as Data





#定义训练batch

batch_size =256#定义训练batch




#装载数据
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)#调用d2l包函数






#定义模型  
num_inputs,num_hiddens,num_outputs = 784,256,10

net =nn.Sequential(d2l.FlattenLayer(),
                   nn.Linear(num_inputs,num_hiddens),
                   nn.ReLU(),
                   nn.Linear(num_hiddens,num_outputs)
                   )

print(net)#查看我们定义的模型


#初始化参数(权值，偏差)

for params in net.parameters():#sequential容器可以用.parameters()方法获取到参数
    init.normal_(params,mean =0,std=0.01)#所有参数初始化成均值为 0  标准差0.01的数值
    

#定义损失函数
loss =torch.nn.CrossEntropyLoss()

#定义优化算法
optimizer =torch.optim.SGD(net.parameters(),lr=0.1)


#训练模型
num_epochs =5

# # for X,y in train_iter:
# #     print(y.shape[0] , y.shape)
# #     break

# test_acc =d2l.evaluate_accuracy(test_iter,net)
# print(test_acc)

for epoch in range(num_epochs):
    
    train_l_sum=0.0#训练集总loss
    train_acc_sum =0.0#训练集预测正确的个数
    n=0
    #先取出数据
    for X,y in train_iter:
        #计算y_hat
        y_hat =net(X)
        #计算loss
        l =loss(y_hat,y).sum()
        
        
        #梯度清零
        optimizer.zero_grad()
        
        l.backward()#误差反向传播
        
        # 参数更新
        optimizer.step()
    
        train_l_sum +=l.item() #计算一轮训练下来的训练集总损失
        train_acc_sum +=(y_hat.argmax(dim=1)==y).sum().item()#计算y_hat中预测正确的个数
        n+=y.shape[0]#shape是torch.size[]，加上索引才能把值拿出来
    
    # 一个批次训练后 计算测试集的准确率
    test_acc =d2l.evaluate_accuracy(test_iter,net)
    
    print('epoch{},loss{:.4f},train acc{:.3f},test acc{:.3f}'.format(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))  

        
    
# 结果：
# epoch1,loss0.0041,train acc0.633,test acc0.768
# epoch2,loss0.0023,train acc0.792,test acc0.795
# epoch3,loss0.0020,train acc0.821,test acc0.811
# epoch4,loss0.0019,train acc0.830,test acc0.803
# epoch5,loss0.0018,train acc0.840,test acc0.828







    
   
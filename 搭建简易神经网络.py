# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:12:23 2020

@author: zhoubo
"""

#搭建一个简单的神经网络

import torch


batch_n = 100#一个批次中输入的数据的数量 100个数据
hidden_layer =100#定义隐藏层特征数量100
input_data = 1000#输入的数据的特征数量 1000个特征
output_data =10#输出数据的特征数量 10个特征  最后得到10个分类结果
 


x =torch.randn(batch_n,input_data)#生成100个样本，每个样本有1000个特征
# print(x)
y = torch.randn(batch_n,output_data)#生成样本的y值，每个样本y值有10个特征
# print(y)

#初始化权值矩阵
w1 =torch.randn(input_data,hidden_layer)#从输入层到隐藏层的w1权值矩阵w1 是1000*100矩阵
w2 =torch.randn(hidden_layer,output_data)#从隐藏层到输出层的权值矩阵w2 是100*10

epoch_n =[i for i in range(0,2000,100)] #定义训练的次数20次

learning_rate =0.0000001#学习率初始化为0.00000001
 
loss_axis =[0]

for i in range(len(epoch_n)):#对于不同的迭代次数的最终loss
    
    for epoch in range(epoch_n[i]):
        
        h1 =torch.mm(x,w1)#将x，与w1做矩阵乘法得到h1 隐藏层矩阵  h1 应该是100*100
        h1 =h1.clamp(min =0) #将h1矩阵中小于0的变为0 
        y_pred =torch.mm(h1,w2)#计算出我们100个样本预测的值 100*10
        #print(y_pred.size())
        loss =(y_pred-y).pow(2).sum()#计算这100个样本的loss
        #print('Epoch{},Loss:{:.4f}'.format(epoch,loss))
                
        grad_y_pred =2*(y_pred-y)
        grad_w2 =torch.mm(h1.t(),grad_y_pred)
                 
        grad_h =grad_y_pred.clone()
        grad_h = torch.mm(grad_h,w2.t())
                
        grad_h.clamp(min=0)
        grad_w1 =torch.mm(x.t(),grad_h)
                
        w1 =w1 -learning_rate*grad_w1
        w2 =w2 -learning_rate*grad_w2
        
        if epoch == epoch_n[i]-1:#如果是最后一次迭代
            final_loss =loss
            print('final_LOSS',final_loss)
            loss_axis.append(final_loss)

#画图部分
from matplotlib import pyplot as plt
import matplotlib
font = {'family' : 'MicroSoft YaHei',
              'weight' : 'bold',
              'size'   : '10'}
matplotlib.rc('font',**font)

plt.xlabel('迭代次数')
plt.ylabel('loss')
plt.title('迭代次数-损失值')

plt.plot(epoch_n,loss_axis)
plt.xticks(epoch_n,rotation=90)
plt.show()
   
#从图上可以看到迭代500次的时候瞬时值减小缓慢



   

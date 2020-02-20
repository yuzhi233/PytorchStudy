# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:30:14 2020

@author: zhoubo
"""

#多层感知机的从0开始实现
import torch
import numpy as np
import sys
import d2lzh_pytorch as d2l
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

#获取和读取数据

#数据集仍然采用FashionMNSIT数据集

#数据集下载
mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, download=True, transform=transforms.ToTensor())

batch_size =256#定义一个批次数据量

#数据集装载 
train_iter =Data.DataLoader(mnist_train,batch_size =batch_size,shuffle =True)
test_iter =Data.DataLoader(mnist_test,batch_size =batch_size,shuffle =True)
    

#定义模型参数
num_inputs,num_hiddens,num_outputs =784,256,10#输入神经元个数784 隐藏层256 输出层个数10

#定义权值矩阵W1,b1,W2,b2（相当于初始化了）

W1 =torch.tensor(np.random.normal(0,0.01,(num_inputs,num_hiddens)),dtype =torch.float)
b1 =torch.zeros(1,num_hiddens,dtype =torch.float)
# print(W1.shape,b1.shape)

W2 =torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_outputs)),dtype =torch.float)
b2 =torch.zeros(1,num_outputs,dtype =torch.float)

#选择需要计算梯度的参数
params =[W1,b1,W2,b2]

for param in params:
    param.requires_grad_(requires_grad=True)#将w1w2b1b2设置成需要计算梯度
  

#定义激活函数  自己实现
    
def relu(X):
    return torch.max(input =X,other =torch.tensor(0.0))
    




def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2




# #定义模型
# def net(X):
#     X=X.view((-1,num_inputs)) #输入的数据以防万一弄成符合列数的
#     H =relu(torch.matmul(X,W1)+b1)#因为X不是2维张量 不能用mm  而要用matmul 
#     return torch.matmul(H, W2) + b2
# 定义损失函数
loss =torch.nn.CrossEntropyLoss()#定义损失函数是交叉熵损失函数
 

# for X,y in train_iter:
#     print('X.shape:\n',X.shape)# batch_size，chanel H W
#     X =X.view(-1,num_inputs)
#     print('after view X.shape:\n',X.shape)
#     H=torch.matmul(X, W1) + b1
#     print('H.shape:\n',H.shape)
#     print('W1.shape:\n',W1.shape)
#     print('W2.shape:\n',W2.shape)
#     O=torch.matmul(H, W2) + b2
   
    
    
#     break   
    


   
  #训练模型  
    
 # 注：由于原书的mxnet中的SoftmaxCrossEntropyLoss在反向传播的时候相对于沿batch维求和了，
# 而PyTorch默认的是求平均，所以用PyTorch计算得到的loss比mxnet小很多（大概是maxnet计算得到的1/batch_size这个量级），
# 所以反向传播得到的梯度也小很多，所以为了得到差不多的学习效果，我们把学习率调得成原书的约batch_size倍，原书的学习率为0.5，
# 这里设置成100.0。(之所以这么大，应该是因为d2lzh_pytorch里面的sgd函数在更新的时候除以了batch_size，
# 其实PyTorch在计算loss的时候已经除过一次了，sgd这里应该不用除了)   
    
    
    
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)   
    
    
    
    
    
    
    
    
    
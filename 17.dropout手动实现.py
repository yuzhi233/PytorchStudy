# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:21:02 2020

@author: zhoubo
"""

#从零实现dropout
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l


def dropout(X,drop_prob):#传入X和dropout概率
    X=X.float()
    assert 0<=drop_prob<=1#先检查输入的drop_prob的概率值是不是0-1之间。不是直接报错
    keep_prob =1 - drop_prob
    
    #如果传入的丢弃概率为1 意思就全丢完
    if keep_prob == 0:
        return torch.zeros_like(X)#返回跟X一样的全是0的tensor矩阵
    mask=(torch.randn(X.shape)<keep_prob).float()#随机生成 一个跟X shape一样的矩阵因为randn是服从标准正态分布的,数值都在0-1之间，然后比较操作得到bool值矩阵，然后让数值小于keep_pro的位置为1,大于的置0 制作一个掩膜 
    return mask*X/keep_prob #张量乘法  对应元素相乘 dropout的规则

X=torch.arange(16).view(2,8)

# #测试dropout函数:
# print(dropout(X,0))#0代表不采用dropout
# print(dropout(X,0.5))#dropout概率设置0.5 一半的概率 会被dropout掉
# print(dropout(X,1))#全dropout


#定义模型参数
# 实验中，依然使用Fashion-MNIST数据集。
# 我们将定义一个包含两个隐藏层的多层感知机，其中两个隐藏层的输出个数都是256

num_inputs,num_outputs,num_hiddens1,num_hiddens2 =784,10,256,256

#手动生成W1,b1，W2,b2，W3,b3矩阵
W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params=[W1,b1,W2,b2,W3,b3]

#定义模型

drop_prob1,drop_porb2= 0.2,0.5#分别设置每个层的丢弃概率

def net(X,is_training=True):#默认是dropout
    X =X.view(-1,num_inputs)#对传入的样本矩阵先view以下确保传入的特征数相匹配
    H1 = (torch.matmul(X,W1)+b1).relu()#这是tensor的方法
    
    if is_training:#只在模型训练的时候使用丢弃法
        H1 =dropout(H1,drop_prob1)#在第一层全连接后添加丢弃层
        
    H2 =(torch.matmul(H1,W2)+b2).relu()
    
    if is_training:
        H2 = dropout(H2, drop_porb2)  # 在第二层全连接后添加丢弃层
        
    return torch.matmul(H2,W3)+b3

        
# 我们在对模型评估的时候不应该进行丢弃!!!
    
#训练和测试模型

num_epochs,lr,batch_size =5,100.0,256  #注意学习率搞成float 因为没有用pytorch的SGD用的是自己写的所以学习率调大点逻辑不一样

#定义损失函数 由于是10分类问题用交叉熵做损失函数
loss = torch.nn.CrossEntropyLoss()

#载入数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
#参数1 :传入的是我们定义的net
#参数2 :传入的训练集数据生成器
#参数3 :传入的测试集数据生成器
#参数4：传入定义的损失函数是哪一种
#参数5 :传入一个batch大小
#参数6 :总共遍历几次全样本
# 参数7 :传入参数 列表形式[w1,b1..]
# 参数8 :学习率






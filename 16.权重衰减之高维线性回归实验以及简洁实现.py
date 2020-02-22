# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:01:08 2020

@author: zhoubo
"""
#高维线性回归实验

# 我们以高维线性回归为例来引入一个过拟合问题，并使用权重衰减来应对过拟合
# 设特征维度为p,使用y =0.05+0.01*(x1+x2+..xn)+e 来生成样本标签噪声项e服从均值为0、标准差为0.01的正态分布

import numpy as np
import torch
import torch.nn as nn
import d2lzh_pytorch as d2l
import sys
sys.path.append("..") 

n_train,n_test = 20,100#训练集样本数20，测试集100  这样训练样本少很容易过拟合
num_inputs =200#输入特征200个

true_w ,true_b = torch.ones(num_inputs,1)*0.01,0.05
e =torch.tensor(np.random.normal(0,0.01,size =(n_test+n_train,1)),dtype=torch.float)

features =torch.randn(n_test+n_train,num_inputs)#生成样本矩阵
print(features.shape)
labels =features.mm(true_w)+true_b#按公式算 这里生成的是采用矩阵乘法 前面几个案例是feature每1列乘以权值
labels +=e#加上噪声 labels制作完成

train_features,test_features = features[0:n_train,:],features[n_train:,:]#划分训练集和测试集

train_labels,test_labels =labels[0:n_train],labels[n_train:]#划分训练集labels和测试集labels



# 开始从0实现权重衰减

# 定义随机初始化模型参数的函数
def init_params():
    w =torch.randn((num_inputs,1),requires_grad=True)#W是num_inputs行 1列
    b =torch.zeros(1,requires_grad=True)
    return [w,b]

# 定义L2范数惩罚项
def l2_penatly(w):
    return(w**2).sum()/2

# 定义模型训练和测试

# 下面定义如何在训练数据集和测试数据集上分别训练和测试模型。
# 与前面几节中不同的是，这里在计算最终的损失函数时添加了L2范数惩罚项。

batch_size,num_epochs,lr = 1,100,0.003
net =d2l.linreg
loss =d2l.squared_loss

dataset =torch.utils.data.TensorDataset(train_features,train_labels)
train_iter =torch.utils.data.DataLoader(dataset,batch_size,shuffle =True)




def fit_and_plot(lambd):
    w,b=init_params()#调用上面定义的初始化参数的函数  其实没啥必要还不如直接写
    train_ls ,test_ls =[],[]#申请俩列表用于存储每一次计算loos 时候算出的loss 后面画epoch-loss图用
    for _ in range(num_epochs):#循环100次  注意这里batch_size =1  
        for X,y in train_iter:
            #添加L2范数
            l =loss(net(X,w,b),y)+lambd*l2_penatly(w)
            l=l.sum()
            
            if w.grad is not None:#若w梯度矩阵不是空的
                #梯度清零
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()#反向传播计算梯度
            d2l.sgd([w,b],lr,batch_size)#进行小批量梯度下降 d2l.sgd里面包含了梯度更新
        #把对训练集进行小批量梯度下降一次算出的loss append到列表
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())#算的是整个训练集的平均loss
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())#算的是测试集的平局误差
            
            #画对数y轴图
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',#第一张图 迭代次数---训练集误差（注意是整个训练集的平均误差）
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])#第二张图 迭代次数---测试集误差
    print('L2 norm of w:', w.norm().item())  #打印w的L2范数  


# 观察过拟合
            
# lambd设置成0  也就是没有正则项           
# fit_and_plot(lambd=0)#观察图 发现 训练集误差很快下降但是测试集误差基本数还是很大 这就说明过拟合


# 使用权重衰减
#labda设置成3  
# fit_and_plot(lambd=3)
# 可以看出，训练误差虽然有所提高，但测试集上的误差有所下降。过拟合现象得到一定程度的缓解。

# net =nn.Linear(num_inputs,1)

#=================================================简洁实现================================================================  

# wd : weight_decay#权重衰减 默认下，PyTorch会对权重和偏差同时衰减。我们可以分别对权重和偏差构造优化器实例，从而只对权重衰减。
def fit_and_plot_pytorch(wd):
    net =nn.Linear(num_inputs,1)
    print(net.parameters())
    nn.init.normal_(net.weight,mean=0,std =1)
    nn.init.normal_(net.bias,mean=0,std=1)
    
    #分别定义优化器
    optimizer_w =torch.optim.SGD(params =[net.weight],lr =lr,weight_decay=wd)#w优化器指定权重衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减
    
    train_ls,test_ls=[],[]
    for _ in range(num_epochs):
        for X,y in train_iter:
            #计算loss
            l =loss(net(X),y).mean()
            
            #梯度清零
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            
            l.backward()
            
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features),train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())                
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())        
    
    
#传入的是wd也就是weight_decay  lambd 
# fit_and_plot_pytorch(0)

fit_and_plot_pytorch(3)


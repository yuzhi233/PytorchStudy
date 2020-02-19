# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:11:21 2020

@author: zhoubo
"""

#%% softmax回归的从零开始实现

import torch
import torchvision
import numpy as np
import sys

import d2lzh_pytorch as d2l


#获取和读取数据
# 使用Fashion-MNIST数据集，并设置批量大小为256。

batch_size =256

#======================================================装载fashion_mnsit数据（先读取再装载，读取的步骤再函数里 ）=========================================================
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

# for X,y in train_iter:
#     print(X)  #!!!!  MNIST每个批次的装载数据都是4维的，维度的构成从前往后分别为batch _size,channel,height和weight
#     print(y)
#     break

#=================================================初始化模型参数================================================================  

num_inputs =784
num_outputs =10

#生成权值矩阵 人工生成
W =torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype =torch.float)
b =torch.zeros(num_outputs,dtype =torch.float)#一定要搞清多分类 和线性回归的 b的形状  b是1行  分类数 列

#W，b需要追踪梯度计算 所以设置成true
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 

#定义softmax运算（层）
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)#每一行的元素sum  dim=1代表 列元素求和 其实相当于行方向求和 keepdim保留原来的维度
    return X_exp / partition  # 这里应用了广播机制

# # 测试一下函数
# x =torch.randn(3,2)
# print(softmax(x))#OK没问题 0<softmax(x)<1



#=======================================================定义模型======================================================================
def net(X):
    return softmax(torch.mm(X.view(-1,num_inputs),W)+b)#通过view将原始数据转成num_inputs列的向量



#=======================================================定义损失函数===========================================================
    
#小例子：如何得到标签的预测概率

# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])#假设每一行是一个样本对3分类 每一类的预测值
# y =torch.LongTensor([0,2])#y表示他们真实的类别 0 代表是第一类  2 代表第三列 也即是这两行的真实y 一个是第一类 一个是第3类
# y_hat.gather(dim = 1,y.view(-1,1))#y.view（-1，1)先将 y变成列向量 然后做聚合操作 dim=1，二维 是按行方向 对应着y向量索引对应的值  得到了得到标签的预测概率，是0.1 0.5


def cross_entropy(y_hat,y):
    return -torch.log(y_hat.gather(1,y.view(-1,1)))#因为数据集的labels    是0-9每个类别只有一个数字代表他的类别 用上面小例子方法找到标签对应我们模型的预测的概率 然后计算交叉熵   并不是独热编码



#===================================================定义net在数据集的准确率==================================================
#计算分类准确率 分类准确率即正确预测数量与总预测数量之比

# def accuracy(y_hat, y):
#     return (y_hat.argmax(dim=1) == y).float().mean().item()



def evaluate_accuracy(data_iter,net):
    acc_sum, n =0.0,0
    for X,y in data_iter:
        acc_sum+=(net(X).argmax(dim =1) == y).float().sum().item()
        n += y.shape[0]#记录总数量 这里一个批次是256个
    return acc_sum/n#返回值是预测在准确率

print(evaluate_accuracy(test_iter,net))#查看以下 测试集没开始训练模型前生成的模型参数 的准确率  基本数约等于1/10 10类别 就瞎猜的概率


#==============================================================训练模型===============================================================

# 我们同样使用小批量随机梯度下降来优化模型的损失函数

num_epochs ,lr =5,0.1#迭代周期4  学习率0.1


#定义训练函数  
def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,
              params =None,lr=None,optimizer=None):
    #遍历所有样本num_epoch次
    for epoch in range(num_epochs):
        
        train_l_sum,train_acc_sum,n = 0.0,0.0,0#初始化 训练集总误差，训练集正确的总个数，训练集的参加计算的总样本数n
        
        for X,y in train_iter:#对训练集样本开始迭代（小批量）256
            y_hat =net(X)#计算y预测
            l =loss(y_hat,y).sum()
            
            #梯度清零
            if optimizer is not None:#如果定义了优化算法  
                optimizer.zero_grad()#优化算法的参数梯度清零
            elif params is not None and params[0].grad is not None:#如果自己事先定义了权值矩阵且grad不为0
                for param in params:#每个参数梯度清零
                    param.grad.data.zero_()
            
            #反向传播
            l.backward()
            
            #参数更新
            if optimizer is None:#如果没有自己定义优化算法
                d2l.sgd(params,lr,batch_size)#更新参数就按小批量梯度下降的方法更新参数
            else:
                optimizer.step()#定义了优化算法就step自动更新
                
            train_l_sum += l.item()#每一批次样本计算完记录下loss 这里的loss是训练集的loss
            
            train_acc_sum +=(y_hat.argmax(dim=1) == y).sum().item()#累加每轮训练正确的个数
            
            n+=y.shape[0]#shape的结果是一个批次中样本的个数也就是batchsize
        
        test_acc =evaluate_accuracy(test_iter,net)#以批次256遍历所有样本后计算测试集准确率
        print('epoch{},loss{:.4f},train acc{:.3f},test acc{:.3f}'.format(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))#train_l_sum/算的一平均个样本的损失
        



#开始训练，得到结果
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)



#=================================================进行预测=====================================================


# 需求：给定一系列图像（第三行图像输出），我们比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）。

X, y = iter(test_iter).next()#拿出test_iter中一个批次的

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:10], titles[0:10])
























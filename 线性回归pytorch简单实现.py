# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:11:29 2020

@author: zhoubo
"""
import torch
import numpy as np
#-------------------------生成数据集-------------------------------------------
num_inputs = 2 #输入特征数2
num_examples = 1000#1000样本
true_w = [2, -3.4]
true_b = 4.2

features =torch.randn(num_examples,num_inputs,dtype = torch.float32)

labels =features[:,0]*true_w[0]+features[:,1]*true_w[1]+true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# print(features[0],labels[0])

#------------------------读取数据--------------------------------------------
#PyTorch提供了data包来读取数据。由于data常用作变量名，我们将导入的data模块用Data代替
import torch.utils.data as Data

#一个批次的数据个数
batch_size =10

#将特征（样本）和标签组合 成数据集
dataset =Data.TensorDataset(features,labels)
print(dataset[0])#可以看出是一个元组的类型

#读取数据  传入数据集 数据个数  打乱
data_iter = Data.DataLoader(dataset,batch_size,shuffle =True)

#查看一下数据集的一个数据
# for X,y in dataset:
#     print(X,y)
#     break

#----------------------------------定义模型------------------------------------
from torch import nn

# class LinearNet(nn.Module):#继承父类
#     def __init__(self, n_feature):
#         super(LinearNet, self).__init__()#调用父类构造
#         self.linear = nn.Linear(n_feature, 1)
#     # forward 定义前向传播
#     def forward(self, x):
#         y = self.linear(x)
#         return y

# net = LinearNet(num_inputs)
# print(net) # 使用print可以打印出网络的结构


# 事实上我们还可以用nn.Sequential来更加方便地搭建网络，Sequential是一个有序的容器，
# 网络层将按照在传入Sequential的顺序依次被添加到计算图中。

# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# # 写法二
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # net.add_module ......

# # 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#           ('linear', nn.Linear(num_inputs, 1))
#           # ......
#         ]))

# print(net)
# print(net[0])

# 可以通过net.parameters()来查看模型所有的可学习参数，此函数将返回一个生成器。
for param in net.parameters():#查看我们构造层后他自己生成的参数 （权值矩阵，偏差什么的）
    print(param)



#----------------------初始化模型参数-----------------------------------------
from torch.nn import init

# 在使用net前，我们需要初始化模型参数，如线性回归模型中的权重和偏差。
# PyTorch在init模块中提供了多种参数初始化方法。这里的init是initializer的缩写形式。

init.normal_(net[0].weight,mean =0 ,std =0.01)#将权值初始化成 均值0 标准差0.01 
init.constant_(net[0].bias,val =0)#将bais初始化成0

#----------------------------定义损失函数-------------------------------------
# 我们现在使用它提供的 均方误差 损失作为模型的损失函数。(pytorch这个均方误差(y_hat-y)**2可没有除以2)
loss =nn.MSELoss()



#------------------------------定义优化算法--------------------------------------
# 我们也无须自己实现小批量随机梯度下降算法。
# torch.optim模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等。
# 下面我们创建一个用于优化net所有参数的优化器实例，并指定学习率为0.03的小批量随机梯度下降（SGD）为优化算法。

import torch.optim as optim

optimizer =optim.SGD(net.parameters(),lr =0.03)
print(optimizer)#可以查看优化算法的一些参数


# 我们还可以为不同子网络设置不同的学习率，这在finetune（微调）时经常用到。例：
# 只是个例子不要去注释：因为我们这个例子没子网络
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)


#如果不想学习率为一个定值
# 主要有两种做法。一种是修改optimizer.param_groups中对应的学习率，
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
    
# 另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。
# 但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。

#----------------------------训练模型-------------------------------------------
    
    
# 我们通过调用optim实例的step函数来迭代模型参数。
# 按照小批量随机梯度下降的定义，我们在step函数中指明批量大小，从而对批量中样本梯度求平均。
    
num_epochs =3#3个循环周期  （把所有样本进行3次小批量梯度下降）

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))#-1自动推断  #计算损失 

        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()   每次循环都要清0不然会累加
        
        l.backward()#反向传播计算梯度
        optimizer.step()#模型参数迭代更新
    print('epoch %d, loss: %f' % (epoch, l.item()))
    
    
dense =net[0]#Dense层就是所谓的全连接神经网络层这里就一层 也是全连接 net是用sequential容器所以有索引
print(true_w, dense.weight)
print(true_b, dense.bias)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


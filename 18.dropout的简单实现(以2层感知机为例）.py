# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:50:24 2020

@author: zhoubo
"""
import torch
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l
#dropout简单实现 
# 在训练模型时，Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素；
# 在测试模型时（即model.eval()后），Dropout层并不发挥作用。


num_inputs,num_outputs,num_hiddens1,num_hiddens2 =784,10,256,256

num_epochs,batch_size =5,256 

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#损失函数  交叉熵 
loss = torch.nn.CrossEntropyLoss()

#定义两个dropout层概率
drop_prob1,drop_prob2 =0.2,0.5

#模型创建 
net =nn.Sequential(d2l.FlattenLayer(),
                    nn.Linear(num_inputs,num_hiddens1),
                    nn.ReLU(),#别忘了ReLU一下
                    nn.Dropout(drop_prob1),#看这里
                    nn.Linear(num_hiddens1,num_hiddens2),
                    nn.ReLU(),
                    nn.Dropout(drop_prob2),
                    nn.Linear(num_hiddens2,num_outputs)
                  )
#参数初始化 注意写法
for param in net.parameters():
    nn.init.normal_(param,mean=0,std=0.01)#注意别初始化std成1了  
    
# 注：由于这里使用的是PyTorch的SGD而不是d2lzh_pytorch里面的sgd，所以就不存在那样学习率看起来很大的问题了。
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)







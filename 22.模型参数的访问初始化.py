# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:37:37 2020

@author: zhoubo
"""

#%% 模型的参数访问

# 我们先定义一个与上一节中相同的含单隐藏层的多层感知机。我们依然使用默认方式初始化它的参数，
# 并做一次前向计算。与之前不同的是，在这里我们从nn中导入了init模块，它包含了多种模型初始化方法。

import torch
from torch import nn
from torch.nn import init#注意写法

net = nn.Sequential(nn.Linear(4, 3),
                    nn.ReLU(), 
                    nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()
# Sequential类与Module类的继承关系。对于Sequential实例中含模型参数的层，
# 我们可以通过Module类的parameters()或者named_parameters方法来访问所有参数（以迭代器的形式返回），
# 后者除了返回参数Tensor外还会返回其名字
#=======================访问模型参数=========================================

#----------------------访问整个net的参数--------------------------
print(type(net.named_parameters()))#查看类型发现是个 generator
for name,param in net.named_parameters():
    print(name,param.size())
    
# for param in net.parameters():
#     print(param)
#--------------------访问net 中具体某层的参数-------------------
    
# 可见返回的名字自动加上了层数的索引作为前缀。 我们再来访问net中单层的参数
# 对于使用Sequential类构造的神经网络， 我们可以通过方括号[]来访问网络的任一层。
# 索引0表示隐藏层为Sequential实例最先添加的层。
for name,param in net[0].named_parameters():
    print(type(param))
    print(name,param.size())
    
# 因为这里是单层的所以没有了层数索引的前缀。
# 另外返回的param的类型为torch.nn.parameter.Parameter，其实这是Tensor的子类，
# 和Tensor不同的是如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里
# 那么它会自动被添加到模型的参数列表里，来看下面这个例子。
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass

n = MyModel()
for name, param in n.named_parameters():
    print(name)
# 上面的代码中weight1在参数列表中但是weight2却没在参数列表中。
# 因为Parameter是Tensor，即Tensor拥有的属性它都有，比如可以根据data来访问参数数值，用grad来访问参数梯度。
    
    
#%%模型参数的初始化
# PyTorch中nn.Module的模块参数都采取了较为合理的初始化策略
# 但我们经常需要使用其他方法来初始化权重。PyTorch的init模块里提供了多种预设的初始化方法。但我们经常需要使用其他方法来初始化权重
# PyTorch的init模块里提供了多种预设的初始化方法。
# 在下面的例子中，我们将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零。 
import torch
from torch import nn as nn
from torch.nn import init


#随便定义一个net
net = nn.Sequential(nn.Linear(4, 3),
                    nn.ReLU(), 
                    nn.Linear(3, 1))  # pytorch已进行默认初始化


print(net)



for name,param in net.named_parameters():
    if 'weight' in name:#如果name中 有'weight'
        init.normal_(param,mean=0,std=0.01)#就将这次循环的param初始化
        print(name)
        print(param.data)
        print('****************')
       
# 下面使用常数来初始化权重参数
for name,param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param,val=0)#权值全部置0
        print(name,param.data)
 
    
    
    
print('==========================自定义初始化方法=====================================')
# 有时候我们需要的初始化方法并没有在init模块中提供。这时，可以实现一个初始化方法，
# 从而能够像使用其他初始化方法那样使用它
        
# 在下面的例子里，我们令权重有一半概率初始化为0，
# 有另一半概率初始化为[−10,−5]和[5,10]两个区间里均匀分布的随机数。  

def init_weight_(tensor):
    with torch.no_grad():#注意这个操作  初始权值肯定不希望记录 我们初始化对权重参数做的操作！
        tensor.uniform_(-10,10)#先把整个param中的tensor按照服从(-10,10)的均匀分布进行生成
        tensor*=(tensor.abs()>=5).float()#然后将(-5,5)的元素 通过bool操作整成false 再float成0 再与原来的tensor广播相乘  得到的就是满足条件的分布，且(-5,5)恰好为0了

for name,param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name,param.data)
        
    
    
# 我们还可以通过改变这些参数的data来改写模型参数值同时不会影响梯度:
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)
        
    
print('==============================共享模型参数==============================')  
# 在有些情况下，我们希望在多个层之间共享模型参数。4.1.3节提到了如何共享模型参数: Module类的forward函数里多次调用同一个层。

# 此外，如果我们传入Sequential的模块是同一个Module实例的话参数也是共享的，下面来看一个例子:
linear = nn.Linear(1, 1, bias=False)#这个例子把bias设置成了false

net = nn.Sequential(linear, linear) 
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)#结果发现就1个参数
    
#在内存中，这两个线性层其实一个对象  验证一下
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))



# 因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的:
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad) # 单次梯度是3，两次所以就是6

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
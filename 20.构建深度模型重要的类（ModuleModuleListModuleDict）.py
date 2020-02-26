# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:08:59 2020

@author: zhoubo

"""

# %%深度学习模型构造

#%% 基于Module类的模型构造方法：它让模型构造更加灵活
import torch
from torch import nn

# Module类是nn模块里提供的一个模型构造类，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型
# 这里定义的MLP类重载了Module类的__init__函数和forward函数。它们分别用于创建模型参数和定义前向计算。前向计算也即正向传播。


class MLP(nn.Module):
    
    def __init__(self,**kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Linear(784,256)#隐藏层
        self.act =nn.ReLU()#激活
        self.output =nn.Linear(256,10)#输出层
    
    # 定义模型的前向传播，即如何根据输入的x算出需要的模型输出
    def forward(self, X):
        a =self.act(self.hidden(X))#先计算从输入层到隐藏层并完成激活操作
        return self.output(a)#把a传进去计算隐藏层到输出层   再返回模型输出
    
 # 以上的MLP类中无须定义反向传播函数。系统将通过自动求梯度而自动生成反向传播所需的backward函数 
    

X = torch.rand(2,784)# 随便生成两个样本
net = MLP()   #实例化一个MLP类对象 名字叫做net
# print(net._modules)
# print(net(X))


# Module类是一个通用的部件。事实上，PyTorch还实现了继承自Module的可以方便构建模型的类
# 如Sequential、ModuleList和ModuleDict等等。

# 手动实现一个与Sequential类具有相同功能的MySequential类 还理解Sequential类的工作机制

class MySequential(nn.Module):
    
    def __init__(self,*args):
        super(MySequential,self).__init__()
        #如果 传入的参数 arg=(a,b,c...) 长度是1 就传入一个参数
        if len(args)==1 and isinstance(args[0],OrderedDict): # 如果传入的是一个OrderedDict
            for key ,module in args[0].items():#items()是python中字典的方法  返回可遍历的(键, 值) 元组数组
                self.add_module(key,module)# add_module方法会将module添加进self._modules(一个OrderedDict)
            
        
        else: #如果传入的是一些Module
            for idx ,module in enumerate(args):# (inx,module)的元组形式 
                self.add_module(str(idx),module)
                
            
    def forward(self,X):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():#按添加模型的顺序 OrderedDict的 值 取出来也就是取出模型
            X = module(X)#每次循环 X就相当于上次的运算结果
        return X
        
# 我们用MySequential类来实现前面描述的MLP类，并使用随机初始化的模型做一次前向计算。
        
net = MySequential(nn.Linear(784,256),
                   nn.ReLU(),
                   nn.Linear(256,10)
                   )

print(net(X))#创建出一个net 给他传参X,因为net是Module类的子类，传入的X会自动调用父类的Module中的__call__方法，call又会调用forward方法


#%% ModuleList 类
import torch
from torch import nn
# ModuleList接收一个子模块的列表作为输入，然后也可以类似List那样进行append和extend操作:
net =nn.ModuleList([nn.Linear(784,256),
                    nn.ReLU()])
net.append(nn.Linear(256,10))# 类似List的append操作
print(net[-1]) 
print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError 说明了ModuleList没有实现forward
# 既然Sequential和ModuleList都可以进行列表化构造网络，那二者区别是什么呢。
# ModuleList仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），而且没有实现forward功能需要自己实现，所以上面执行net(torch.zeros(1, 784))会报NotImplementedError；而Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现

# 总结：ModuleList的出现只是让网络定义前向传播时更加灵活，见下面官网的例子
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)#
        return x


net1=MyModule()
print(net1)




#%% ModuleList不同于一般的Python的list，加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中，下面看一个例子对比一下
import torch
from torch import nn

class Module_ModuleList(nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10)])

class Module_List(nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        self.linears = [nn.Linear(10, 10)]

net1 = Module_ModuleList()
net2 = Module_List()

print("net1:")
for p in net1.parameters():
    print(p.size())

print("net2:")
for p in net2.parameters():
    print(p)
#net2之所以没有打印出参数 正说明了ModuleList([])和python中的 []区别很大 前者会把里面的参数添加到网络

#%% ModuleDict类
# ModuleDict接收一个子模块的字典作为输入, 然后也可以类似字典那样进行添加访问操作:
import torch
from torch import nn  
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError
 # 和ModuleList一样，ModuleDict实例仅仅是存放了一些模块的字典，并没有定义forward函数需要自己定义。同样，ModuleDict也与Python的Dict有所不同，ModuleDict里的所有模块的参数会被自动添加到整个网络中




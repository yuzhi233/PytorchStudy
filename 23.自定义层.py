# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:35:33 2020

@author: zhoubo
"""

# 然PyTorch提供了大量常用的层，但有时候我们依然希望自定义层。
# 如何使用Module来自定义层，从而可以被重复调用。
#%%不含模型参数的自定义层
import torch
from torch import nn


#定义一个不含模型参数的自定义层CenteredLayer
class CenteredLayer(nn.Module):#继承
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

layer=CenteredLayer()#实例化一个CenteredLayer对象

#做前向计算
print(layer( torch.tensor([1,2,3,4,5],dtype =torch.float) ))

# 我们也可以用它来构造更复杂的模型
net = nn.Sequential(nn.Linear(8, 128),
                    CenteredLayer())#<----------- 
y = net(torch.rand(4, 8))



#%%  含模型参数的自定义层
import torch
from torch import nn
# 我们还可以自定义含模型参数的自定义层。其中的模型参数可以通过训练学出

# 在4.2节（模型参数的访问、初始化和共享）中介绍了Parameter类其实是Tensor的子类，如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里。
# 所以在自定义含模型参数的层时，我们应该将参数定义成Parameter，
# 除了像4.2.1节那样直接定义成Parameter类外，还可以使用ParameterList和ParameterDict分别定义参数的列表和字典


#---------------------------------------用ParameteList(参数列表)创建模型参数---------------------------------------------------------------------

class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])#ParameterList接收一个Parameter实例的列表作为输入然后得到一个参数列表，使用的时候可以用索引来访问某个参数，另外也可以使用append和extend在列表后面新增参数。
  
        self.params.append(nn.Parameter(torch.randn(4, 1)))#ParameterList接收一个Parameter实例的列表作为输入然后得到一个参数列表，使用的时候可以用索引来访问某个参数，另外也可以使用append和extend在列表后面新增参数
        # print(self.params[2])#例子：使用的时候可以用索引来访问某个参数
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])#每次循环拿输入X与 我们自定义的模型参数 一层一层的相乘
        return x
    
net =MyListDense()

print(net)

#------------------------------------用ParameterDict(参数字典)创建模型参数---------------------------------------------------------------
# ParameterDict接收一个Parameter实例的字典作为输入然后得到一个参数字典，然后可以按照字典的规则使用了。
# 例如使用update()新增参数，使用keys()返回所有键值，使用items()返回所有键值对等等
class MyDictDense(nn.Module):
    def __init__(self,**kwargs):
        super(MyDictDense,self).__init__()
        self.params = nn.ParameterDict({
            'linear1':nn.Parameter(torch.randn(4,4)),
            'linear2':nn.Parameter(torch.randn(4,3))
            })
        self.params.update({'linear3':nn.Parameter(torch.randn(4,1))})# 例如使用update()新增参数
        
    def forward(self,X,choice ='linear1'):
        return torch.mm(X,self.params[choice])
        
net2 =MyDictDense()
print(net2)

x = torch.ones(1, 4)
print(net2(x, 'linear1'))
print(net2(x, 'linear2'))
print(net2(x, 'linear3'))





# -------------------我们也可以使用自定义层构造模型。它和PyTorch的其他层在使用上很类似。--------------------


net3 =nn.Sequential(MyDictDense(),
                    MyListDense())
print(net3)
print(net3(x))
    
    





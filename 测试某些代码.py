# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:07:13 2020

@author: zhoubo
"""

#%% a.data  a


import torch
from torch.autograd import Variable
w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)#需要求导的话，requires_grad=True属性是必须的。
w2 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)

print(w1.data) # 0.2 版本打印的是 None
print(w1) # 0.2 版本打印的是 



a =torch.tensor([1.,2.,3.],requires_grad= True)
print(a)
print(a.data)
#%%继承复习！！

class A( ):
    def __init__(self):
        print('A')
        self.a =15
    def hello(self):
        print('helloA!')
        
# a =A()
# print(a.a)

class B(A):
    def __init__(self):
        super(B,self).__init__()#调用父类的构造方法/属性       
        print('bbb_init')
        
    def  hello(self):
       pass
        
b =B()

#%% 测试一下tensor的一个方法 transpose
import torch

#对于2维度
a = torch.arange(6).reshape(2,3)
print('a:\n',a)
print('a.transpose()\n',a.transpose(1,0))



#%%next(iterobject,defalt)

b=[1,2,3,4]

c =next(iter(b))

print(c)
print(c)
print(c)

#%% torch.max()

import torch

a =torch.randn(3,4)
print(a)
print(torch.max(a,1))






 
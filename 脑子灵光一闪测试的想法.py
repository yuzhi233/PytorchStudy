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
        
    def  hello(self):
       pass
        
b =B()
b.hello()

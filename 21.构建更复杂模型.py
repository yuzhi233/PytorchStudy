# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:59:23 2020

@author: zhoubo
"""

#%%构建复杂模型
# 虽然上面介绍的这些类可以使模型构造更加简单，且不需要定义forward函数，
# 但直接继承Module类可以极大地拓展模型构造的灵活性。

# 下面我们构造一个稍微复杂点的网络FancyMLP。在这个网络中，
# 我们通过get_constant函数创建训练中不被迭代的参数，即常数参数。
# 在前向计算中，除了使用创建的常数参数外，我们还使用Tensor的函数和Python的控制流，并多次调用相同的层。
import torch
from torch import nn


class FancyMLP(nn.Module):
    def __init__(self):
        super(FancyMLP, self).__init__()

        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


# 在这个FancyMLP模型中，我们使用了常数权重rand_weight（注意它不是可训练模型参数）、做了矩阵乘法操作（torch.mm）并重复使用了相同的Linear层。下面我们来测试该模型的前向计算

# X = torch.rand(2, 20)
# net = FancyMLP()
# print(net)
# net(X)


# 因为FancyMLP和Sequential类都是Module类的子类，所以我们可以嵌套调用它们。
class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU()) 

    def forward(self, x):
        return self.net(x)

net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

X = torch.rand(2, 40)
print(net)
net(X)





# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:30:43 2020

@author: zhoubo
"""

#%% GPU计算

# 到目前为止，我们一直在使用CPU计算。对复杂的神经网络和大规模的数据来说，使用CPU来计算可能不够高效
# 。在本节中，我们将介绍如何使用单块NVIDIA GPU来计算。所以需要确保已经安装好了PyTorch GPU版本。准备工作都完成后
# ，下面就可以通过nvidia-smi命令来查看显卡信息了

# Fri Feb 28 12:32:49 2020       
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 441.22       Driver Version: 441.22       CUDA Version: 10.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  GeForce GTX 1060   WDDM  | 00000000:01:00.0 Off |                  N/A |
# | N/A   50C    P8     6W /  N/A |     92MiB /  6144MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
                                                                               
# +-----------------------------------------------------------------------------+
# | Processes:                                                       GPU Memory |
# |  GPU       PID   Type   Process name                             Usage      |
# |=============================================================================|
# |  No running processes found                                                 |
# +-----------------------------------------------------------------------------+

#计算设备
# PyTorch可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。
# 默认情况下，PyTorch会将数据创建在内存，然后利用CPU来计算。

import torch
from torch import nn

# 用torch.cuda.is_available()查看GPU是否可用:
print(torch.cuda.is_available()) # 输出 True

# 查看GPU数量：
print(torch.cuda.device_count()) # 输出 1

# 查看当前GPU索引号，索引号从0开始
print(torch.cuda.current_device()) # 输出 0

# 根据索引号查看GPU名字:
print(torch.cuda.get_device_name(0)) # 输出 'GeForce GTX 1050'


#%%------------------------------Tensor的GPU计算----------------------------------
import torch
from torch import nn

# 默认情况下，Tensor会被存在内存上。因此，之前我们每次打印Tensor的时候看不到GPU相关标识。
x = torch.tensor([1, 2, 3])
print(x)
# 使用.cuda()可以将CPU上的Tensor转换（复制）到GPU上。
# 如果有多块GPU，我们用.cuda(i)来表示第 i 块GPU及相应的显存（i从0开始）且cuda(0)和cuda()等价。
x = x.cuda(0)
print(x)

# 我们可以通过Tensor的device属性来查看该Tensor所在的设备。
print(x.device)

# 如果对在GPU上的数据进行运算，那么结果还是存放在GPU上。
y = x**2
print(y)

# 需要注意的是，存储在不同位置中的数据是不可以直接进行计算的。
# 即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，
# 位于不同GPU上的数据也是不能直接进行计算的。

# z = y + x.cpu()#报错


#%%--------------------------------模型的GPU计算------------------------------------
# 同Tensor类似，PyTorch模型也可以通过.cuda转换到GPU上。
# 可以通过检查模型的参数的device属性来查看存放模型的设备。
import torch
from torch import nn as nn


net = nn.Linear(3, 1)
print(list(net.parameters())[0].device)
# 可见模型在CPU上，将其转换到GPU上:
net.cuda()
print(list(net.parameters())[0].device)

x=torch.rand(2,3).cuda()
print(net(x))


# PyTorch可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。在默认情况下，PyTorch会将数据创建在内存，然后利用CPU来计算。
# PyTorch要求计算的所有输入数据都在内存或同一块显卡的显存上






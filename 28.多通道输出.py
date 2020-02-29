# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 14:23:44 2020

@author: zhoubo
"""

#%%实现含多个输入通道的互相关运算（卷积运算）
import torch
from torch import nn
import sys
import d2lzh_pytorch as d2l


#这里传入的X是三维的 (chanel,h,w) 计算多通道的卷积运算
def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])#初始化一个res 因为维度比较抽象不好创建
    for i in range(1, X.shape[0]):#注意这里的i是从1开始的 因为上面我们已经计算了 第0维和X的卷积这里从第一维开始卷积再累加
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res


#生成一个2通道的 X 3维度 (2,3,3)  作为输入
X = torch.tensor([ [[0, 1, 2],
                    [3, 4, 5], 
                    [6, 7, 8]],
                      
                      [[1, 2, 3], 
                       [4, 5, 6],
                       [7, 8, 9]] ])


#再创建一个Kernel Kernel也应该是2个通道 是个3维的
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X, K))

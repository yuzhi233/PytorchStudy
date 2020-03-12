# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:17:03 2020

@author: zhoubo
"""

#NIN NET
# LeNet、AlexNet和VGG在设计上的共同之处是：先以由卷积层构成的模块充分抽取空间特征，再以由全连接层构成的模块来输出分类结果。
# NiN它提出了另外一个思路，即串联多个由卷积层和“全连接”层构成的小网络来构建一个深层网络。
# NiN使用1×1卷积层来替代全连接层，从而使空间信息能够自然传递到后面的层中去。图5.7对比了NiN同AlexNet和VGG等网络在结构上的主要区别。

import time
import torch
from torch import nn, optim
import sys
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk

# NiN是在AlexNet问世不久后提出的。它们的卷积层设定有类似之处
# NiN使用卷积窗口形状分别为11×11,5×5和3×3的卷积层，相应的输出通道数也与AlexNet中的一致。
# 每个NiN块后接一个步幅为2、窗口形状为3×3的最大池化层。

# 除使用NiN块以外，NiN还有一个设计与AlexNet显著不同：NiN去掉了AlexNet最后的3个全连接层，
# 取而代之地，NiN使用了输出通道数等于标签类别数的NiN块，然后使用全局平均池化层对每个通道中所有元素求平均并直接用于分类。
# 这里的全局平均池化层即窗口形状等于输入空间维形状的平均池化层。NiN的这个设计的好处是可以显著减小模型参数尺寸，从而缓解过拟合
# 然而，该设计有时会造成获得有效模型的训练时间的增加。

import torch.nn.functional as F

# 局平均池化层:窗口形状等于输入空间维形状的平均池化层
class GlobalAvgPool2d(nn.Module):
    # # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现 等于一张图 全局池化成一个数
    def __init__(self):
        super(GlobalAvgPool2d,self).__init__()

    def forward(self,x):
        return F.avg_pool2d(x,kernel_size =x.size()[2:])#图像是按 样本数 通道数 H W  这里取的是HW

net =nn.Sequential(

                    nin_block(1, 96, kernel_size=11, stride=4, padding=0),#经过一个block后(224-11)/4+1 =54
                    nn.MaxPool2d(kernel_size=3, stride=2),#26
                    nin_block(96, 256, kernel_size=5, stride=1, padding=2),#26
                    nn.MaxPool2d(kernel_size=3, stride=2),#12
                    nin_block(256, 384, kernel_size=3, stride=1, padding=1),#12
                    nn.MaxPool2d(kernel_size=3, stride=2),#5
                    nn.Dropout(0.5),
                    # 标签类别数是10
                    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
                    GlobalAvgPool2d(),
                    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
                    d2l.FlattenLayer()
                    )


print(net)
X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)


batch_size = 128
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)






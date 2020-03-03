# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:30:56 2020

@author: zhoubo
"""

#%% 实现LeNet模型
import time 
import torch

from torch import nn,optim 
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv =nn.Sequential(
            #步长默认为1,padding默认是0 图像是32*328------>卷积后：  
            nn.Conv2d(1,6,5),## in_channels, out_channels, kernel_size   第一个卷积层  输入通道1(后面用Fashion_MNIST只有一通道)，输出通道 第一层设计了6个输出通道，卷积核size5*5  计算后得到卷积后图像为28*28  
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),#池化层 最大池化  kernel size =2*2 池化操作默认 stride=keren_size=2
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)#这步池化后得到的图像尺寸为4*4
        #至此 卷积部分结束
            )
        self.fc =nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
            )
    def forward(self,img):
        feature =self.conv(img)
        output =self.fc(feature.view(img.shape[0],-1))#数据扁平化 一个图像整成一行数据
        return output
        
        
net=LeNet()
print(net)    

#  获取数据和训练模型
# 下面我们来实验LeNet模型。实验中，我们仍然使用Fashion-MNIST作为训练数据集。

batch_size = 256#设置一个batch256个样本

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)#加载数据到迭代器
lr =0.001
num_epochs =5
optimizer =torch.optim.Adam(net.parameters(),lr)
d2l.train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)






    
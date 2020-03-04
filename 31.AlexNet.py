# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:53:50 2020

@author: zhoubo
"""

# AlexNet使用了8层卷积神经网络，并以很大的优势赢得了ImageNet 2012图像识别挑战赛
# AlexNet第一层中的卷积窗口形状是11×11为ImageNet中绝大多数图像的高和宽均比MNIST图像的高和宽大10倍以上，
# ImageNet图像的物体占用更多的像素，所以需要更大的卷积窗口来捕获物体
import torch
from torch import nn
import time
import d2lzh_pytorch as d2l
import torchvision

# 与相对较小的LeNet相比，AlexNet包含8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。
device =torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print('deveice support:\n',device)

#开始搭建AlexNet:(简化版的ALexNet)
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv =nn.Sequential(
            #输入图像是227*227  这跟这节的后面的代码不一样  我是按227输入的 但是这书是按224 224计算比较烦 还是227好
           
            #input_chanel,out_chanel,Kernel_size,stride,padding
            nn.Conv2d(1,96,11,4),#经过这一次卷积后 图像size变成 227-11/4+1 =55   55*55 chanels=96    
            nn.ReLU(),
            
            nn.MaxPool2d(3,2),#经过(#kernel_size=3*3,stride=2)最大池化后 图像size 55-3/2+1 =  27*27 chanel=96
            
            #减小卷积窗口，使用padding=2使得输入和输出的宽高一致# 前两个卷积层后不使用池化层来减小输入的高和宽
            
            nn.Conv2d(96,256,5,1,2),#经过卷积后 图像size变成 27+2*2-5/1+1 =27(不变)  27*27 chanels =256
            nn.ReLU(),
            nn.MaxPool2d(3,2),#经过(kernel_size =3,stride=2)最大池化 图像size 27-3/2 =13
            
            #在进行3次卷积 ，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256,384,3,1,1),#卷积后  图像size变成 13+2*1-3/1+1 =13(不变)  13*13 chanels =384
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),#卷积后 图像 size变成 13+2*1-3/1+1 =13(还是不变)  13*13 chanels =384
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),#卷积后 size 13+2*1-3/1+1 =13(还是不变)  13*13 chanels=256
            nn.ReLU(),
            nn.MaxPool2d(3,2) #经过(kernel_size =3,stride=2)的最大池化后  图像size = 13-3/2+1 =6  chanel=256
            
            )
        # 因为是简化版的AlexNet chanel和图像的size还是太大 所以这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合（正好复习一波）
        self.fc =nn.Sequential(
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(0.5),#经过dropout层会将上一层的神经元一半置为0
            # nn.Linear(4096,4096)#第二个全连接层 我这里想改一下 感觉 4096-4096 不太好
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(2048,10)
            )
        
        
        #定义前向传播
    def forward(self,img):
        features =self.conv(img)
        output =self.fc(features.view(img.shape[0],-1))
        return output
        
        
# 打印网络 看有错的不
net =AlexNet()
print(net)
                     
# 读取数据
# 虽然论文中AlexNet使用ImageNet数据集，但因为ImageNet数据集训练时间较长，我们仍用前面的Fashion-MNIST数据集来演示AlexNet。
        
# 读取数据的时候我们额外做了一步将图像高和宽扩大到AlexNet使用的图像高和宽227 这里不一样！
# 这个可以通过torchvision.transforms.Resize实例来实现。
# 也就是说，我们在ToTensor实例前使用Resize实例，然后使用Compose实例来将这两个变换串联以方便调用。

batch_size =128  
     
train_iter,test_iter =d2l.load_data_fashion_mnist(batch_size,resize =227)     

#训练
# 这时候我们可以开始训练AlexNet了。相对于LeNet，由于图片尺寸变大了而且模型变大了，
# 所以需要更大的显存，也需要更长的训练时间了


lr =0.001
num_epochs =5
optimizer =torch.optim.Adam(net.parameters(),lr =lr)
d2l.train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)
             
            
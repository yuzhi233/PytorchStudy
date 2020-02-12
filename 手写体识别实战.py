# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:44:14 2020

@author: zhoubo
"""

#手写体识别实战！

#torch vis i on 包的主要功能是实现数据的处理、导入和预览等，所以如果需要对计算机视
#觉的相关问题进行处理，就可以借用在torchvision包中提供的大量的类来完成相应的工作


#导入必须的包
import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable

from matplotlib import pyplot as plt
import numpy as np


device = torch.device('cuda:0'if torch.cuda.is_available() else "cpu")
print(device)

#--------------------------------------------打印64张（一个批次的训练集图片）并查看标签-------------------------------------------------------

# 在计算机视觉中处理的数据集有很大－部分是图片类型的，
#而在PyTorch 中实际进行计算的是Tensor 数据类型的变量，所以我们首先需要解决的是数据类型转换的问题
# 如果获取的数据是格式或者大小不一的图片，则还需要进行归一化和大小缩放等操作
# 这些方法在torch. transforms 中都能找到。

#我们可以将以上代码中的trochvision.transforms.Compose类看作一种容器它能够同
# 时对多种数据变换进行组合。传入的参数是一个列表，列表中的元素就是对载入的数据进
# 行的各种变换操作



#定义需要变换的操作 进行数据的转换操作 先转成方便计算的张量tensor，再进行归一化 （自定义了均值和标准差0.5,每个通道的通道值均值标准差)
transform = transforms.Compose([transforms.ToTensor(),                     
                                transforms.Normalize(mean=[0.5],std=[0.5])])



#训练集数据（数据集下载）
#读取数据 and 处理数据
data_train = datasets.MNIST(root ='./data/',transform =transform ,train=True,download=True)
data_test = datasets.MNIST(root= './data/',transform =transform,train =False)
# print(data_train)
# print(data_test)



#载入数据  载入的同时打乱图片
data_loader_train = torch.utils.data.DataLoader(dataset =data_train,
                                                batch_size =64,
                                                shuffle =True)
data_loader_test = torch.utils.data.DataLoader(dataset =data_train,
                                               batch_size =64,
                                                shuffle =True)

#浏览数据
images,labels =next(iter(data_loader_train))#从可迭代对象中依次取出一个批次的数据（64个）images是4维度的(batch_size,chanel,height,weight))
# print('images的shape是:\n',images.shape)
img = torchvision.utils.make_grid(images)#将取出一个批次的图像，按网格划分，此时 img的维度是3维的，shape为(chanel,height,weight)
# print('img没换轴之前的shape是:\n',img.shape)
img = img.numpy().transpose(1,2,0)#matplotlib中图像是按（height,weight,chanel)才能正确显示的,所以这一步先给他转成numpy再用numpy的transpose将2轴元素互换以便能正确显示
# print('img换轴之后的shape是:\n',img.shape)


#定义均值方差
std =[0.5,0.5,0.5]
mean =[0.5,0.5,0.5]

img = img*std+mean
print([labels[i] for i in range(64)])#打印图片标签

# plt.imshow(img)

#------------------------------------模型搭建和参数优化-------------------------

class Model(torch.nn.Module):#定义自己的model 继承父类torch.nn.module
    def __init__(self):
        super(Model,self).__init__()#继承父类的构造 
        #卷积层
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size =3,stride=1,padding =1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64,128,kernel_size =3,stride =1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride =2,kernel_size =2) 
                                        )
        #全连接层
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128,1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024,10)
                                         )
    def forward(self,x):
            x = self.conv1(x)
            x = x.view(-1,14*14*128)
            x = self.dense(x)
            return x

    
model =Model().to(device)#实例化出我们的一个模型
cost_function = torch.nn.CrossEntropyLoss()#定义损失函数为交叉熵损失函数
optimizer =torch.optim.Adam(model.parameters())#定义优化算法 ，将model中的参数传进去

print(model)

#模型训练和优化
epoch_n =5
 
for epoch in range(epoch_n):
    running_loss =0.0
    running_correct =0
    print('Epoch{}/{}'.format(epoch,epoch_n))
    print('-'*10)
    #对每一个批次（64张）的图片
    for data in data_loader_train:
        X_train,y_train =data
        X_train,y_train =Variable(X_train),Variable(y_train)
        
        X_train =X_train.to(device)
        y_train =y_train.to(device)
        
        outputs = model(X_train)
        _,pred =torch.max(outputs.data,1)#返回的pred每一行是最大值元素的索引 结果是64行10列
        
        loss =cost_function(outputs,y_train)
        optimizer.zero_grad()
      
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data
        running_correct += torch.sum(pred ==y_train.data)
        
    testing_correct =0
    for data in data_loader_test:
        X_test,y_test =data
        X_test,y_test =Variable(X_test),Variable(y_test)
            
        X_test =X_test.to(device)
        y_test =y_test.to(device)
            
        outputs =model(X_test)
        _,pred = torch.max(outputs,1)#返回的pred每一行是最大值元素的索引
        
        testing_correct += torch.sum(pred == y_test.data)
        
    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(running_loss/len(data_train),100*running_correct/len(data_train),100*testing_correct/len(data_test)))





















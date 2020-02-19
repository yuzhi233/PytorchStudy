# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:41:12 2020

@author: zhoubo
"""
#%%FashionMNIST数据集下载 和 查看
#获取数据集
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import time 
import d2lzh_pytorch as d2l



#数据集简单介绍：
# 训练集中和测试集中的每个类别的图像数分别为6,000和1,000。
# 因为有10个类别，所以训练集和测试集的样本数分别为60,000和10,000。
# transforms.ToTensor()将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片或者数据类型
# 为np.uint8的NumPy数组转换数据类型为torch.float32为尺寸为(C x H x W)

#下载数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, download=True, transform=transforms.ToTensor())

print(mnist_train,mnist_test)
print(type(mnist_train))#class 'torchvision.datasets.mnist.FashionMNIST
print(len(mnist_train),len(mnist_test))
# print('查看第一个图像的数据',mnist_train[0][0])

#通过下标访问任意一个样本
features,label =mnist_train[0]
print(features.shape,label) #chanel height weight

#t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）
# 、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴） 10个类别

#以下函数可以将数值标签转成相应的文本标签。
# def get_fashion_mnist_labels(labels):
#     text_labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]
print(d2l.get_fashion_mnist_labels([9]))#返回的是个列表

#查看训练数据中前10个样本的图像内容和文本标签  方便看 下面是直接调用函数了
# 以下函数可以在一行里画出多张图像和对应标签的函数    
# def show_fashion_mnist(images, labels):
#     use_svg_display()
#     # 这里的_表示我们忽略（不使用）的变量
#     _, figs = plt.subplots(1, len(images), figsize=(12, 12))#1行 len（images）列
#     for f, img, lbl in zip(figs, images, labels):
#         f.imshow(img.view((28, 28)).numpy().transpose(1,2,0))#先将img转成numpy()数组，再将通道轴和C轴和W轴交换 matplotlib才能正确显示
        
#         f.set_title(lbl)
#         f.axes.get_xaxis().set_visible(False)
#         f.axes.get_yaxis().set_visible(False)
    # plt.show()

#展示10张图片 和它对应的标签
# X,y=[],[] #将X,y分别用列表装
# for i in range(10):#看10个数据
#     X.append(mnist_train[i][0])#每一行数据的第一列   对应的是feature
#     y.append(mnist_train[i][1])#每一行数据的第二列   对应的是label
# d2l.show_fashion_mnist(X,d2l.get_fashion_mnist_labels(y))



#读取小批量

#mnist_train是torch.utils.data.Dataset的子类，
# 所以我们可以将其传入torch.utils.data.DataLoader来创建一个读取小批量数据样本的DataLoader实例

batch_size = 256 #一次读取256张 28*28的图
#设置多进程读取（4个进程）
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看读取一遍训练数据需要的时间。
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))












import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
import sys
from torch import nn
#这个包是敲代码过程中收集的    也是书上写的



def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data



def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 本函数已保存在d2lzh包中方便以后使用
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)
 
       
# Fashion-MNIST中一共包括了10个类别，分别为t-shirt（T恤）、trouser（裤子）、
# pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
# 以下函数可以将数值标签转成相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
   
    


        
# 以下函数可以在一行里画出多张图像和对应标签的函数    
def show_fashion_mnist(images, labels):#传入的第一个参数是 数据集里的图像(列表的形式)  第二个参数是标签（列表）
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _,figs  = plt.subplots(1, len(images), figsize=(12, 12))#1行 len（images）列
    for f, img, lbl in zip(figs, images, labels):#打包成一个可迭代序列（fig,image,label)
        img = img.view((28, 28)). numpy()#img是从数据集（数据集是3维度的）拿出来被转成了tensor 是按(C,H,W)储存的是3维的 图像是2维度的，要先转换成2维度才能显示
        f.imshow(img)#显示图像  至于为啥是这个色 我还在研究 感觉是通道不对，但是灰度图不是就1通道吗 头大
        f.set_title(lbl)#将每次读取到的标签设置成标题
        f.axes.get_xaxis().set_visible(False)#不显示x轴
        f.axes.get_yaxis().set_visible(False)#不显示y轴
    plt.show()
    
    
    
    
# 获取并读取Fashion-MNIST数据集
def load_data_fashion_mnist(batch_size):
    
    mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets', train=False, download=True, transform=transforms.ToTensor())
    
    
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter,test_iter




#定义net在数据集的准确率   主要是传入测试集
def evaluate_accuracy(data_iter,net):
    acc_sum, n =0.0,0
    for X,y in data_iter:
        acc_sum+=(net(X).argmax(dim =1) == y).float().sum().item()
        n += y.shape[0]#记录总数量 这里一个批次是256个
    return acc_sum/n
    
    
    
#模型训练并计算准确率   
def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params =None,lr=None,optimizer=None):
    
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n = 0.0,0.0,0
        for X,y in train_iter:
            y_hat =net(X)
            l =loss(y_hat,y).sum()
            
            #梯度清零
            if optimizer is not None:#如果定义了优化算法  
                optimizer.zero_grad()#优化算法的参数梯度清零
            elif params is not None and params[0].grad is not None:#如果定义了参数
                for param in params:#每个参数梯度清零
                    param.grad.data.zero_()
            
            #反向传播
            l.backward()
            
            #参数更新
            if optimizer is None:#如果没有自己定义优化算法
                sgd(params,lr,batch_size)#更新参数就按小批量梯度下降的方法更新参数
            else:
                optimizer.step()#定义了优化算法就step自动更新
                
                
            train_l_sum += l.item()#每一批次样本计算完记录下loss 这里的loss是训练集的loss  l是tensor类型 ,item()是转成python number
            
            train_acc_sum +=(y_hat.argmax(dim=1) == y).sum().item()# 计算每个批次y_hat中预测正确的个数并累加
            
            n+=y.shape[0]#shape的结果是一个批次中样本的个数也就是batchsize #shape是torch.size[256]，加上索引才能把值拿出来
        
        test_acc =evaluate_accuracy(test_iter,net)
        print('epoch{},loss{:.4f},train acc{:.3f},test acc{:.3f}'.format(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))    
    
    
    
 # 对x的形状转换的这个功能自定义一个FlattenLayer并记录在d2lzh_pytorch中方便后面使用。
# 本函数已保存在d2lzh_pytorch包中方便以后使用
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
    
    
    
    
    
    
    
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


# 线性回归的矢量计算表达式
def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2





def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data 因为你的更新参数的操作不能保存在计算图中



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




#定义net在数据集的准确率   主要是传入测试集  已经被下面升级版的测试集评估函数取代👇
# def evaluate_accuracy(data_iter,net):
#     acc_sum, n =0.0,0
#     for X,y in data_iter:
#         acc_sum+=(net(X).argmax(dim =1) == y).float().sum().item()
#         n += y.shape[0]#记录总数量 这里一个批次是256个
#     return acc_sum/n
    

#我们在对（测试集）模型评估的时候不应该进行dropout，所以我们修改一下d2lzh_pytorch中的evaluate_accuracy函数:
# 本函数已保存在d2lzh_pytorch  注意：如果net是通过sequetical创建 下面方法会报错 用上面的👆
def evaluate_accuracy(data_iter, net):#用于评估测试集准确率 目的是要实现评估的时候要自动关闭dropout net是序列容器生成的会报错用上面那个
    acc_sum, n = 0.0, 0
    for X, y in data_iter:#从data_iter取出一个batch的X，y
         #先判断你这个net是怎么产生的是你自己手写的还是利用pytorch快速生成的
        if isinstance(net, torch.nn.Module):#判断net是不是用torch.nn.Module创建的实例
            net.eval() # #如果是上面方法创建的 那么开启评估模式 dropout层全部关闭
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()#判断正确的个数
            net.train() # 改回训练模式
        else: # 如果是我们自定义的模型
            if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() #先将is_training设置成 False 关闭dropout
            else:#(形参)没有is_training这个参数
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
        n += y.shape[0]#其实就是算了以下一个批次有多少样本 每次循环累加一下参加计算的样本数
    return acc_sum / n#在所有批次循环后  计算准确率 拿 准确的个数/总个数

            
            

    
    
#模型训练并计算准确率
#参数1 :传入的是我们定义的net
#参数2 :传入的训练集数据生成器
#参数3 :传入的测试集数据生成器
#参数4：传入定义的损失函数是哪一种
#参数5 :传入一个batch大小
#参数6 :总共遍历几次全样本
#参数7 :传入参数 列表形式[w1,b1..]
#参数8 :学习率
#参数9 :优化器
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
    
    
    
    
# 本函数已保存在d2lzh_pytorch包中方便以后使用
        #画图 y轴是对数尺度的
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals :
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

    
    
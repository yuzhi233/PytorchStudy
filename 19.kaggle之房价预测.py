# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:53:44 2020

@author: zhoubo
"""

#获取和读取数据集
# 比赛数据分为训练数据集和测试数据集。两个数据集都包括每栋房子的特征，如街道类型、建造年份、房顶类型、地下室状况等特征值。
# 这些特征值有连续的数字、离散的标签甚至是缺失值“na”。只有训练数据集包括了每栋房子的价格，也就是标签

import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import d2lzh_pytorch as d2l

#读取数据
train_data =pd.read_csv('./Datasets/house-prices-advanced-regression-techniques/train.csv')#1460个样本 80个特征
test_data =pd.read_csv('./Datasets/house-prices-advanced-regression-techniques/test.csv')#1459

#查看前4个样本 的 前4个特征和后两个特征  和标签
print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])
#可以看到第一列 是ID 它能帮助模型记住每个训练样本，但难以推广到测试样本，所以我们不使用它来训练。我们将所有的训练数据和测试数据的79个特征按样本连结。

#将训练集和测试集按行拼接到一起 同时舍弃ID这一列 也就是从索引1开始
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))#2919*79  2919=1459+1460


#数据预处理

# 我们对连续数值的特征做标准化（standardization）:
# 设该特征在整个数据集上的均值为μ，标准差为σ。我们可以将该特征的每个值先减去μ再除以σ得到标准化后的每个特征值。
# 对于缺失的特征值，我们将其替换成该特征的均值。对于字符串我们直接不取这些列

#先取出连续特征
# all_features.dtypes返回一个每一列的数据类型的Series. all_features.dtypes != 'object'判断特是不是字符串返回bool Series，然后Series索引这个bool取出了所有不是字符串的特征的Series然后用index方法取出这些特征的索引
numeric_features =all_features.dtypes[all_features.dtypes != 'object'].index 

all_features[numeric_features]=all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))#将lambda函数运用到每一列
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] =all_features[numeric_features].fillna(0)


#再处理离散特征
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape # (2919, 331)

# 最后，通过values属性得到NumPy格式的数据，并转成Tensor方便后面的训练。
n_train =train_data.shape[0]
train_features =torch.tensor(all_features[:n_train].values,dtype = torch.float)
test_features =torch.tensor(all_features[n_train:].values,dtype =torch.float)

train_labels =torch.tensor(train_data.iloc[:n_train,-1],dtype =torch.float).view(-1,1)#转成一列

#我们使用一个基本的线性回归模型和平方损失函数来训练模型。

loss =torch.nn.MSELoss()

def get_net(feature_num):#参数为  输入的特征数
    net = nn.Linear(feature_num,1)#创建一个 features_num行1列的线性层
    #初始参数为均值0 方差0.01
    for param in net.parameters():
        nn.init.normal_(param,mean=0,std= 0.01)
    return net

def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()


#定义模型训练函数
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, 
          learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []#老规矩存储训练集 和 测试集的loss
    
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)#制作datase
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)#装载数据
    
    # 定义优化算法  这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay) #权值衰减开启
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:           
            #计算loss
            l = loss(net(X.float()), y.float())
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            l.backward()
            #参数更新
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))#一个epoch所有样本计算完loss存入列表 因为房价预测比赛评价模型的使用的是 对数均方根误差
        if test_labels is not None:#如果参数中传了 testlabels 就将testlabels的误差也计算存入列表
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证 将被用来选择模型设计并调节超参数
# 下面实现了一个函数，它返回第i折交叉验证时所需要的训练和验证数据。

def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1#首先判断你要进行几折交叉验证 这个数字必须得是大于1的 才有意义 小于1 报错
    fold_size = X.shape[0] // k #如果进行K折交叉验证那么需要划分K个子集  这里是计算每个子集的大小 地板除
    X_train, y_train = None, None
    
    for j in range(k):#循环K次 j=0，1，2，3，4...K
        
        idx = slice(j * fold_size, (j + 1) * fold_size)# slice() 函数实现切片对象 inx =slice(1,4) a=[0,1,2,3,4,5,6,7,8,9,10] a[inx]：[1, 2, 3] 
        X_part, y_part = X[idx, :], y[idx]#截取一个fold_size的X数据和y数据
        
        if j == i:#判断当前循环次数是否是 我们要取出当测试集的那一次（第i次）如果是：
            X_valid, y_valid = X_part, y_part#那么把这次循环截取到的子集 作为验证集
        elif X_train is None:#如果X_train为空 （一般来说第一次循环的时候是None）就把截取到的数据 复制给X_train y_train
            X_train, y_train = X_part, y_part
        
        else:#如果X_train y_train 有值了或者 当前循环取出的数据不是我们要用作测试集的 
            X_train = torch.cat((X_train, X_part), dim=0)#竖着拼接拼成一列
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid



# 在K折交叉验证中我们训练K次并返回训练和验证的平均误差。
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    
    train_l_sum, valid_l_sum = 0, 0#定义训练集loss和验证集 loss置为0
    
    for i in range(k):#相当于K个测试集-验证集 循环K次
        data = get_k_fold_data(k, i, X_train, y_train)# 获取K折交叉验证所用的测试集验证集数据     data此时应该是一个元组 的形式
        net = get_net(X_train.shape[1])#创建模型 简单的2层 输出层一个单元
        
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)#K折 i折验证的 训练集误差 和验证集误差
        
        train_l_sum += train_ls[-1]#对 把 计算第K折训练集上的误差  累加给 训练总误差
        valid_l_sum += valid_ls[-1]#第K折验证集上的 误差同理
        if i == 0:#画出第i==0次 训练集-rmse  验证集-rmse 对数图
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
            
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))#
        
    return train_l_sum / k, valid_l_sum / k #返回的是在K折交叉验证的平均误差


#=================================进行K(K=5)折交叉验证=================================
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 15, 0, 64
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
# print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

#一定要弄清K折交叉验证作用是什么  :简单来说 就是调参要在这里进行 这里调的差不多再扔真正的测试集！


#============================在真正的测试集上进行预测====================================

# 在预测之前，我们会使用完整的训练数据集来重新训练模型，并将预测结果存成提交所需要的格式。



def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    
    #用我上面定义的函数 创建简单模型
    net = get_net(train_features.shape[1])#train_features是训练集 shape[1]取出的列数 就是输入的特征数
    
    #用上面的训练函数  训练完整的训练集 得到完整训练集上的loss （按kaggle评估标准的loss）
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    #画图y轴为对数轴  epochs-rmse 图
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    
    print('train rmse %f' % train_ls[-1])#打印出  完整训练集 误差 
    
    preds = net(test_features).detach().numpy()#预测  将真正测试集传入 模型 ---得到预测值 
    
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])#series因为是传入的是numpy的pred得给他整成一行 才能生成Series
    
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)#按行方向拼接横着拼接
    
    submission.to_csv('./submission.csv', index=False)
    
    
    # 设计好模型并调好超参数之后，下一步就是对测试数据集上的房屋样本做价格预测。
    # 如果我们得到与交叉验证时差不多的训练误差，那么这个结果很可能是理想的，可以在Kaggle上提交结果。
    
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)




































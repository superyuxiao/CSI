#!E:\Python\Python368-64\python.exe
# -*- encoding: utf-8 -*-
'''
@File    :   gesture_recognition_3.2.py
@Time    :   2021/07/05 09:45:10
@Author  :   Yu Xiao 于潇 
@Version :   1.0
@Contact :   superyuxiao@icloud.com
@License :   (C)Copyright 2020-2021, Key Laboratory of University Wireless Communication
                Beijing University of Posts and Telecommunications
@Desc    :   None
'''

# ------------------------------ file details ------------------------------ #
# 四个人，一个位置，巴特沃斯低通，PCA，九个天线对，81*9输入CNN
# 使用pytorch重构
# 创建自己的数据集，但是速度特别特别特别慢
# ------------------------------ file details ------------------------------ #

import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import random
from numpy.lib.scimath import _fix_real_abs_gt_1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import gzip
import json
from matplotlib.pyplot import subplot
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import datetime
import math

#! 
def get_scale_csi(csi_st):
    #Pull out csi
    csi = csi_st['csi']
    # print(csi.shape)
    # print(csi)
    #Calculate the scale factor between normalized CSI and RSSI (mW)
    csi_sq = np.multiply(csi, np.conj(csi)).real
    csi_pwr = np.sum(csi_sq, axis=0)
    # csi_pwr = csi_pwr.reshape(1, csi_pwr.shape[0], -1)
    csi_pwr = np.reshape(csi_pwr,(csi_pwr.shape[0],-1))
    rssi_pwr = dbinv(get_total_rss(csi_st))

    scale = rssi_pwr / (csi_pwr / 30)

    if csi_st['noise'] == -127:
        noise_db = -92
    else:
        noise_db = csi_st['noise']
    thermal_noise_pwr = dbinv(noise_db)

    quant_error_pwr = scale * (csi_st['Nrx'] * csi_st['Ntx'])

    total_noise_pwr = thermal_noise_pwr + quant_error_pwr
    ret = csi * np.sqrt(scale / total_noise_pwr)
    if csi_st['Ntx'] == 2:
        ret = ret * math.sqrt(2)
    elif csi_st['Ntx'] == 3:
        ret = ret * math.sqrt(dbinv(4.5))
    return ret

def get_total_rss(csi_st):
    # Careful here: rssis could be zero
    rssi_mag = 0
    if csi_st['rssi_a'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_a'])
    if csi_st['rssi_b'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_b'])
    if csi_st['rssi_c'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_c'])
    return db(rssi_mag, 'power') - 44 - csi_st['agc']

def dbinv(x):
    return math.pow(10, x / 10)

def db(X, U):
    R = 1
    if 'power'.startswith(U):
        assert X >= 0
    else:
        X = math.pow(abs(X), 2) / R

    return (10 * math.log10(X) + 300) - 300
#!

def read_sample(filepath):
    """
    @description  : 读取csi样本，并归一化csi
    ---------
    @param  : filepath：样本路径
    -------
    @Returns  : scale_csi：归一化csi
    -------
    """

    # 读取样本
    sample = np.load(filepath, allow_pickle=True)
    # 设置csi容器，格式为样本长度（帧数）*子载波数30*发送天线3*接收天线3，复数
    scale_csi = np.empty((len(sample),30,3,3), dtype = complex)
    # 逐帧将csi归一化
    for i in range(len(sample)):
        scale_csi[i] = get_scale_csi(sample[i])

    return scale_csi


def butterworth_lowpass(scale_csi, order, wn):
    """
    @description  : 巴特沃斯低通滤波器
    ---------
    @param  : scale_csi：归一化后的csi，order：滤波器阶数，wn：归一化截至角频率 
    -------
    @Returns  : 低通滤波后的csi幅度
    -------
    """
    # 设置csi容器，格式为样本长度（帧数）*子载波数30*发送天线3*接收天线3
    csi = np.empty((len(scale_csi),30,3,3))
    wn = 0.05
    order = 4
    # 引入butter函数
    b, a = signal.butter(order, wn, 'lowpass', analog = False)
    # i发射天线，j接收天线，k子载波序号
    for i in range(3):
        for j in range(3):
            for k in range(30):
                data = abs(scale_csi[:,k,i,j])
                csi[:,k,i,j] = signal.filtfilt(b, a, data, axis=0)

    return csi

def PCA_9(csi_abs, n_components, whiten):
    """
    @description  : PCA，根据天线对分成9组，每组得到一组主成分
    ---------
    @param  : csi_abs：csi的幅度矩阵，n_components：主成分数，whiten：是否白化True/False
    -------
    @Returns  : 返回csi_pca，主成分矩阵
    -------
    """
    
    pca = PCA(n_components=n_components, whiten=whiten)
    # 设置csi容器，格式为样本长度（帧数）*主成分数n_components*发送天线3*接收天线3
    csi_pca  = np.empty((len(csi_abs),n_components,3,3))
    for i in range(3):
        for j in range(3):
            data = csi_abs[:,:,i,j]
            data = np.reshape(data, (data.shape[0],-1)) #转换成二维矩阵
            pca.fit(data)
            data_pca = pca.transform(data)
            csi_pca[:,:,i,j] = data_pca[:,:]

    return csi_pca



def data_processing(path, feature_number):
    for i in range(50):
        # 样本路径
        filepath = path 
        # 读取样本
        scale_csi = read_sample(filepath)
        # 低通滤波
        csi_lowpass = butterworth_lowpass(scale_csi, 7, 0.01)
        # PCA
        csi_pca_9 = PCA_9(csi_abs=csi_lowpass, n_components=1, whiten=False)
        # 画幅度图
        #plt_9_amplitude(csi_pca_9,range(1))
        csi_pca = csi_pca_9[:,0,:,:]
        # 截取长度800，步进10采样
        csi_vector = np.zeros((81,3,3))
        if np.shape(csi_pca)[0] < 810:
            csi_empty = np.zeros((810,3,3))
            csi_empty[:np.shape(csi_pca)[0]] = csi_pca[:,:,:]
            csi_vector[:] = csi_empty[::10,:,:]
        else:
            csi_pca = csi_pca[:809,:,:]
            csi_vector[:] = csi_pca[::10,:,:]
        # 添加标签
        csi_vector = np.reshape(csi_vector, (81,9))
        csi_vector.dtype = 'float64'
        # 返回数据
        data = csi_vector
    return data
 
class MyDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,root, datatxt, transform=None, target_transform=None): #初始化一些需要传入的参数
        
        fh = open(root + datatxt, 'r')                  #按照传入的路径和txt文本参数，打开这个文本，并读取内容
        sample = []                                     #创建一个名为img的空列表，一会儿用来装东西
        for line in fh:                                 #按行循环txt文本中的内容
            line = line.rstrip()                        # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()                        #通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            sample.append((words[0],int(words[1])))     #把txt里的内容读入sample列表保存，具体是words几要看txt内容而定
        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.sample = sample
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):#这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.sample[index] #fn是图片path #fn和label分别获得sample[index]也即是刚才每行中word[0]和word[1]的信息
        csi_vector = data_processing(fn, 81*3*3)
        # print(np.shape(scale_csi))
        # segment = np.expand_dims(segment,axis=0)
        # segment = segment.astype(float)
        if self.transform is not None:
            csi_vector = self.transform(csi_vector) #是否进行transform
        csi_vector = csi_vector.to(torch.float32)
        return csi_vector, label
        #return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
 
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.sample)

# 定义模型结构
class CNN(nn.Module):
     def __init__(self):
         super(CNN, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=5)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=1, padding=5)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是10
         self.fc = nn.Linear(in_features=1440, out_features=3)
         
   # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
   # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = x.view([x.shape[0], 1440])
         x = self.fc(x)
         x = F.softmax(x, dim=1)
         
         return x

if __name__ == '__main__':

    print(os.path.abspath('..'))
    # # 设置下标 划分训练集和测试集
    # index = [i for i in range(51)]
    # random.shuffle(index)
    # split_ratio = 0.8       # 训练集比例
    # train_index = index[:int(len(index)*split_ratio)]
    # test_index = index[int(len(index)*split_ratio):]
    # #! train dataset
    # #with open('E:\CSI\CSI\classroom_data_unit\dataset.txt', 'w') as f:
    # with open('E:/project/CSI/classroom_data_unit/train_data.txt', 'w') as f:
    #     for i  in range(1,2):
    #         for j in train_index:
    #             f.write('./CSI/classroom_data_unit/DX/O/gresture_O_location_' + str(i) + '_' + str(j) + '.npy 0\n')
    #             f.write('./CSI/classroom_data_unit/DX/PO/gresture_PO_location_' + str(i) + '_' + str(j) + '.npy 1\n')
    #             f.write('./CSI/classroom_data_unit/DX/X/gresture_X_location_' + str(i) + '_' + str(j) + '.npy 2\n')
    
    # #! test dataset
    # #with open('E:\CSI\CSI\classroom_data_unit\dataset.txt', 'w') as f:
    # with open('E:/project/CSI/classroom_data_unit/test_data.txt', 'w') as f:
    #     for i  in range(1,2):
    #         for j in test_index:
    #             f.write('./CSI/classroom_data_unit/DX/O/gresture_O_location_' + str(i) + '_' + str(j) + '.npy 0\n')
    #             f.write('./CSI/classroom_data_unit/DX/PO/gresture_PO_location_' + str(i) + '_' + str(j) + '.npy 1\n')
    #             f.write('./CSI/classroom_data_unit/DX/X/gresture_X_location_' + str(i) + '_' + str(j) + '.npy 2\n')

    root = 'E:/CSI/CSI/classroom_data_unit/'
    # root = 'E:/project/CSI/classroom_data_unit/'
    #根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    train_data=MyDataset(root, 'train_data.txt', transform=transforms.ToTensor())
    test_data=MyDataset(root, 'test_data.txt', transform=transforms.ToTensor())
    print(train_data.__len__(), test_data.__len__())
    batch_size = 12
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # for epoch in range(2):
    #     for i, data in enumerate(train_loader):
    #         inputs, labels = data

    #         print(inputs.data.size(),labels.data.size())

    model = CNN()
    model.train()

    #设置不同初始学习率
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    # optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())
    # optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.1, parameter_list=model.parameters())
    criterion = nn.CrossEntropyLoss()
    EPOCH_NUM = 30
    loss_set = []
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader):
            #准备数据，变得更加简洁
            image, label = data
            # image = torch.from_numpy(image_data)
            # label = torch.from_numpy(label_data).squeeze()
            label = label.squeeze()
            # 清除梯度
            optimizer.zero_grad()
            #前向计算的过程
            predict = model(image)
            #计算损失，取一个批次样本损失的平均值
            loss = criterion(predict, label)
            loss_set.append(loss.detach().numpy())
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 2 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, loss.detach().numpy()))
            
            #后向传播，更新参数的过程
            loss.backward()
            optimizer.step()
    plt.plot(loss_set)
    plt.show()
    #保存模型参数
    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)

    model = CNN()
    model.load_state_dict(torch.load(PATH))

    model.eval()
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(test_loader):
        image, label = data
        # image = torch.from_numpy(images)
        # label = torch.from_numpy(labels).squeeze()
        label = label.squeeze()
        outputs = model(image)
        loss = F.cross_entropy(outputs, label)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == label).sum().item()/batch_size
        acc_set.append(acc)
        avg_loss_set.append(float(loss.detach().numpy()))
    
    #计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))
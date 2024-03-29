# -*- encoding: utf-8 -*-
'''
@File    :   gesture_recognition_3.5.py
@Time    :   2021/07/19 10:17:14
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
# 四个人，一个位置，巴特沃斯低通，30路子载波，一个天线对，81*30输入CNN。修改了网络，添加了一个全连接层。
# （模型不收敛可能是全连接层的输入输出分配不好，也可能是学习率的问题，目前0.001）
# 整理原始数据的读取方式
# 按不同人划分训练集和测试集
# ------------------------------ file details ------------------------------ #

# 加载相关库
import os
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
from sklearn.utils import shuffle
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


# !
def get_scale_csi(csi_st):
    # Pull out csi
    csi = csi_st['csi']
    # print(csi.shape)
    # print(csi)
    # Calculate the scale factor between normalized CSI and RSSI (mW)
    csi_sq = np.multiply(csi, np.conj(csi)).real
    csi_pwr = np.sum(csi_sq, axis=0)
    # csi_pwr = csi_pwr.reshape(1, csi_pwr.shape[0], -1)
    csi_pwr = np.reshape(csi_pwr, (csi_pwr.shape[0], -1))
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


# !

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
    scale_csi = np.empty((len(sample), 30, 3, 3), dtype=complex)
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
    csi = np.empty((len(scale_csi), 30, 3, 3))
    wn = 0.05
    order = 4
    # 引入butter函数
    b, a = signal.butter(order, wn, 'lowpass', analog=False)
    # i发射天线，j接收天线，k子载波序号
    for i in range(3):
        for j in range(3):
            for k in range(30):
                data = abs(scale_csi[:, k, i, j])
                csi[:, k, i, j] = signal.filtfilt(b, a, data, axis=0)

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
    csi_pca = np.empty((len(csi_abs), n_components, 3, 3))
    for i in range(3):
        for j in range(3):
            data = csi_abs[:, :, i, j]
            data = np.reshape(data, (data.shape[0], -1))  # 转换成二维矩阵
            pca.fit(data)
            data_pca = pca.transform(data)
            csi_pca[:, :, i, j] = data_pca[:, :]

    return csi_pca


def PCA_1(csi_abs, n_components, whiten):
    """
    @description  : PCA，30*3*3=270路子载波，得到一组主成分
    ---------
    @param  : csi_abs：csi的幅度矩阵，n_components：主成分数（1），whiten：是否白化True/False
    -------
    @Returns  : 返回主成分矩阵
    -------
    """

    pca = PCA(n_components=n_components, whiten=whiten)
    data = csi_abs
    data = np.reshape(data, (data.shape[0], -1))  # 转换成二维矩阵
    pca.fit(data)
    data_pca = pca.transform(data)

    return data_pca


def data_processing(path, feature_number, label):
    csi_data = np.empty((50, feature_number + 1))
    for i in range(50):
        # 样本路径
        filepath = path + str(i) + '.npy'
        # 读取样本
        scale_csi = read_sample(filepath)
        # 低通滤波
        csi_lowpass = butterworth_lowpass(scale_csi, 7, 0.01)
        # 不使用PCA选取子载波
        # 只选取天线对0-0
        csi_pca = csi_lowpass[:, :, 0, 0]
        # 截取长度800，步进10采样
        csi_vector = np.zeros((81, 30))
        if np.shape(csi_pca)[0] < 810:
            csi_empty = np.zeros((810, 30))
            csi_empty[:np.shape(csi_pca)[0]] = csi_pca[:, :]
            csi_vector[:] = csi_empty[::10, :]
        else:
            csi_pca = csi_pca[:809, :]
            csi_vector[:] = csi_pca[::10, :]
        # 添加标签
        csi_vector = np.reshape(csi_vector, (81, 30))
        csi_vector = np.reshape(csi_vector, (2430,))
        csi_data[i] = np.append(csi_vector, label)
        csi_data.dtype = 'float64'
        # 返回数据
    
        data = csi_data
    return data

class datatree(object):
    def __init__(self, name, gesture, location):
        super(datatree, self).__init__()

        self.name = name
        self.gesture = gesture
        self.location = location

    @property
    def name(self):
        return self._name

    @property
    def gesture(self):
        return self._gesture

    @property
    def location(self):
        return self._location

    @name.setter
    def name(self, name):
        self._name = name

    @gesture.setter
    def gesture(self, gesture):
        self._gesture = gesture

    @location.setter
    def location(self, location):
        self._location = location

    def __str__(self):
        return '姓名%s'%self._name+',动作%s'%self._gesture+',位置%s'%self._location


# 定义数据集读取器
def load_data(filepath, datatreelist ):
    # ! 读取数据文件
    # * 读取数据
    feature_number = 81 * 30
    # ! DX
    # 手势O，位置1
    filepath_O_1 = filepath + 'DX/O/gresture_O_location_1_'
    csi_DX_O_1 = data_processing(filepath_O_1, feature_number, 0)
    # 手势X，位置1
    filepath_X_1 = filepath + 'DX/X/gresture_X_location_1_'
    csi_DX_X_1 = data_processing(filepath_X_1, feature_number, 1)
    # 手势PO，位置1
    filepath_PO_1 = filepath + 'DX/PO/gresture_PO_location_1_'
    csi_DX_PO_1 = data_processing(filepath_PO_1, feature_number, 2)
    # 整合
    csi_DX_1 = np.array((csi_DX_O_1, csi_DX_X_1, csi_DX_PO_1))
    csi_DX_1 = np.reshape(csi_DX_1, (-1, feature_number + 1))  # ! 注意修改
    print(datetime.datetime.now())
    # ! LJP
    # 手势O，位置1
    filepath_O_1 = filepath + 'LJP/O/gresture_O_location_1_'
    csi_LJP_O_1 = data_processing(filepath_O_1, feature_number, 0)
    # 手势X，位置1
    filepath_X_1 = filepath + 'LJP/X/gresture_X_location_1_'
    csi_LJP_X_1 = data_processing(filepath_X_1, feature_number, 1)
    # 手势PO，位置1
    filepath_PO_1 = filepath + 'LJP/PO/gresture_PO_location_1_'
    csi_LJP_PO_1 = data_processing(filepath_PO_1, feature_number, 2)
    # 整合
    csi_LJP_1 = np.array((csi_LJP_O_1, csi_LJP_X_1, csi_LJP_PO_1))
    csi_LJP_1 = np.reshape(csi_LJP_1, (-1, feature_number + 1))
    print(datetime.datetime.now())
    # ! LZW
    # 手势O，位置1
    filepath_O_1 = filepath + 'LZW/O/gresture_O_location_1_'
    csi_LZW_O_1 = data_processing(filepath_O_1, feature_number, 0)
    # 手势X，位置1
    filepath_X_1 = filepath + 'LZW/X/gresture_X_location_1_'
    csi_LZW_X_1 = data_processing(filepath_X_1, feature_number, 1)
    # 手势PO，位置1
    filepath_PO_1 = filepath + 'LZW/PO/gresture_PO_location_1_'
    csi_LZW_PO_1 = data_processing(filepath_PO_1, feature_number, 2)
    # 整合
    csi_LZW_1 = np.array((csi_LZW_O_1, csi_LZW_X_1, csi_LZW_PO_1))
    csi_LZW_1 = np.reshape(csi_LZW_1, (-1, feature_number + 1))
    print(datetime.datetime.now())
    # ! MYW
    # 手势O，位置1
    # ? 只有手势O
    filepath_O_1 = filepath + 'MYW/O/gresture_O_location_1_'
    csi_MYW_O_1 = data_processing(filepath_O_1, feature_number, 0)
    # 整合
    csi_MYW_1 = np.array((csi_MYW_O_1))
    csi_MYW_1 = np.reshape(csi_MYW_1, (-1, feature_number + 1))
    print(datetime.datetime.now())
    # * 整合所有样本，乱序，分割
    # 整理数据集
    csi_1 = np.array((csi_LJP_1, csi_LZW_1))
    csi_1 = np.reshape(csi_1, (-1, feature_number + 1))
    csi_1 = np.append(csi_1, csi_MYW_1, axis=0)
    csi_1 = np.reshape(csi_1, (-1, feature_number + 1))
    # 分割特征和标签
    # train_feature, train_label = np.split(csi_1, (feature_number,), axis=1)
    # test_feature, test_label = np.split(csi_LZW_1, (feature_number,), axis=1)
    # train_feature, train_label = shuffle(train_feature, train_label, random_state=1)
    # test_feature, test_label = shuffle(test_feature, test_label, random_state=1)
    # feature, label = np.split(csi_1, (feature_number,),
    #                           axis=1)  # feature(150,5),label(150,1) #pylint: disable=unbalanced-tuple-unpacking #防止出现一条警告
    # 划分训练集和测试集
    # train_feature, test_feature, train_label, test_label = train_test_split(feature, label, random_state=1,
    #                                                         
    train_data = csi_1
    test_data = csi_DX_1

    return train_data, test_data



def load_dataset(mode='train', data=None, BATCHSIZE=15):
    # ! 读取数据文件
    # * 读取数据
    feature_number = 81 * 30
   
    # 分割特征和标签
    # train_feature, train_label = np.split(csi_1, (feature_number,), axis=1)
    # test_feature, test_label = np.split(csi_LZW_1, (feature_number,), axis=1)
    # train_feature, train_label = shuffle(train_feature, train_label, random_state=1)
    # test_feature, test_label = shuffle(test_feature, test_label, random_state=1)
    # feature, label = np.split(csi_1, (feature_number,),
    #                           axis=1)  # feature(150,5),label(150,1) #pylint: disable=unbalanced-tuple-unpacking #防止出现一条警告
    # 划分训练集和测试集
    # train_feature, test_feature, train_label, test_label = train_test_split(feature, label, random_state=1,
    #                                                                         test_size=0.3)

    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        train_feature, train_label = np.split(data, (feature_number,), axis=1)
        train_feature, train_label = shuffle(train_feature, train_label, random_state=1)
        imgs = train_feature
        labels = train_label
    elif mode == 'test':
        test_feature, test_label = np.split(data, (feature_number,), axis=1)
        test_feature, test_label = shuffle(test_feature, test_label, random_state=1)
        imgs = test_feature
        labels = test_label
    # 获得所有图像的数量
    imgs_length = len(imgs)
    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    # BATCHSIZE = 15

    # 定义数据生成器
    def data_generator():

        imgs_list = []
        labels_list = []
        # 按照索引读取数据 
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i], [1, 81, 30]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img)
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator



# 定义模型结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=5)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=5)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=5)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是10
        self.fc1 = nn.Linear(in_features=405, out_features=81)
        # 定义一层全连接层，输出维度是10
        self.fc2 = nn.Linear(in_features=81, out_features=18)
        # 定义一层全连接层，输出维度是10
        self.fc3 = nn.Linear(in_features=18, out_features=3)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = x.view([x.shape[0], 405])
        x = self.fc1(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x


if __name__ == '__main__':


    
    datatreeset = [datatree('DX', 'O', 1), datatree('DX', 'PO', 1), datatree('DX', 'X', 1),
                    datatree('LJP', 'O', 1), datatree('LJP', 'PO', 1), datatree('LJP', 'X', 1),
                    datatree('LZW', 'O', 1), datatree('LZW', 'PO', 1), datatree('LZW', 'X', 1),
                    datatree('MYW', 'O', 1)]
    # for datatree in datatreeset:
    #     print(datatree.name,datatree.gesture)
        
    # 仅优化算法的设置有所差别
    model = CNN()
    model.train()
    params = list(model.parameters())

    BATCHSIZE = 20
    # 调用加载数据的函数
    train_data, test_data = load_data('E:/CSI/CSI/classroom_data_unit/', datatreeset)
    train_loader = load_dataset('train', train_data, BATCHSIZE)
    # 设置不同初始学习率
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    print('train......')
    EPOCH_NUM = 100
    for epoch_id in range(EPOCH_NUM):
        acc_set = []
        avg_loss_set = []
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，变得更加简洁
            image_data, label_data = data
            image = torch.from_numpy(image_data)
            label = torch.from_numpy(label_data).squeeze()
            # 清除梯度
            optimizer.zero_grad()
            # 前向计算的过程
            predict = model(image)
            # 计算损失，取一个批次样本损失的平均值
            loss = criterion(predict, label)
            # 准确率
            _, predicted = torch.max(predict, 1)
            acc = (predicted == label).sum().item() / BATCHSIZE
            acc_set.append(acc)
            avg_loss_set.append(float(loss.detach().numpy()))

            # # 每训练了200批次的数据，打印下当前Loss的情况
            # if batch_id % 2 == 0:
            #     print("epoch: {}, batch: {}, loss is: {}, acc is: {}".format(epoch_id, batch_id, loss.detach().numpy(),
            #                                                                  acc))

            # 后向传播，更新参数的过程
            loss.backward()
            optimizer.step()
        # 计算多个batch的平均损失和准确率
        acc_val_mean = np.array(acc_set).mean()
        avg_loss_val_mean = np.array(avg_loss_set).mean()

        print('epoch: {}, loss={}, acc={}'.format(epoch_id, avg_loss_val_mean, acc_val_mean))
        
        # model.eval()
        # test_loader = load_dataset('test', test_data, BATCHSIZE)
        # acc_set = []
        # avg_loss_set = []
        # for batch_id, data in enumerate(test_loader()):
        #     images, labels = data
        #     image = torch.from_numpy(images)
        #     label = torch.from_numpy(labels).squeeze()
        #     outputs = model(image)
        #     loss = F.cross_entropy(outputs, label)
        #     _, predicted = torch.max(outputs, 1)
        #     acc = (predicted == label).sum().item() / BATCHSIZE
        #     acc_set.append(acc)
        #     avg_loss_set.append(float(loss.detach().numpy()))

        # # 计算多个batch的平均损失和准确率
        # acc_val_mean = np.array(acc_set).mean()
        # avg_loss_val_mean = np.array(avg_loss_set).mean()

        # print('test...., loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))

    # 保存模型参数
    PATH = '../model/gesture_recognition_3-4-t.pth'
    torch.save(model.state_dict(), PATH)

    model = CNN()
    model.load_state_dict(torch.load(PATH))
    print('test......')
    model.eval()
    test_loader = load_dataset('test', test_data, BATCHSIZE)
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(test_loader()):
        images, labels = data
        image = torch.from_numpy(images)
        label = torch.from_numpy(labels).squeeze()
        outputs = model(image)
        loss = F.cross_entropy(outputs, label)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == label).sum().item() / BATCHSIZE
        acc_set.append(acc)
        avg_loss_set.append(float(loss.detach().numpy()))

    # 计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))

    # 81*3*3
    # loss=0.6554504831631979, acc=0.899999996026357
    # loss=0.659913182258606, acc=0.8999999999999999
    # 81*30
    # BatchSize = 50 epoch = 30 loss=0.6156755884488424, acc=0.9533333333333333
    # BatchSize = 50 epoch = 50 loss=0.5701029102007548, acc=0.9933333333333333
    # BatchSize = 15 epoch = 50 loss=0.5590923130512238, acc=0.9933333333333334
    # BatchSize = 15 epoch = 50 loss=0.5587734162807465, acc=0.9933333333333334
    # BatchSize = 15 epoch = 50 loss=0.5524427711963653, acc=1.0
    # 不同人划分训练集测试集
    # DX测试，其他训练
    # 训练集 epoch: 49, batch: 22, loss is: 0.5519987940788269, acc is: 1.0
    # 测试集 loss=0.967512023448944, acc=0.5866666666666667
    # LJP测试
    # epoch: 49, batch: 16, loss is: 0.7997506260871887, acc is: 0.7333333333333333
    # epoch: 49, batch: 18, loss is: 0.8115711212158203, acc is: 0.7333333333333333
    # epoch: 49, batch: 20, loss is: 0.8097754120826721, acc is: 0.7333333333333333
    # epoch: 49, batch: 22, loss is: 1.0042685270309448, acc is: 0.5333333333333333
    # loss = 0.9660914719104767, acc = 0.6266666666666666
    # epoch: 49, batch: 22, loss is: 0.5522249937057495, acc is: 1.0
    # loss = 1.2215073466300965, acc = 0.33333333333333337
    # LZW测试
    # epoch: 49, batch: 22, loss is: 0.5517404675483704, acc is: 1.0
    # loss=1.2179097414016724, acc=0.33333333333333337

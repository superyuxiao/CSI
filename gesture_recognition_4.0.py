# -*- coding: utf-8 -*-
# @Author   : YuXiao 于潇
# @Time     : 2021/7/20 10:09 下午
# @File     : gesture_recognition_4.0.py
# @Project  : CSI
# @Contact  : superyuxiao@icloud.com
# @License  : (C)Copyright 2020-2021, Key Laboratory of University Wireless Communication
#                Beijing University of Posts and Telecommunications

# --------------------------- file details --------------------------- #
# RNN
#
# --------------------------- file details --------------------------- #

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
        # 去除直射径
        # scale_csi[:, :, 0, 0] = scale_csi[:, :, 0, 0] - scale_csi[:, :, 1, 0]
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


# 定义数据集读取器
def load_data(filepath=None):
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
    csi_1 = np.array((csi_LJP_1, csi_DX_1, csi_LZW_1))
    csi_1 = np.reshape(csi_1, (-1, feature_number + 1))
    csi_1 = np.append(csi_1, csi_MYW_1, axis=0)
    csi_1 = np.reshape(csi_1, (-1, feature_number + 1))
    # 分割特征和标签
    # train_feature, train_label = np.split(csi_1, (feature_number,), axis=1)
    # test_feature, test_label = np.split(csi_LZW_1, (feature_number,), axis=1)
    # train_feature, train_label = shuffle(train_feature, train_label, random_state=1)
    # test_feature, test_label = shuffle(test_feature, test_label, random_state=1)
    feature, label = np.split(csi_1, (feature_number,),
                              axis=1)  # feature(150,5),label(150,1) #pylint: disable=unbalanced-tuple-unpacking #防止出现一条警告
    # 划分训练集和测试集
    train_feature, test_feature, train_label, test_label = train_test_split(feature, label, random_state=1,
                                                                            test_size=0.3)
    return train_feature, test_feature, train_label, test_label


def load_dataset(mode='train', train_feature=None, test_feature=None, train_label=None, test_label=None, BATCHSIZE=15):
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_feature
        labels = train_label
    elif mode == 'test':
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


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_class):
        super(RNN,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # input: (batch_size, sequence_size, input_size)
        # many to one mode
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self,x):
        # initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0),self.hidden_dim)
        # output size: (batch_size, sequnce_size, hidden_size)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


if __name__ == '__main__':

    BATCHSIZE = 15
    # 调用加载数据的函数
    # train_feature, test_feature, train_label, test_label = load_data('E:/CSI/CSI/classroom_data_unit/')
    # train_feature, test_feature, train_label, test_label = load_data('/Users/yuxiao/CSI_data/classroom_data_unit/')
    # train_loader = load_dataset(mode='train', train_feature=train_feature, train_label=train_label, BATCHSIZE=BATCHSIZE)

    filepath = '/Users/yuxiao/CSI_data/classroom_data_unit/DX/O/gresture_O_location_1_0.npy'
    a = read_sample(filepath)
    input_data = torch.from_numpy(abs(a[:, 0, 0, 0]))
    input_data = np.reshape(input_data, (1, -1, 1))
    input_data = torch.tensor(input_data, dtype=torch.float32)
    model = RNN(1, 32, 2, 3)
    params = list(model.parameters())
    predict = model(input_data)

    # # 设置不同初始学习率
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # criterion = nn.CrossEntropyLoss()
    # EPOCH_NUM = 10
    # for epoch_id in range(EPOCH_NUM):
    #     acc_set = []
    #     avg_loss_set = []
    #     for batch_id, data in enumerate(train_loader()):
    #         # 准备数据，变得更加简洁
    #         image_data, label_data = data
    #         image = torch.from_numpy(image_data)
    #         label = torch.from_numpy(label_data).squeeze()
    #         # 清除梯度
    #         optimizer.zero_grad()
    #         # 前向计算的过程
    #         predict = model(image)
    #         # 计算损失，取一个批次样本损失的平均值
    #         loss = criterion(predict, label)
    #         # 准确率
    #         _, predicted = torch.max(predict, 1)
    #         acc = (predicted == label).sum().item() / BATCHSIZE
    #         acc_set.append(acc)
    #         avg_loss_set.append(float(loss.detach().numpy()))
    #
    #         # 每训练了200批次的数据，打印下当前Loss的情况
    #         if batch_id % 2 == 0:
    #             print("epoch: {}, batch: {}, loss is: {}, acc is: {}".format(epoch_id, batch_id, loss.detach().numpy(),acc))
    #
    #         # 后向传播，更新参数的过程
    #         loss.backward()
    #         optimizer.step()
    #     # 计算多个batch的平均损失和准确率
        # acc_val_mean = np.array(acc_set).mean()
        # avg_loss_val_mean = np.array(avg_loss_set).mean()
        #
        # print('epoch: {}, loss={}, acc={}'.format(epoch_id, avg_loss_val_mean, acc_val_mean))

    # # 保存模型参数
    # PATH = 'model/gesture_recognition_3-6.pth'
    # torch.save(model.state_dict(), PATH)
    #
    # model = CNN()
    # model.load_state_dict(torch.load(PATH))
    # print('test......')
    # model.eval()
    # test_loader = load_dataset(mode='test', test_feature= test_feature,test_label= test_label,BATCHSIZE= BATCHSIZE)
    #
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
    #
    # # 计算多个batch的平均损失和准确率
    # acc_val_mean = np.array(acc_set).mean()
    # avg_loss_val_mean = np.array(avg_loss_set).mean()
    #
    # print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))



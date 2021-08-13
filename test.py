#!E:\Python\Python368-64\python.exe
# -*- encoding: utf-8 -*-
'''
@File    :   gesture_recognition_3.6.py
@Time    :   2021/07/19 17:12:38
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
# 特征增强，相关信息提取
# ------------------------------ file details ------------------------------ #

# 加载相关库
import os
import random
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
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import datetime
import math
import seaborn as sns
from get_scale_csi import get_scale_csi


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
    Den_set = np.zeros([50, feature_number + 1])
    for i in range(50):
        # 样本路径
        filepath = path + str(i + 1) + '.npy'
        # 读取样本
        scale_csi = read_sample(filepath)
        # 取1*3天线对
        raw_csi = scale_csi[:, :, 0, :]
        raw_csi = np.reshape(raw_csi, [raw_csi.shape[0], 30, 1, 3])
        # print(raw_csi.shape)
        st_csi = np.zeros_like(raw_csi)
        # 遗忘因子
        theta = 0.5
        # theta = 0.5 acc = 0.96
        # theta = 0.4 acc = 0.9733
        # theta = 0.1 acc = 0.9933
        # theta = 0.05 acc = 0.9933
        for j in range(1, raw_csi.shape[0]):
            st_csi[j, :, :, :] = theta * raw_csi[j, :, :, :] + (1 - theta) * st_csi[j - 1, :, :, :]
        dy_csi = raw_csi - st_csi
        # 整理shape
        abs_dy_csi = abs(dy_csi)
        abs_dy_csi = np.reshape(abs_dy_csi, [abs_dy_csi.shape[0], 30, 3])
        abs_dy_csi = np.reshape(abs_dy_csi, [abs_dy_csi.shape[0], 90])
        # print(abs_dy_csi.shape)
        # 分段
        k = 2
        split_index = [i for i in range(int(abs_dy_csi.shape[0] / k), abs_dy_csi.shape[0], int(abs_dy_csi.shape[0] / k))]
        segment_dy_csi = np.split(abs_dy_csi, split_index, axis=0)
        if np.shape(segment_dy_csi[0]) != np.shape(segment_dy_csi[-1]):
            segment_dy_csi = segment_dy_csi[:-1]  # 去除最后一个， 保证各片段等长度
        # 相关计算
        cross_segment = []
        for m in range(len(segment_dy_csi)):
            for n in range(m, len(segment_dy_csi)):
                # t = np.mean(segment_dy_csi[m])
                u1 = segment_dy_csi[m] - np.mean(segment_dy_csi[m])
                u2 = segment_dy_csi[n] - np.mean(segment_dy_csi[n])
                mul = np.matmul(np.transpose(u1), u2)
                cross_segment.append(mul)
        cross_segment = np.array(cross_segment)
        cross_segment = np.reshape(cross_segment, [cross_segment.shape[1], -1])
        # 缩小尺寸
        Den = np.matmul(cross_segment, np.transpose(cross_segment))
        # hotmap
        if(i == 42):
            sns.heatmap(Den / 10000000000, cmap='Reds')
            plt.title(path)
            plt.show()
            print(i)
        Den = np.reshape(Den, [-1, ])
        Den = Den / 10000000000
        Den = np.append(Den, label)
        Den_set[i] = Den

    return Den_set


# 定义数据集读取器
def load_data(filepath=None):
    # ! 读取数据文件
    # * 读取数据
    feature_number = 90 * 90
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
    csi_2 = np.array((csi_DX_1))
    csi_2 = np.reshape(csi_2, (-1, feature_number + 1))
    # csi_2 = np.append(csi_2, csi_MYW_1, axis=0)
    # csi_2 = np.reshape(csi_2, (-1, feature_number + 1))
    # 分割特征和标签
    train_feature, train_label = np.split(csi_1, (feature_number,), axis=1)
    test_feature, test_label = np.split(csi_2, (feature_number,), axis=1)
    train_feature, train_label = shuffle(train_feature, train_label, random_state=1)
    test_feature, test_label = shuffle(test_feature, test_label, random_state=1)
    # feature, label = np.split(csi_1, (feature_number,),
    #                           axis=1)  # feature(150,5),label(150,1) #pylint: disable=unbalanced-tuple-unpacking #防止出现一条警告
    # # 划分训练集和测试集
    # train_feature, test_feature, train_label, test_label = train_test_split(feature, label, random_state=1,
    #                                                                         test_size=0.3)
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
            img = np.reshape(imgs[i], [1, 90, 90]).astype('float32')
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
        self.fc1 = nn.Linear(in_features=768, out_features=192)
        # 定义一层全连接层，输出维度是10
        self.fc2 = nn.Linear(in_features=192, out_features=48)
        # 定义一层全连接层，输出维度是10
        self.fc3 = nn.Linear(in_features=48, out_features=3)

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
        x = x.view([x.shape[0], 768])
        x = self.fc1(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x


if __name__ == '__main__':

    filepath = '/Users/yuxiao/CSI_data/classroom_data_unit/'
    feature_number = 90 * 90
    # ! DX
    # 手势O，位置1
    path = filepath + 'LZW/O/gresture_O_location_1_'
    # csi_DX_O_1 = data_processing(filepath_O_1, feature_number, 0)

    for i in range(31, 32):
        # 样本路径
        filepath = path + str(i + 1) + '.npy'
        # 读取样本
        scale_csi = read_sample(filepath)
        # 取0-0天线对
        raw_csi = scale_csi[:, :, 0, 0:3]
        print(raw_csi.shape)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 15))
        sns.heatmap(np.transpose(abs(raw_csi[:, :, 0])), cmap='Reds', ax=ax1)
        sns.heatmap(np.transpose(abs(raw_csi[:, :, 1])), cmap='Reds', ax=ax2)
        # csi ratio, 存在分母为0的情况，经过arctan后可以从无穷变为pi/2
        ratio_csi = np.arctan(raw_csi[:, :, 0] / raw_csi[:, :, 1])
        sns.heatmap(np.transpose(abs(ratio_csi)), cmap='Reds', ax=ax3)
        # ax3.plot(np.arctan(abs(ratio_csi)))
        plt.show()

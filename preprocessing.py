#!E:\Python\Python368-64\python.exe
# -*- encoding: utf-8 -*-
'''
@File    :   preprocessing.py
@Time    :   2021/04/16 15:02:21
@Author  :   Yu Xiao 于潇 
@Version :   1.0
@Contact :   superyuxiao@icloud.com
@License :   (C)Copyright 2020-2021, Key Laboratory of University Wireless Communication
                Beijing University of Posts and Telecommunications
@Desc    :   None
'''

# ------------------------------ file details ------------------------------ #
# 数据预处理
# 去噪 
# 降维
# ------------------------------ file details ------------------------------ #

import numpy as np
from sklearn.decomposition import PCA
from scipy import signal
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


# 不同人不同位置具有相同的数据处理过程
# 根据不同工程，对应修改函数代码
def ratio(path, feature_number, label):
    csi_data = np.empty((50, feature_number + 1))
    for i in range(50):
        # 样本路径
        filepath = path + str(i) + '.npy'
        # 读取样本
        scale_csi = read_sample(filepath)
        # ! 去除前20帧
        scale_csi = scale_csi[20:, :, :, :]
        # print(np.shape(scale_csi))
        ones_csi = np.ones((800, 30, 3, 3))
        ones_csi.dtype = 'float64'
        # ! 截取长度800
        if np.shape(scale_csi)[0] < 800:
            scale_csi = ones_csi
        else:
            scale_csi = scale_csi[:800, :, :]
        # print(np.shape(scale_csi))
        # ! 求csi ratio
        csi_ratio = scale_csi[:, :, 0, 0] / scale_csi[:, :, 0, 1]
        # print(np.shape(csi_ratio))
        # csi ratio phase
        csi_ratio_phase = np.unwrap(np.angle(np.transpose(csi_ratio)))
        # ! 归一化
        # normalizer = MinMaxScaler()
        # csi_normalize = normalizer.fit_transform(csi_ratio_phase)
        # csi_normalize = minmax_scale(csi_ratio_phase,axis=3)
        csi_max = np.max(csi_ratio_phase)
        csi_min = np.min(csi_ratio_phase)
        csi_normalize = (csi_ratio_phase - csi_min) / (csi_max - csi_min)
        # 添加标签
        csi_vector = np.reshape(csi_normalize, (24000,))
        csi_data[i] = np.append(csi_vector, label)
        csi_data.dtype = 'float64'
        # 返回数据
    data = csi_data

    return data


# 单天线对-30路子载波拼接二维
def mul_subcarries(path, feature_number, label):
    csi_data = np.empty((50, feature_number + 1))
    for i in range(50):
        # 样本路径
        filepath = path + str(i + 1) + '.npy'
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

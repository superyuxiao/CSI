#!E:\Python\Python368-64\python.exe
# -*- encoding: utf-8 -*-
'''
@File    :   PCA.py
@Time    :   2021/01/13 11:49:41
@Author  :   Yu Xiao 于潇 
@Version :   1.0
@Contact :   superyuxiao@icloud.com
@License :   (C)Copyright 2020-2021, Key Laboratory of University Wireless Communication
                Beijing University of Posts and Telecommunications
@Desc    :   None
'''

# ------------------------------ file details ------------------------------ #
# PCA
# ------------------------------ file details ------------------------------ #

from get_scale_csi import get_scale_csi
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# !读取样本
gresture_O_location_1_6 = np.load('CSI/classroom_data_unit/DX/O/gresture_O_location_1_6.npy',allow_pickle=True)
# !样本长度
t = np.arange(0,len(gresture_O_location_1_6))
# !设置csi容器，格式为样本长度（帧数）*子载波数30*发送天线3*接收天线3，复数
csi = np.empty((len(gresture_O_location_1_6),30,3,3), dtype = complex)
# !逐帧将csi归一化
for i in range(len(gresture_O_location_1_6)):
    csi[i] = get_scale_csi(gresture_O_location_1_6[i])
# # ! PCA
# data = abs(csi[:,:,0,0])
# data = np.reshape(data, (data.shape[0],-1))
# print(data.shape)
# pca = PCA(n_components=1, whiten=True)
# pca.fit(data)
# print(pca.explained_variance_ratio_)
# data_pca = pca.transform(data)
# print(data_pca.shape)
# for i in range(data_pca.shape[1]):
#     print(i)
#     subcarrier = data_pca[:,i]
#     plt.title("PCA Waveform")
#     plt.plot(subcarrier) # 幅度
# plt.show()

#! 以子载波为横轴
data = abs(csi[:,:,0,0])
data = np.reshape(data, (data.shape[0],-1))
data = np.transpose(data)
print(data.shape)
for i in range(data.shape[1]):
    # print(i)
    subcarrier = data[:,i]
    plt.title("PCA Waveform")
    plt.plot(subcarrier) # 幅度
# pca = PCA(n_components=0.99, whiten=True)
# pca.fit(data)
# print(pca.explained_variance_ratio_)
# data_pca = pca.transform(data)
# print(data_pca.shape)
# for i in range(data_pca.shape[1]):
#     print(i)
#     subcarrier = data_pca[:,i]
#     plt.title("PCA Waveform")
#     plt.plot(subcarrier) # 幅度
plt.show()
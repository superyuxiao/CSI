#!E:\Python\Python368-64\python.exe
# -*- encoding: utf-8 -*-
'''
@File    :   read_sample.py
@Time    :   2021/01/13 11:22:22
@Author  :   Yu Xiao 于潇 
@Version :   1.0
@Contact :   superyuxiao@icloud.com
@License :   (C)Copyright 2020-2023, Key Laboratory of University Wireless Communication
                Beijing University of Posts and Telecommunications
@Desc    :   None
'''

# ------------------------------ file details ------------------------------ #
# 读取分割后的数据样本，绘出未处理的幅度图或相位图。
# ------------------------------ file details ------------------------------ #

from matplotlib.pyplot import plot
import numpy as np
from numpy import random
from Bfee import Bfee
from get_scale_csi import get_scale_csi
from sklearn import model_selection
import matplotlib.pyplot as plt

# !读取样本
gresture_O_location_1_6 = np.load('classroom_data_unit/DX/O/gresture_O_location_1_6.npy',allow_pickle=True)
# !样本长度
t = np.arange(0,len(gresture_O_location_1_6))
# !设置csi容器，格式为样本长度（帧数）*子载波数30*发送天线3*接收天线3，复数
csi = np.empty((len(gresture_O_location_1_6),30,3,3), dtype = complex)
# !逐帧将csi归一化
for i in range(len(gresture_O_location_1_6)):
    csi[i] = get_scale_csi(gresture_O_location_1_6[i])
#print(csi.shape)
# !以子载波为曲线单位，画出幅度或相位变化。横坐标为帧序号（时间）
for i in range(25,30):
    # !天线对可选，默认0-0
    subcarrier = csi[:,i,0,0]
    plt.plot(t, abs(subcarrier)) # *幅度
    #plt.plot(np.arctan(subcarrier.imag/subcarrier.real)/1.5707963) # *相位 

""" for i in range(25,30):
    # !天线对可选，默认0-0
    subcarrier = csi[:,i,0,0]
    #plt.scatter(t, abs(subcarrier), c = abs(subcarrier))
    plt.scatter(t, np.arctan(subcarrier.imag/subcarrier.real)/1.5707963, c = np.arctan(subcarrier.imag/subcarrier.real)/1.5707963)
plt.colorbar()  """

plt.show()
    
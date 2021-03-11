#!E:\Python\Python368-64\python.exe
# -*- encoding: utf-8 -*-
'''
@File    :   butterworth.py
@Time    :   2021/01/13 11:42:18
@Author  :   Yu Xiao 于潇 
@Version :   1.0
@Contact :   superyuxiao@icloud.com
@License :   (C)Copyright 2020-2021, Key Laboratory of University Wireless Communication
                Beijing University of Posts and Telecommunications
@Desc    :   None
'''

# ------------------------------ file details ------------------------------ #
# butterworth 低通滤波
# ------------------------------ file details ------------------------------ #


from get_scale_csi import get_scale_csi
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math

# !读取样本
gresture_O_location_1_6 = np.load('classroom_data_unit/DX/O/gresture_O_location_1_6.npy',allow_pickle=True)
# !样本长度
t = np.arange(0,len(gresture_O_location_1_6))
# !设置csi容器，格式为样本长度（帧数）*子载波数30*发送天线3*接收天线3，复数
csi = np.empty((len(gresture_O_location_1_6),30,3,3), dtype = complex)
# !逐帧将csi归一化
for i in range(len(gresture_O_location_1_6)):
    csi[i] = get_scale_csi(gresture_O_location_1_6[i])
# ! butterworth for all subcarrier
cutoff = 40000
fs = 200
wn = 0.05
order = 4
b, a = signal.butter(order, wn, 'lowpass', analog = False)
for i in range(30):
    data = abs(csi[:,i,0,0])
    output = signal.filtfilt(b, a, data, axis=0)
    plt.title("butterworth for all subcarrier")
    plt.plot(output)
plt.show()
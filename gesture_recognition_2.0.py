#!E:\Python\Python368-64\python.exe
# -*- encoding: utf-8 -*-
'''
@File    :   gesture_recognition.py
@Time    :   2021/01/28 17:29:09
@Author  :   Yu Xiao 于潇 
@Version :   2.0
@Contact :   superyuxiao@icloud.com
@License :   (C)Copyright 2020-2021, Key Laboratory of University Wireless Communication
                Beijing University of Posts and Telecommunications
@Desc    :   None
'''

# ------------------------------ file details ------------------------------ #
# 1. 区分三种手势（X，O，PO）
# 2. butterworth低通滤波（阶数，wn的取值
# 3. PCA降维（如何选取主成分，天线对之间如何取舍，30路与270路子载波主成分是不同的
# 4. 提取特征（均值，绝对均值，标准差，最大值，最小值
# 5. 训练模型（决策树
# ------------------------------ file details ------------------------------ #
from matplotlib.pyplot import subplot
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy import signal
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
    csi_pwr = csi_pwr.reshape(1, csi_pwr.shape[0], -1)
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


def plt_9_amplitude(scale_csi_abs, subcarries_range):
    subplot(3,3,1)
    for i in subcarries_range:
        plt.plot(scale_csi_abs[:,i,0,0])
    subplot(3,3,2)
    for i in subcarries_range:
        plt.plot(scale_csi_abs[:,i,0,1])
    subplot(3,3,3)
    for i in subcarries_range:
        plt.plot(scale_csi_abs[:,i,0,2])
    subplot(3,3,4)
    for i in subcarries_range:
        plt.plot(scale_csi_abs[:,i,1,0])
    subplot(3,3,5)
    for i in subcarries_range:
        plt.plot(scale_csi_abs[:,i,1,1])
    subplot(3,3,6)
    for i in subcarries_range:
        plt.plot(scale_csi_abs[:,i,1,2])
    subplot(3,3,7)
    for i in subcarries_range:
        plt.plot(scale_csi_abs[:,i,2,0])
    subplot(3,3,8)
    for i in subcarries_range:
        plt.plot(scale_csi_abs[:,i,2,1])
    subplot(3,3,9)
    for i in subcarries_range:
        plt.plot(scale_csi_abs[:,i,2,2])
    plt.show()


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
    data = np.reshape(data, (data.shape[0],-1)) #转换成二维矩阵
    pca.fit(data)
    data_pca = pca.transform(data)

    return data_pca


def csi_feature(csi_pca):

    #* 均值每次计算都不一样，值太小了，需要排除一下0点附近的值，每个天线对分开计算
    csi_9_mean = np.empty((1,3,3))
    for i in range(3):
        for j in range(3):
            data = csi_pca[:,:,i,j]
            data = data[np.abs(data)>1]
            csi_9_mean[0,i,j] = np.mean(data)
    #* 绝对均值
    csi_9_absmean = np.mean(abs(csi_pca), axis=0)
    #* 标准差
    csi_9_std = np.std(csi_pca, axis=0, ddof=1)
    #* 极大值
    csi_9_max = np.max(csi_pca, axis=0)
    #* 极小值
    csi_9_min = np.min(csi_pca, axis=0)

    csi_9_feature = np.array((csi_9_mean, csi_9_absmean, csi_9_std, csi_9_max, csi_9_min))

    return csi_9_feature


if __name__ == '__main__':

    #* 记录程序运行时间，开始时间
    starttime = datetime.datetime.now()
    print(starttime)
    # 不用科学计数法显示
    np.set_printoptions(suppress=True)
    csi = np.empty((3,5,50,800,30,3,3))
    filepath = 'classroom_data_unit/DX/O/gresture_O_location_1_23.npy'
    scale_csi = read_sample(filepath)
    print(csi.shape)
    csi[1,1,1,:,:,1,1] = scale_csi[20:820,:,1,1]
    print(csi[1,1,1,:,:,1,1])
    print(csi.shape)
    #* 记录程序运行时间，结束时间
    endtime = datetime.datetime.now()
    print("程序运行时间：", endtime - starttime)
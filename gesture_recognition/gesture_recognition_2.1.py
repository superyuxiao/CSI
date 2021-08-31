#!E:\Python\Python368-64\python.exe
# -*- encoding: utf-8 -*-
'''
@File    :   gesture_recognition_2.1.py
@Time    :   2021/04/12 20:55:56
@Author  :   Yu Xiao 于潇 
@Version :   1.0
@Contact :   superyuxiao@icloud.com
@License :   (C)Copyright 2020-2021, Key Laboratory of University Wireless Communication
                Beijing University of Posts and Telecommunications
@Desc    :   None
'''

# ------------------------------ file details ------------------------------ #
# 增加统计特征
# ------------------------------ file details ------------------------------ #

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import datetime
import math
import os

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
    #* 均方根
    csi_9_rms = np.sqrt(np.sum(np.square(csi_pca), axis=0)/np.shape(csi_pca)[0])
    #* 上四分位矩
    csi_9_up_quantile = np.quantile(csi_pca , 0.75 , interpolation='higher', axis=0)
    #* 中点
    csi_9_mid_quantile = np.quantile(csi_pca , 0.5 , interpolation='midpoint', axis=0)
    #* 下四分位矩
    csi_9_down_quantile = np.quantile(csi_pca , 0.25 , interpolation='lower', axis=0)
    #* 偏度skewness
    csi_9_skew = stats.skew(csi_pca)
    #* 峰度kurtosis
    csi_9_kurtosis = stats.kurtosis(csi_pca)

    csi_9_feature = np.array((csi_9_mean, csi_9_absmean, csi_9_std, csi_9_max, csi_9_min, csi_9_rms, 
                                csi_9_up_quantile, csi_9_mid_quantile, csi_9_down_quantile, csi_9_skew, csi_9_kurtosis))

    return csi_9_feature

# 不同人不同位置具有相同的数据处理过程
# 根据不同工程，对应修改函数代码
def data_processing(path, label):
    csi_data = np.empty((50,12))
    for i in range(50):
        # 样本路径
        filepath = path + str(i) +'.npy'
        # 读取样本
        scale_csi = read_sample(filepath)
        # 低通滤波
        csi_lowpass = butterworth_lowpass(scale_csi, 7, 0.01)
        # PCA
        csi_pca_9 = PCA_9(csi_abs=csi_lowpass, n_components=1, whiten=False)
        # 画幅度图
        #plt_9_amplitude(csi_pca_9,range(1))
        # 特征提取
        csi_feature_9_5 = csi_feature(csi_pca_9)
        # 只选取天线对0-0
        csi_feature_5 = csi_feature_9_5[:,0,0,0]
        # 添加标签
        csi_data[i] = np.append(csi_feature_5, label)
        csi_data.dtype = 'float64'
        # 返回数据
        data = csi_data
    return data

# 整合所有数据
def get_feature_label():

    #* 读取数据
    #! DX
    # 手势O，位置1
    filepath_O_1 = 'CSI/classroom_data_unit/DX/O/gresture_O_location_1_'
    csi_DX_O_1 = data_processing(filepath_O_1, 0)
    # 手势X，位置1
    filepath_X_1 = 'CSI/classroom_data_unit/DX/X/gresture_X_location_1_'
    csi_DX_X_1 = data_processing(filepath_X_1, 1)
    # 手势PO，位置1
    filepath_PO_1 = 'CSI/classroom_data_unit/DX/PO/gresture_PO_location_1_'
    csi_DX_PO_1 = data_processing(filepath_PO_1, 2)
    # 整合
    csi_DX_1 = np.array((csi_DX_O_1, csi_DX_X_1, csi_DX_PO_1))
    csi_DX_1 = np.reshape(csi_DX_1, (-1,12))
    print(datetime.datetime.now())
    #! LJP
    # 手势O，位置1
    filepath_O_1 = 'CSI/classroom_data_unit/LJP/O/gresture_O_location_1_'
    csi_LJP_O_1 = data_processing(filepath_O_1, 0)
    # 手势X，位置1
    filepath_X_1 = 'CSI/classroom_data_unit/LJP/X/gresture_X_location_1_'
    csi_LJP_X_1 = data_processing(filepath_X_1, 1)
    # 手势PO，位置1
    filepath_PO_1 = 'CSI/classroom_data_unit/LJP/PO/gresture_PO_location_1_'
    csi_LJP_PO_1 = data_processing(filepath_PO_1, 2)
    # 整合
    csi_LJP_1 = np.array((csi_LJP_O_1, csi_LJP_X_1, csi_LJP_PO_1))
    csi_LJP_1 = np.reshape(csi_LJP_1, (-1,12))
    print(datetime.datetime.now())
    #! LZW
    # 手势O，位置1
    filepath_O_1 = 'CSI/classroom_data_unit/LZW/O/gresture_O_location_1_'
    csi_LZW_O_1 = data_processing(filepath_O_1, 0)
    # 手势X，位置1
    filepath_X_1 = 'CSI/classroom_data_unit/LZW/X/gresture_X_location_1_'
    csi_LZW_X_1 = data_processing(filepath_X_1, 1)
    # 手势PO，位置1
    filepath_PO_1 = 'CSI/classroom_data_unit/LZW/PO/gresture_PO_location_1_'
    csi_LZW_PO_1 = data_processing(filepath_PO_1, 2)
    # 整合
    csi_LZW_1 = np.array((csi_LZW_O_1, csi_LZW_X_1, csi_LZW_PO_1))
    csi_LZW_1 = np.reshape(csi_LZW_1, (-1,12))
    print(datetime.datetime.now())
    #! MYW
    # 手势O，位置1
    #? 只有手势O
    filepath_O_1 = 'CSI/classroom_data_unit/MYW/O/gresture_O_location_1_'
    csi_MYW_O_1 = data_processing(filepath_O_1, 0)
    # 整合
    csi_MYW_1 = np.array((csi_MYW_O_1))
    csi_MYW_1 = np.reshape(csi_MYW_1, (-1,12))
    print(datetime.datetime.now())
    #* 整合所有样本，乱序，分割
    # 整理数据集
    csi_1 = np.array((csi_DX_1, csi_LJP_1, csi_LZW_1))
    csi_1 = np.reshape(csi_1, (-1,12))
    csi_1 = np.append(csi_1, csi_MYW_1, axis=0)
    csi_1 = np.reshape(csi_1, (-1,12))
    # 分割特征和标签
    feature, label = np.split(csi_1, (11,), axis=1) #feature(150,5),label(150,1) #pylint: disable=unbalanced-tuple-unpacking #防止出现一条警告

    return feature, label

if __name__ == '__main__':

    #* 记录程序运行时间，开始时间
    starttime = datetime.datetime.now()
    print(starttime)
    #? 不用科学计数法显示
    np.set_printoptions(suppress=True)
    #? 缓存数据
    path = os.path.split(os.path.abspath(__file__))[0] + '/cache/' + os.path.split(os.path.abspath(__file__))[1].split('.')[0]
    feature_name = path + '/feature'
    label_name = path + '/label'
    is_feature_exists = os.path.exists(feature_name + '.npy')
    is_label_exists = os.path.exists(label_name + '.npy')
    if is_feature_exists and is_label_exists:
        feature = np.load(feature_name + '.npy', allow_pickle=True)
        label = np.load(label_name + '.npy', allow_pickle=True)
    else:
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path) 
            print(path +' 创建成功')
        else:
            print(path +' 目录已存在')
        #! 数据
        feature, label = get_feature_label()
        #! 保存
        np.save(feature_name, feature)
        np.save(label_name, label)
    # 划分训练集和测试集
    train_feature, test_feature, train_label, test_label = train_test_split(feature, label, random_state=1, test_size=0.3)
    #* 识别方法
    #! 决策树
    # 建立模型
    DTtree = DecisionTreeClassifier()
    print('========================决策树============================')
    # 训练模型
    DTtree = DTtree.fit(train_feature, train_label)
    # 准确率
    score_train = DTtree.score(train_feature, train_label)
    print('模型训练准确率：', format(score_train))
    score_test = DTtree.score(test_feature, test_label)
    print('模型预测准确率：', format(score_test))
    # 精确率 召回率 调和均值F1
    pred_label = DTtree.predict(test_feature)
    report = classification_report(test_label, pred_label)
    print(report)
    cm = confusion_matrix(test_label, pred_label, labels=DTtree.classes_)
    # cm = cm / (sum(sum(cm))/cm.shape[0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DTtree.classes_)
    disp.plot()
    plt.show()
    #! 提升树
    # 建立模型
    #model = AdaBoostClassifier(n_estimators=200, random_state=0)
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=2, random_state=0)
    print('========================提升树============================')
    # 训练模型
    train_label = np.reshape(train_label, (-1))
    model.fit(train_feature, train_label)
    # 准确率
    score_train = model.score(train_feature, train_label)
    print('模型训练准确率：', format(score_train))
    score_test = model.score(test_feature, test_label)
    print('模型预测准确率：', format(score_test))
    # 精确率 召回率 调和均值F1
    pred_label = model.predict(test_feature)
    report = classification_report(test_label, pred_label)
    print(report)
    # plot_confusion_matrix(model, X=test_feature, y_true=test_label)  
    # plt.show()
    #! 支持向量机
    # 建立模型
    svm_model = svm.NuSVC()
    print('========================SVM============================')
    # 训练模型
    train_label = np.reshape(train_label, (-1))
    svm_model.fit(train_feature, train_label)
    # 准确率
    score_train = svm_model.score(train_feature, train_label)
    print('模型训练准确率：', format(score_train))
    score_test = svm_model.score(test_feature, test_label)
    print('模型预测准确率：', format(score_test))
    # 精确率 召回率 调和均值F1
    pred_label = svm_model.predict(test_feature)
    report = classification_report(test_label, pred_label)
    print(report)
    # plot_confusion_matrix(svm_model, X=test_feature, y_true=test_label)  
    # plt.show()
    #! 决策分析
    # 建立模型
    DA_model = LinearDiscriminantAnalysis()
    print('========================决策分析============================')
    # 训练模型
    DA_model.fit(train_feature, train_label)
    # 准确率
    score_train = DA_model.score(train_feature, train_label)
    print('模型训练准确率：', format(score_train))
    score_test = DA_model.score(test_feature, test_label)
    print('模型预测准确率：', format(score_test))
    # 精确率 召回率 调和均值F1
    pred_label = DA_model.predict(test_feature)
    report = classification_report(test_label, pred_label)
    print(report)
    # plot_confusion_matrix(DA_model, X=test_feature, y_true=test_label)  
    # plt.show()
    #! KNN  
    # 建立模型
    knn_model = KNeighborsClassifier(n_neighbors=2)
    print('========================K近邻============================')
    # 训练模型
    train_label = np.reshape(train_label, (-1))
    knn_model.fit(train_feature, train_label)
    # 准确率
    score_train = knn_model.score(train_feature, train_label)
    print('模型训练准确率：', format(score_train))
    score_test = knn_model.score(test_feature, test_label)
    print('模型预测准确率：', format(score_test))
    # 精确率 召回率 调和均值F1
    pred_label = knn_model.predict(test_feature)
    report = classification_report(test_label, pred_label)
    print(report)
    # plot_confusion_matrix(knn_model, X=test_feature, y_true=test_label)  
    # plt.show()

    #* 记录程序运行时间，结束时间
    endtime = datetime.datetime.now()
    print("程序运行时间：", endtime - starttime)

#* 5个特征    
# ========================决策树============================
# 模型训练准确率： 1.0
# 模型预测准确率： 0.7266666666666667
#               precision    recall  f1-score   support

#          0.0       0.87      0.81      0.84        67
#          1.0       0.62      0.70      0.66        37
#          2.0       0.63      0.63      0.63        46

#     accuracy                           0.73       150
#    macro avg       0.71      0.71      0.71       150
# weighted avg       0.74      0.73      0.73       150

# ========================提升树============================
# 模型训练准确率： 1.0
# 模型预测准确率： 0.7533333333333333
#               precision    recall  f1-score   support

#          0.0       0.88      0.75      0.81        67
#          1.0       0.71      0.73      0.72        37
#          2.0       0.65      0.78      0.71        46

#     accuracy                           0.75       150
#    macro avg       0.75      0.75      0.75       150
# weighted avg       0.77      0.75      0.76       150

# ========================SVM============================
# 模型训练准确率： 0.7542857142857143
# 模型预测准确率： 0.78
#               precision    recall  f1-score   support

#          0.0       0.93      0.82      0.87        67
#          1.0       0.67      0.65      0.66        37
#          2.0       0.69      0.83      0.75        46

#     accuracy                           0.78       150
#    macro avg       0.76      0.77      0.76       150
# weighted avg       0.79      0.78      0.78       150

# ========================决策分析============================
# 模型训练准确率： 0.6942857142857143
# 模型预测准确率： 0.7866666666666666
#               precision    recall  f1-score   support

#          0.0       0.96      0.82      0.89        67
#          1.0       0.64      0.86      0.74        37
#          2.0       0.72      0.67      0.70        46

#     accuracy                           0.79       150
#    macro avg       0.78      0.79      0.77       150
# weighted avg       0.81      0.79      0.79       150

# ========================K近邻============================
# 模型训练准确率： 0.8457142857142858
# 模型预测准确率： 0.6733333333333333
#               precision    recall  f1-score   support

#          0.0       0.70      0.84      0.76        67
#          1.0       0.61      0.68      0.64        37
#          2.0       0.69      0.43      0.53        46

#     accuracy                           0.67       150
#    macro avg       0.67      0.65      0.65       150
# weighted avg       0.67      0.67      0.66       150
#* 11个特征    
# ========================决策树============================
# 模型训练准确率： 1.0
# 模型预测准确率： 0.6866666666666666
#               precision    recall  f1-score   support

#          0.0       0.78      0.73      0.75        67
#          1.0       0.63      0.70      0.67        37
#          2.0       0.61      0.61      0.61        46

#     accuracy                           0.69       150
#    macro avg       0.67      0.68      0.68       150
# weighted avg       0.69      0.69      0.69       150

# ========================提升树============================
# 模型训练准确率： 1.0
# 模型预测准确率： 0.76
#               precision    recall  f1-score   support

#          0.0       0.87      0.81      0.84        67
#          1.0       0.75      0.73      0.74        37
#          2.0       0.63      0.72      0.67        46

#     accuracy                           0.76       150
#    macro avg       0.75      0.75      0.75       150
# weighted avg       0.77      0.76      0.76       150

# ========================SVM============================
# 模型训练准确率： 0.7714285714285715
# 模型预测准确率： 0.78
#               precision    recall  f1-score   support

#          0.0       0.93      0.79      0.85        67
#          1.0       0.68      0.68      0.68        37
#          2.0       0.70      0.85      0.76        46

#     accuracy                           0.78       150
#    macro avg       0.77      0.77      0.77       150
# weighted avg       0.80      0.78      0.78       150

# ========================决策分析============================
# 模型训练准确率： 0.7571428571428571
# 模型预测准确率： 0.7666666666666667
#               precision    recall  f1-score   support

#          0.0       0.93      0.82      0.87        67
#          1.0       0.64      0.73      0.68        37
#          2.0       0.67      0.72      0.69        46

#     accuracy                           0.77       150
#    macro avg       0.75      0.76      0.75       150
# weighted avg       0.78      0.77      0.77       150

# ========================K近邻============================
# 模型训练准确率： 0.8714285714285714
# 模型预测准确率： 0.72
#               precision    recall  f1-score   support

#          0.0       0.76      0.84      0.79        67
#          1.0       0.67      0.76      0.71        37
#          2.0       0.71      0.52      0.60        46

#     accuracy                           0.72       150
#    macro avg       0.71      0.70      0.70       150
# weighted avg       0.72      0.72      0.71       150
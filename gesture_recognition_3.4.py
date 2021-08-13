# -*- encoding: utf-8 -*-
'''
@File    :   gesture_recognition_3.4.py
@Time    :   2021/07/18 19:09:12
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
# 按不同人划分训练集和测试集
# ------------------------------ file details ------------------------------ #

# 加载相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.utils import shuffle
import datetime
from preprocessing import mul_subcarries


# 定义数据集读取器
def load_data(filepath=None):
    # ! 读取数据文件
    # * 读取数据
    feature_number = 81 * 30
    # ! DX
    # 手势O，位置1
    filepath_O_1 = filepath + 'DX/O/gresture_O_location_1_'
    csi_DX_O_1 = mul_subcarries(filepath_O_1, feature_number, 0)
    # 手势X，位置1
    filepath_X_1 = filepath + 'DX/X/gresture_X_location_1_'
    csi_DX_X_1 = mul_subcarries(filepath_X_1, feature_number, 1)
    # 手势PO，位置1
    filepath_PO_1 = filepath + 'DX/PO/gresture_PO_location_1_'
    csi_DX_PO_1 = mul_subcarries(filepath_PO_1, feature_number, 2)
    # 整合
    csi_DX_1 = np.array((csi_DX_O_1, csi_DX_X_1, csi_DX_PO_1))
    csi_DX_1 = np.reshape(csi_DX_1, (-1, feature_number + 1))  # ! 注意修改
    print(datetime.datetime.now())
    # ! LJP
    # 手势O，位置1
    filepath_O_1 = filepath + 'LJP/O/gresture_O_location_1_'
    csi_LJP_O_1 = mul_subcarries(filepath_O_1, feature_number, 0)
    # 手势X，位置1
    filepath_X_1 = filepath + 'LJP/X/gresture_X_location_1_'
    csi_LJP_X_1 = mul_subcarries(filepath_X_1, feature_number, 1)
    # 手势PO，位置1
    filepath_PO_1 = filepath + 'LJP/PO/gresture_PO_location_1_'
    csi_LJP_PO_1 = mul_subcarries(filepath_PO_1, feature_number, 2)
    # 整合
    csi_LJP_1 = np.array((csi_LJP_O_1, csi_LJP_X_1, csi_LJP_PO_1))
    csi_LJP_1 = np.reshape(csi_LJP_1, (-1, feature_number + 1))
    print(datetime.datetime.now())
    # ! LZW
    # 手势O，位置1
    filepath_O_1 = filepath + 'LZW/O/gresture_O_location_1_'
    csi_LZW_O_1 = mul_subcarries(filepath_O_1, feature_number, 0)
    # 手势X，位置1
    filepath_X_1 = filepath + 'LZW/X/gresture_X_location_1_'
    csi_LZW_X_1 = mul_subcarries(filepath_X_1, feature_number, 1)
    # 手势PO，位置1
    filepath_PO_1 = filepath + 'LZW/PO/gresture_PO_location_1_'
    csi_LZW_PO_1 = mul_subcarries(filepath_PO_1, feature_number, 2)
    # 整合
    csi_LZW_1 = np.array((csi_LZW_O_1, csi_LZW_X_1, csi_LZW_PO_1))
    csi_LZW_1 = np.reshape(csi_LZW_1, (-1, feature_number + 1))
    print(datetime.datetime.now())
    # ! MYW
    # 手势O，位置1
    # ? 只有手势O
    filepath_O_1 = filepath + 'MYW/O/gresture_O_location_1_'
    csi_MYW_O_1 = mul_subcarries(filepath_O_1, feature_number, 0)
    # 整合
    csi_MYW_1 = np.array((csi_MYW_O_1))
    csi_MYW_1 = np.reshape(csi_MYW_1, (-1, feature_number + 1))
    print(datetime.datetime.now())
    # * 整合所有样本，乱序，分割
    # 整理数据集
    csi_1 = np.array((csi_LJP_1, csi_DX_1))
    csi_1 = np.reshape(csi_1, (-1, feature_number + 1))
    csi_1 = np.append(csi_1, csi_MYW_1, axis=0)
    csi_1 = np.reshape(csi_1, (-1, feature_number + 1))
    # 分割特征和标签
    train_feature, train_label = np.split(csi_1, (feature_number,), axis=1)
    test_feature, test_label = np.split(csi_LZW_1, (feature_number,), axis=1)
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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=5)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=1, padding=5)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是10
        self.fc1 = nn.Linear(in_features=2880, out_features=96)
        # 定义一层全连接层，输出维度是10
        self.fc2 = nn.Linear(in_features=96, out_features=3)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = x.view([x.shape[0], 2880])
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


if __name__ == '__main__':

    # 仅优化算法的设置有所差别
    model = CNN()
    model.train()
    params = list(model.parameters())

    BATCHSIZE = 15
    # 调用加载数据的函数
    train_feature, test_feature, train_label, test_label = load_data('/Users/yuxiao/CSI_data/classroom_data_unit/')
    train_loader = load_dataset(mode='train', train_feature=train_feature, train_label=train_label, BATCHSIZE=BATCHSIZE)
    # 设置不同初始学习率
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    EPOCH_NUM = 50
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

    # 保存模型参数
    PATH = 'model/gesture_recognition_3-4.pth'
    torch.save(model.state_dict(), PATH)

    model = CNN()
    model.load_state_dict(torch.load(PATH))
    print('test......')
    model.eval()
    test_loader = load_dataset(mode='test', test_feature=test_feature, test_label=test_label, BATCHSIZE=BATCHSIZE)

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
    # epoch: 49, loss = 0.551763728260994, acc = 0.9722222222222223
    # loss = 1.2158974289894104, acc = 0.33333333333333337

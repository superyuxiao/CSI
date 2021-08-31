# -*- coding: utf-8 -*-
# @Author   : YuXiao 于潇
# @Time     : 2021/8/26 4:39 下午
# @File     : compare_cross_location.py
# @Project  : CSI-Project
# @Contact  : superyuxiao@icloud.com
# @License  : (C)Copyright 2020-2021, Key Laboratory of University Wireless Communication
#                Beijing University of Posts and Telecommunications

# --------------------------- file details --------------------------- #
#
#
# --------------------------- file details --------------------------- #
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
import csv

if __name__ == '__main__':

    with open('CCFE-cross_location.csv', 'r') as f:
        reader = csv.reader(f)
        headers = next(f)
        print(type(reader))
        common = []
        ccfe = []
        for row in reader:
            common.append(row[5])
            ccfe.append(row[9])
        common = list(map(float, common[:20]))
        ccfe = list(map(float, ccfe[:20]))
        # print(common)
        # print(ccfe)
        x = [i for i in range(20)]
        plt.plot(x, common, 'ro-')
        plt.plot(x, ccfe, 'bo-')
        # plt.ylim((0.3, 1.0))
        plt.show()

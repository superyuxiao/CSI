# -*- coding: utf-8 -*-
# @Author   : YuXiao 于潇
# @Time     : 2021/8/26 11:44 上午
# @File     : ellipsoid.py
# @Project  : CSI-Project
# @Contact  : superyuxiao@icloud.com
# @License  : (C)Copyright 2020-2021, Key Laboratory of University Wireless Communication
#                Beijing University of Posts and Telecommunications

# --------------------------- file details --------------------------- #
# 绘制椭圆形的菲涅尔区
#
# --------------------------- file details --------------------------- #
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
e2 = Ellipse(xy=(0, 0), width=1.81 * 2, height=0.94 * 2, angle=0)
e1 = Ellipse(xy=(0, 0), width=1.91 * 2, height=1.04 * 2, angle=0)
ax.add_artist(e1)
ax.add_artist(e2)
e1.set_edgecolor("white")
e2.set_facecolor("black")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
ax.grid(True)
plt.title("50% Probablity Contour - Homework 4.2")
plt.show()

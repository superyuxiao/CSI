#from tokenize import Bracket
from matplotlib.pyplot import plot
import numpy as np
from numpy import random
from Bfee import Bfee
from get_scale_csi import get_scale_csi
from sklearn import model_selection
import matplotlib.pyplot as plt
import datetime


#* 记录程序运行时间，开始时间
starttime = datetime.datetime.now()
print(starttime)
# wendangming E:/CSI/CSI/classroom/LJP/SIT/wendangming.txt
wendangming_path = 'E:/CSI/CSI/classroom/MYW/UP/4/wendangming.txt'
wendangming = np.loadtxt(wendangming_path,dtype=int,delimiter='****')   
# 帧序号，手势序号，时间戳  
frame_index, gesture_index, timestamps= np.split(wendangming, (1,2), axis=1) #pylint: disable=unbalanced-tuple-unpacking
# print(frame_index)
print(datetime.datetime.now())
# CSI数据
bfee = Bfee.from_file("E:/CSI/CSI/classroom/MYW/UP/4/log.dat", model_name_encode="gb2312")
# print(len(bfee.dicts))
# print(len(bfee.all_csi))
# 数据分割保存
print(datetime.datetime.now())
for i in range(len(frame_index)-1):
    data = bfee.dicts[frame_index[i][0]:frame_index[i+1][0]]
    data_name ='E:/CSI/CSI/classroom_data_unit/MYW/UP/activity_UP_location_4_' + str(i)
    #print(data_name,data)
    np.save(data_name, data)
print(datetime.datetime.now()) 
#* 记录程序运行时间，结束时间
endtime = datetime.datetime.now()
print("程序运行时间：", endtime - starttime)

#* 记录程序运行时间，开始时间
starttime = datetime.datetime.now()
print(starttime)
# wendangming E:/CSI/CSI/classroom/LJP/SIT/wendangming.txt
wendangming_path = 'E:/CSI/CSI/classroom/MYW/UP/5/wendangming.txt'
wendangming = np.loadtxt(wendangming_path,dtype=int,delimiter='****')   
# 帧序号，手势序号，时间戳  
frame_index, gesture_index, timestamps= np.split(wendangming, (1,2), axis=1) #pylint: disable=unbalanced-tuple-unpacking
# print(frame_index)
print(datetime.datetime.now())
# CSI数据
bfee = Bfee.from_file("E:/CSI/CSI/classroom/MYW/UP/5/log.dat", model_name_encode="gb2312")
# print(len(bfee.dicts))
# print(len(bfee.all_csi))
# 数据分割保存
print(datetime.datetime.now())
for i in range(len(frame_index)-1):
    data = bfee.dicts[frame_index[i][0]:frame_index[i+1][0]]
    data_name ='E:/CSI/CSI/classroom_data_unit/MYW/UP/activity_UP_location_5_' + str(i)
    #print(data_name,data)
    np.save(data_name, data)
print(datetime.datetime.now())
#* 记录程序运行时间，结束时间
endtime = datetime.datetime.now()
print("程序运行时间：", endtime - starttime)


# #* 记录程序运行时间，开始时间
# starttime = datetime.datetime.now()
# print(starttime)
# # wendangming
# wendangming_path = 'CSI/classroom/MYW/O/1/wendangming.txt'
# wendangming = np.loadtxt(wendangming_path,dtype=int,delimiter='****')   
# # 帧序号，手势序号，时间戳  
# frame_index, gesture_index, timestamps= np.split(wendangming, (1,2), axis=1) #pylint: disable=unbalanced-tuple-unpacking
# print(frame_index)
# print(datetime.datetime.now())
# # CSI数据
# bfee = Bfee.from_file("CSI/classroom/MYW/O/1/log.dat", model_name_encode="gb2312")
# # print(len(bfee.dicts))
# # print(len(bfee.all_csi))
# # 数据分割保存
# print(datetime.datetime.now())
# for i in range(len(frame_index)-1):
#     data = bfee.dicts[frame_index[i][0]:frame_index[i+1][0]]
#     data_name ='CSI/classroom_data_unit/MYW/O/gresture_O_location_1_' + str(i)
#     #print(data_name,data)
#     np.save(data_name, data)
# print(datetime.datetime.now())
# #* 记录程序运行时间，结束时间
# endtime = datetime.datetime.now()
# print("程序运行时间：", endtime - starttime)

# #* 记录程序运行时间，开始时间
# starttime = datetime.datetime.now()
# print(starttime)
# # wendangming
# wendangming_path = 'CSI/classroom/MYW/PO/1/wendangming.txt'
# wendangming = np.loadtxt(wendangming_path,dtype=int,delimiter='****')   
# # 帧序号，手势序号，时间戳  
# frame_index, gesture_index, timestamps= np.split(wendangming, (1,2), axis=1) #pylint: disable=unbalanced-tuple-unpacking
# # print(frame_index)
# print(datetime.datetime.now())
# # CSI数据
# bfee = Bfee.from_file("CSI/classroom/MYW/PO/1/log.dat", model_name_encode="gb2312")
# # print(len(bfee.dicts))
# # print(len(bfee.all_csi))
# # 数据分割保存
# print(datetime.datetime.now())
# for i in range(len(frame_index)-1):
#     data = bfee.dicts[frame_index[i][0]:frame_index[i+1][0]]
#     data_name ='CSI/classroom_data_unit/MYW/PO/gresture_PO_location_1_' + str(i)
#     #print(data_name,data)
#     np.save(data_name, data)
# print(datetime.datetime.now())
# #* 记录程序运行时间，结束时间
# endtime = datetime.datetime.now()
# print("程序运行时间：", endtime - starttime)

# #* 记录程序运行时间，开始时间
# starttime = datetime.datetime.now()
# print(starttime)
# # wendangming
# wendangming_path = 'classroom/MYW/X/1/wendangming.txt'
# wendangming = np.loadtxt(wendangming_path,dtype=int,delimiter='****')   
# # 帧序号，手势序号，时间戳  
# frame_index, gesture_index, timestamps= np.split(wendangming, (1,2), axis=1) #pylint: disable=unbalanced-tuple-unpacking
# # print(frame_index)
# print(datetime.datetime.now())
# # CSI数据
# bfee = Bfee.from_file("classroom/MYW/X/1/log.dat", model_name_encode="gb2312")
# # print(len(bfee.dicts))
# # print(len(bfee.all_csi))
# # 数据分割保存
# print(datetime.datetime.now())
# for i in range(len(frame_index)-1):
#     data = bfee.dicts[frame_index[i][0]:frame_index[i+1][0]]
#     data_name ='classroom_data_unit/MYW/X/gresture_X_location_1_' + str(i)
#     #print(data_name,data)
#     np.save(data_name, data)
# print(datetime.datetime.now())
# #* 记录程序运行时间，结束时间
# endtime = datetime.datetime.now()
# print("程序运行时间：", endtime - starttime)

# #* 记录程序运行时间，开始时间
# starttime = datetime.datetime.now()
# print(starttime)
# # wendangming
# wendangming_path = 'classroom/MYW/O/1/wendangming.txt'
# wendangming = np.loadtxt(wendangming_path,dtype=int,delimiter='****')   
# # 帧序号，手势序号，时间戳  
# frame_index, gesture_index, timestamps= np.split(wendangming, (1,2), axis=1) #pylint: disable=unbalanced-tuple-unpacking
# # print(frame_index)
# print(datetime.datetime.now())
# # CSI数据
# bfee = Bfee.from_file("classroom/MYW/O/1/log.dat", model_name_encode="gb2312")
# # print(len(bfee.dicts))
# # print(len(bfee.all_csi))
# # 数据分割保存
# print(datetime.datetime.now())
# for i in range(len(frame_index)-1):
#     data = bfee.dicts[frame_index[i][0]:frame_index[i+1][0]]
#     data_name ='classroom_data_unit/MYW/O/gresture_O_location_1_' + str(i)
#     #print(data_name,data)
#     np.save(data_name, data)
# print(datetime.datetime.now())
# #* 记录程序运行时间，结束时间
# endtime = datetime.datetime.now()
# print("程序运行时间：", endtime - starttime)

# #* 记录程序运行时间，开始时间
# starttime = datetime.datetime.now()
# print(starttime)
# # wendangming
# wendangming_path = 'classroom/MYW/PO/1/wendangming.txt'
# wendangming = np.loadtxt(wendangming_path,dtype=int,delimiter='****')   
# # 帧序号，手势序号，时间戳  
# frame_index, gesture_index, timestamps= np.split(wendangming, (1,2), axis=1) #pylint: disable=unbalanced-tuple-unpacking
# # print(frame_index)
# print(datetime.datetime.now())
# # CSI数据
# bfee = Bfee.from_file("classroom/MYW/PO/1/log.dat", model_name_encode="gb2312")
# # print(len(bfee.dicts))
# # print(len(bfee.all_csi))
# # 数据分割保存
# print(datetime.datetime.now())
# for i in range(len(frame_index)-1):
#     data = bfee.dicts[frame_index[i][0]:frame_index[i+1][0]]
#     data_name ='classroom_data_unit/MYW/PO/gresture_PO_location_1_' + str(i)
#     #print(data_name,data)
#     np.save(data_name, data)
# print(datetime.datetime.now())
# #* 记录程序运行时间，结束时间
# endtime = datetime.datetime.now()
# print("程序运行时间：", endtime - starttime)

gresture_O_location_1_6 = np.load('CSI/classroom_data_unit/MYW/X/gresture_X_location_1_6.npy',allow_pickle=True)
t = np.arange(0,len(gresture_O_location_1_6))
csi = np.empty((len(gresture_O_location_1_6),30,3,3), dtype = complex)
for i in range(len(gresture_O_location_1_6)):
    csi[i] = get_scale_csi(gresture_O_location_1_6[i])
    #print(csi)
    #print(csi.shape)
for i in range(25,30):
    subcarrier = csi[:,i,0,0]
    #plt.scatter(t, abs(subcarrier), c = abs(subcarrier))
    #plt.scatter(t, np.arctan(subcarrier.imag/subcarrier.real)/1.5707963, c = np.arctan(subcarrier.imag/subcarrier.real)/1.5707963)
    plt.plot(t, abs(subcarrier)) # 幅度
    #plt.plot(np.arctan(subcarrier.imag/subcarrier.real)/1.5707963) # 相位
    #for i in range(30):
    #subcarrier = csi[:,i,0]
    #plt.scatter(t, i*np.ones(len(t)), c= np.arctan(subcarrier.imag/subcarrier.real), cmap='rainbow', marker='s')
#plt.colorbar() 
plt.show()
    
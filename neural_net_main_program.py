# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:23:16 2018

@author: 泳浩
"""

import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import forward_problem as forward
import smooth_well
# import segyio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy


#采样间隔 2ms
#读取测井数据
data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line1_05.csv"))
del data["Unnamed: 0"]

# data1 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line1_045.csv"))
# del data1["Unnamed: 0"]

# data2 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line1_04.csv"))
# del data2["Unnamed: 0"]

# data3 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line1_035.csv"))
# del data3["Unnamed: 0"]

# data4 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line1_03.csv"))
# del data4["Unnamed: 0"]

# data5 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line1_025.csv"))
# del data5["Unnamed: 0"]

# data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line2_164.csv"))
# del data["Unnamed: 0"]

# data1 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line2_115.csv"))
# del data1["Unnamed: 0"]

# data2 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line2_065.csv"))
# del data2["Unnamed: 0"]

# data3 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line2_047.csv"))
# del data3["Unnamed: 0"]

# data4 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line2_038.csv"))
# del data4["Unnamed: 0"]

# data5 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\geo4_line2_021.csv"))
# del data5["Unnamed: 0"]

# # 平滑法1: 为savgol在一次式时的特殊表达 (moving_avg)
# moving_avg = data.rolling(15).mean()    
# time = np.array(data.iloc[:,[4]]).reshape(1, -1)
# y = range(len(time[0]))
# data_smooth = signal.savgol_filter(time, 4095, 1).reshape(-1, 1)
# plt.figure()
# plt.plot(y, data_smooth)
# plt.show()

#平滑法2: Lowess
backup = copy.copy(data)
smdata = smooth_well.Smooth(data)
data = smdata.sm()
# smdata.plot_map()

#地震子波 （不同角度）
Fmax, Dt = 30, np.arange(-50, 50, 1)   #延续时间 200ms　采样点100个
ricker = (1-2*(np.pi*Fmax*Dt*0.001)**2)*np.exp(-(np.pi*Fmax*Dt*0.001)**2)
theta = np.arange(0, 60, 1)*np.pi/180  #弧度
theta = np.linspace(0, 60, 60)*np.pi/180  #弧度
theta = theta + np.zeros((data.shape[0], len(theta)))
#theta = 90 * np.ones((data.shape[0], 1)).flatten()

# # #划定研究时窗　单位：采样点 （时窗对应相速度时窗）
# time_windows = 1606
# # Thomsen 各向异性相速度 Vp Vsv Vsh
# theta_2pi = np.arange(0, 360, 1)*np.pi/180  #弧度
# weak_elastic = forward.Weak_Anisotropy(data["Vp"], data["Vs"], data["Rho"], theta_2pi, data["Epsilon"], data["Delta"], data["Gamma"])
# Vp = weak_elastic.weakVp_phase()
# Vsv = weak_elastic.weakVsv_phase()
# Vsh = weak_elastic.weakVsh_phase()
# Vp[np.isnan(Vp)] = 0
# Vsv[np.isnan(Vsv)] = 0
# Vsh[np.isnan(Vsh)] = 0
# Vp, Vsv, Vsh = Vp[time_windows], Vsv[time_windows], Vsh[time_windows]
# fig = plt.figure()
# ax = fig.add_subplot(111, polar=True)
# ax.plot(theta_2pi, Vp, color="red", label="$Vp$")
# ax.plot(theta_2pi, Vsv, color="blue", label="$Vsv$")
# ax.plot(theta_2pi, Vsh, color="orange", label="$Vsh$")
# ax.set_title("$Clay ontent:0.5$")
# plt.legend()
# plt.show()

# # #划定研究时窗　单位：采样点 （时窗对应相速度时窗）
# time_windows = 1606
# # Darey_Hron 各向异性相速度 Vp Vsv Vsh
# theta_2pi = np.arange(0, 360, 1)*np.pi/180  #弧度
# weak_elastic = forward.Darey_Hron_trans(data["Vp"], data["Vs"], data["Rho"], theta_2pi, data["Epsilon"], data["Delta"], data["Gamma"])
# Vp = weak_elastic.DH_Vp_phase()
# Vsv = weak_elastic.DH_Vsv_phase()
# Vsh = weak_elastic.DH_Vsh_phase()
# Vp[np.isnan(Vp)] = 0
# Vsv[np.isnan(Vsv)] = 0
# Vsh[np.isnan(Vsh)] = 0
# Vp, Vsv, Vsh = Vp[time_windows], Vsv[time_windows], Vsh[time_windows]
# fig = plt.figure()
# ax = fig.add_subplot(111, polar=True)
# ax.plot(theta_2pi, Vp, color="red", label="$Vp$")
# ax.plot(theta_2pi, Vsv, color="blue", label="$Vsv$")
# ax.plot(theta_2pi, Vsh, color="orange", label="$Vsh$")
# ax.set_title("$Clay ontent(accurate):0.5$")
# plt.legend()
# plt.show()

# # 划定研究时窗　单位：采样点 （时窗对应反射系数时窗）
# time_windows = 1605

# #反射系数
# #1 wiggen 三项式 （基于aki_richard）
# # wiggen = forward.Aki_Richard(data["Vp"], data["Vs"], data["Rho"], theta)
# # reflect_1 = wiggen.wiggens()
# # reflect_1[np.isnan(reflect_1)] = 0
# #2 aki_richards
aki = forward.Aki_Richard(data["Vp"], data["Vs"], data["Rho"], theta)
reflect_2 = aki.aki_richards()
reflect_2[np.isnan(reflect_2)] = 0
aki = forward.Aki_Richard(backup["Vp"], backup["Vs"], backup["Rho"], theta)
reflect_b = aki.aki_richards()
reflect_b[np.isnan(reflect_b)] = 0
# #3 thomsen　横向各向同性　
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# reflect_3 = vti_appr.ruger_approximate()
# reflect_3[np.isnan(reflect_3)] = 0
# #4 thomsen 相速度反射系数
# reflect_4 = forward.Normal(Vsv, data["Rho"])
# reflect_4 =reflect_4.reflect_4ion_p()
# reflect_4[np.isnan(reflect_4)] = 0
# #5 Zoeppritz 精确解表达式　某一时窗
# zoeppritz = forward.Zoeppritz(data["Vp"], data["Vs"], data["Rho"], theta)
# reflect_5 = zoeppritz.zoeppritz_exactly(time_windows)
# reflect_5[np.isnan(reflect_5)] = 0
# #6 Zoeppritz 精确解表达式　全部时窗　
# #zoeppritz = forward.Zoeppritz(data["Vp"], data["Vs"], data["Rho"], theta)
# #reflect_6 = zoeppritz.zoeppritz_exactly_all()
# #reflect_6[np.isnan(reflect_6)] = 0
#7 thomsen 横向各向同性
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# reflect_7 = vti_appr.iso_approximate()
# reflect_7[np.isnan(reflect_7)] = 0

# # 不同泥质含量参数条件下岩石的各向同性与各向异性反射系数计算（包括绘图）
# #各向同性
# aki = forward.Aki_Richard(data["Vp"], data["Vs"], data["Rho"], theta)
# reflect = aki.aki_richards()
# reflect[np.isnan(reflect)] = 0

# aki = forward.Aki_Richard(data1["Vp"], data1["Vs"], data1["Rho"], theta)
# reflect_1 = aki.aki_richards()
# reflect_1[np.isnan(reflect_1)] = 0

# aki = forward.Aki_Richard(data2["Vp"], data2["Vs"], data2["Rho"], theta)
# reflect_2 = aki.aki_richards()
# reflect_2[np.isnan(reflect_2)] = 0

# aki = forward.Aki_Richard(data3["Vp"], data3["Vs"], data3["Rho"], theta)
# reflect_3 = aki.aki_richards()
# reflect_3[np.isnan(reflect_3)] = 0

# aki = forward.Aki_Richard(data4["Vp"], data4["Vs"], data4["Rho"], theta)
# reflect_4 = aki.aki_richards()
# reflect_4[np.isnan(reflect_4)] = 0

# aki = forward.Aki_Richard(data5["Vp"], data5["Vs"], data5["Rho"], theta)
# reflect_5 = aki.aki_richards()
# reflect_5[np.isnan(reflect_5)] = 0
# #各向异性
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# reflecta = vti_appr.ruger_approximate()
# reflecta[np.isnan(reflecta)] = 0

# vti_appr = forward.Thomsen(data1["Vp"], data1["Vs"], data1["Rho"], theta, data1["Epsilon"], data1["Delta"], data1["Gamma"])
# reflect_1a = vti_appr.ruger_approximate()
# reflect_1a[np.isnan(reflect_1a)] = 0

# vti_appr = forward.Thomsen(data2["Vp"], data2["Vs"], data2["Rho"], theta, data2["Epsilon"], data2["Delta"], data2["Gamma"])
# reflect_2a = vti_appr.ruger_approximate()
# reflect_2a[np.isnan(reflect_2a)] = 0

# vti_appr = forward.Thomsen(data3["Vp"], data3["Vs"], data3["Rho"], theta, data3["Epsilon"], data3["Delta"], data3["Gamma"])
# reflect_3a = vti_appr.ruger_approximate()
# reflect_3a[np.isnan(reflect_3a)] = 0

# vti_appr = forward.Thomsen(data4["Vp"], data4["Vs"], data4["Rho"], theta, data4["Epsilon"], data4["Delta"], data4["Gamma"])
# reflect_4a = vti_appr.ruger_approximate()
# reflect_4a[np.isnan(reflect_4a)] = 0

# vti_appr = forward.Thomsen(data5["Vp"], data5["Vs"], data5["Rho"], theta, data5["Epsilon"], data5["Delta"], data5["Gamma"])
# reflect_5a = vti_appr.ruger_approximate()
# reflect_5a[np.isnan(reflect_5a)] = 0

# fig = plt.figure()
# ax = plt.subplot()
# ax.plot(theta[0]*180/np.pi, reflect[time_windows], color="#8c564b", label="iso_clay050")
# ax.plot(theta[0]*180/np.pi, reflect_1[time_windows], color="#1f77b4", label="iso_clay045")
# ax.plot(theta[0]*180/np.pi, reflect_2[time_windows], color="#ff7f0e", label="iso_clay040")
# ax.plot(theta[0]*180/np.pi, reflect_3[time_windows], color="#2ca02c", label="iso_clay035")
# ax.plot(theta[0]*180/np.pi, reflect_4[time_windows], color="#d62728", label="iso_clay030")
# ax.plot(theta[0]*180/np.pi, reflect_5[time_windows], color="#9467bd", label="iso_clay025")
# ax.plot(theta[0]*180/np.pi, reflecta[time_windows], color="#8c564b", linestyle="--", label="ani_clay050")
# ax.plot(theta[0]*180/np.pi, reflect_1a[time_windows], color="#1f77b4", linestyle="--", label="ani_clay045")
# ax.plot(theta[0]*180/np.pi, reflect_2a[time_windows], color="#ff7f0e", linestyle="--", label="ani_clay040")
# ax.plot(theta[0]*180/np.pi, reflect_3a[time_windows], color="#2ca02c", linestyle="--", label="ani_clay035")
# ax.plot(theta[0]*180/np.pi, reflect_4a[time_windows], color="#d62728", linestyle="--", label="ani_clay030")
# ax.plot(theta[0]*180/np.pi, reflect_5a[time_windows], color="#9467bd", linestyle="--", label="ani_clay025")
# plt.grid()
# plt.legend()
# plt.show()

# # 不同孔隙度参数条件下岩石的各向同性与各向异性反射系数计算（包括绘图）
# #各向同性
# aki = forward.Aki_Richard(data["Vp"], data["Vs"], data["Rho"], theta)
# reflect = aki.aki_richards()
# reflect[np.isnan(reflect)] = 0

# aki = forward.Aki_Richard(data1["Vp"], data1["Vs"], data1["Rho"], theta)
# reflect_1 = aki.aki_richards()
# reflect_1[np.isnan(reflect_1)] = 0

# aki = forward.Aki_Richard(data2["Vp"], data2["Vs"], data2["Rho"], theta)
# reflect_2 = aki.aki_richards()
# reflect_2[np.isnan(reflect_2)] = 0

# aki = forward.Aki_Richard(data3["Vp"], data3["Vs"], data3["Rho"], theta)
# reflect_3 = aki.aki_richards()
# reflect_3[np.isnan(reflect_3)] = 0

# aki = forward.Aki_Richard(data4["Vp"], data4["Vs"], data4["Rho"], theta)
# reflect_4 = aki.aki_richards()
# reflect_4[np.isnan(reflect_4)] = 0

# aki = forward.Aki_Richard(data5["Vp"], data5["Vs"], data5["Rho"], theta)
# reflect_5 = aki.aki_richards()
# reflect_5[np.isnan(reflect_5)] = 0
# #各向异性
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# reflecta = vti_appr.ruger_approximate()
# reflecta[np.isnan(reflecta)] = 0

# vti_appr = forward.Thomsen(data1["Vp"], data1["Vs"], data1["Rho"], theta, data1["Epsilon"], data1["Delta"], data1["Gamma"])
# reflect_1a = vti_appr.ruger_approximate()
# reflect_1a[np.isnan(reflect_1a)] = 0

# vti_appr = forward.Thomsen(data2["Vp"], data2["Vs"], data2["Rho"], theta, data2["Epsilon"], data2["Delta"], data2["Gamma"])
# reflect_2a = vti_appr.ruger_approximate()
# reflect_2a[np.isnan(reflect_2a)] = 0

# vti_appr = forward.Thomsen(data3["Vp"], data3["Vs"], data3["Rho"], theta, data3["Epsilon"], data3["Delta"], data3["Gamma"])
# reflect_3a = vti_appr.ruger_approximate()
# reflect_3a[np.isnan(reflect_3a)] = 0

# vti_appr = forward.Thomsen(data4["Vp"], data4["Vs"], data4["Rho"], theta, data4["Epsilon"], data4["Delta"], data4["Gamma"])
# reflect_4a = vti_appr.ruger_approximate()
# reflect_4a[np.isnan(reflect_4a)] = 0

# vti_appr = forward.Thomsen(data5["Vp"], data5["Vs"], data5["Rho"], theta, data5["Epsilon"], data5["Delta"], data5["Gamma"])
# reflect_5a = vti_appr.ruger_approximate()
# reflect_5a[np.isnan(reflect_5a)] = 0

# fig = plt.figure()
# ax = plt.subplot()
# ax.plot(theta[0]*180/np.pi, reflect[time_windows], color="#8c564b", label="iso_porosity164")
# ax.plot(theta[0]*180/np.pi, reflect_1[time_windows], color="#1f77b4", label="iso_porosity115")
# ax.plot(theta[0]*180/np.pi, reflect_2[time_windows], color="#ff7f0e", label="iso_porosity065")
# ax.plot(theta[0]*180/np.pi, reflect_3[time_windows], color="#2ca02c", label="iso_porosity047")
# ax.plot(theta[0]*180/np.pi, reflect_4[time_windows], color="#d62728", label="iso_porosity038")
# ax.plot(theta[0]*180/np.pi, reflect_5[time_windows], color="#9467bd", label="iso_porosity021")
# ax.plot(theta[0]*180/np.pi, reflecta[time_windows], color="#8c564b", linestyle="--", label="ani_porosity164")
# ax.plot(theta[0]*180/np.pi, reflect_1a[time_windows], color="#1f77b4", linestyle="--", label="ani_porosity115")
# ax.plot(theta[0]*180/np.pi, reflect_2a[time_windows], color="#ff7f0e", linestyle="--", label="ani_porosity065")
# ax.plot(theta[0]*180/np.pi, reflect_3a[time_windows], color="#2ca02c", linestyle="--", label="ani_porosity047")
# ax.plot(theta[0]*180/np.pi, reflect_4a[time_windows], color="#d62728", linestyle="--", label="ani_porosity038")
# ax.plot(theta[0]*180/np.pi, reflect_5a[time_windows], color="#9467bd", linestyle="--", label="ani_porosity021")
# plt.grid()
# plt.legend()
# plt.show()

# # 子波 Toeplitz矩阵
# temp = int(ricker.shape[0]/2)
# row = np.zeros((1, data.shape[0])).flatten()
# column = np.zeros((1, data.shape[0]+temp)).flatten()
# column[0:ricker.shape[0]] = ricker
# toeplitz = (linalg.toeplitz(row, column)[:, temp:])[:-1, :-1]

#合成地震记录 （不同角度）
def synthetic(reflect):
	# time = np.arange(reflect.shape[0], dtype=np.float64)
	Syn = np.zeros((reflect.shape[0], reflect.shape[1]))
	for i in range(Syn.shape[1]):
	    Syn[:, i] = np.convolve(reflect[:, i], ricker, "same").flatten()
	return Syn
Syn = synthetic(reflect_2)
Syn_b = synthetic(reflect_b)
#绘制一道的合成地震记录 
#fig = plt.figure() 
#ax = plt.subplot()
#ax.plot(time, Syn[:, 1], color="black")
#ax.fill_between(time, Syn[:, 1], 0, where=Syn[:, 1]>0, facecolor="black")
#plt.show()

# #反演绘制Rp Rs Rr
# R_three = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# Rp, Rs, Rr = R_three.expectation()
# Rs[np.isnan(Rs)] = 0
# fig = plt.figure()
# ax = plt.subplot(311)
# ax.plot(Rp.T)
# ax2 = plt.subplot(312)
# ax2.plot(Rs.T)
# ax3 = plt.subplot(313)
# ax3.plot(Rr.T)
# plt.show()

# #绘制反射系数随角度变化　
# fig = plt.figure()
# ax = plt.subplot()
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Reflection Coefficient$")
# ax.plot(theta[0]*180/np.pi, reflect_2[time_windows], color="blue", label="Aki-Richards")
# ax.plot(theta[0]*180/np.pi, reflect_3[time_windows], color="red", label="Ruger")
# ax.plot(theta[0]*180/np.pi, reflect_7[time_windows], color="purple", label="Thomsen_vti")
# #ax.plot(theta[0]*180/np.pi, reflect_1[time_windows], color="yellow", label="Wiggen")
# ax.plot(theta[0]*180/np.pi, reflect_5[0], color="orange", label="Zoeppritz")
# #ax.plot(theta[0]*180/np.pi, reflect_1[0], color="black", alpha=0.2)
# plt.grid()
# plt.legend()
# plt.show()

# #反射系数随角度变化（残差）
# fig = plt.figure()
# ax = plt.subplot()
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Residual Error$")
# ax.plot(theta[0]*180/np.pi, reflect_2[time_windows]-reflect_5[0], color="blue", label="Aki-Richards")
# ax.plot(theta[0]*180/np.pi, reflect_3[time_windows]-reflect_5[0], color="red", label="Ruger")
# ax.plot(theta[0]*180/np.pi, reflect_7[time_windows]-reflect_5[0], color="purple", label="Thomsen_vti")
# #ax.plot(theta[0]*180/np.pi, reflect_1[time_windows]-reflect_5[0], color="yellow", label="Wiggen")
# #ax.plot(theta[0]*180/np.pi, reflect_5[0], color="orange", label="Zoeppritz")
# #ax.plot(theta[0]*180/np.pi, reflect_1[0], color="black", alpha=0.2)
# plt.grid()
# plt.legend()
# plt.show()

# ## segyio 输出人工合成地震数据 
# fig = plt.figure()
# ax = plt.subplot()
# plt.imshow(Syn[1500:2000, :])
# plt.show()

## segyio 输出人工合成地震数据 
#spec = segyio.spec()
#filename = 'vti.sgy'
#
#spec.sorting = 1
#spec.format = 1
#spec.samples = np.arange(len(time))
#spec.ilines = np.arange(Syn.shape[1])
#spec.xlines = np.arange(1)
#
#with segyio.create(filename, spec) as f:
#
#    # write the line itself to the file and the inline number in all this line's headers
#    for ilno in spec.ilines:
#        f.iline[ilno] = np.zeros((len(spec.xlines), len(spec.samples)), dtype=np.float64) + Syn[:, ilno]
#        f.header.iline[ilno] = {
#            segyio.TraceField.INLINE_3D: ilno,
#       
#        }
#
#    # then do the same for xlines
#    for xlno in spec.xlines:
#        f.header.xline[xlno] = {
#            segyio.TraceField.CROSSLINE_3D: xlno,
#
#           
#        }
##
        

# 神经网络搭建

input = torch.FloatTensor((Syn[1500:1560, :]).reshape(1,1,60,60))
target = torch.FloatTensor(np.array(data["Vp"][1500:1560])).view(1, -1)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 8, (25, 3), stride=(1, 1), padding=(12, 1))
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 8192)
        self.fc3 = nn.Linear(8192, 8192)
        self.fc4 = nn.Linear(8192, 8192)
        self.fc5 = nn.Linear(8192, 4096)
        self.fc6 = nn.Linear(4096, 1024)
        self.fc7 = nn.Linear(1024, 256)
        self.fc8 = nn.Linear(256, 60)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Net
net = Net()
# create your optimizer
optimizer = optim.Adam(net.parameters())
# criterion
criterion = nn.MSELoss()

epoch_value = []
loss_value = []
plt.ion()
plt.figure()
for epoch in range(100):

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update

    epoch_value.append(epoch)
    loss_value.append(loss.item())

    if epoch > 1:
        plt.cla()
        plt.plot(epoch_value, loss_value)
		# plt.plot(epoch_value, loss_value2, color="red")
        plt.pause(0.001)
# torch.save(net.state_dict(), "net.pkl")
plt.ioff()
plt.show()
output = torch.FloatTensor(np.array(backup["Vp"][1140:1200])).view(1, -1)
after = net(input = torch.FloatTensor((Syn_b[1140:1200, :]).reshape(1,1,60,60)))
# plt.figure()
# plt.plot(after, color="red")
# plt.plot(backup)
# plt.show()
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
import copy
import segyio
from scipy.optimize import curve_fit

############################################################################
#采样间隔 2ms
# # #读取测井数据
# data0 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_05.csv"))
# del data0["Unnamed: 0"]

# data1 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_045.csv"))
# del data1["Unnamed: 0"]

# data2 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_04.csv"))
# del data2["Unnamed: 0"]

data3 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_035.csv"))
del data3["Unnamed: 0"]

# data4 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_03.csv"))
# del data4["Unnamed: 0"]

# data5 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_025.csv"))
# del data5["Unnamed: 0"]

# data0 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line2_164.csv"))
# del data0["Unnamed: 0"]

# data1 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line2_115.csv"))
# del data1["Unnamed: 0"]

# data2 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line2_065.csv"))
# del data2["Unnamed: 0"]

# data3 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line2_047.csv"))
# del data3["Unnamed: 0"]

# data4 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line2_038.csv"))
# del data4["Unnamed: 0"]

# data5 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line2_021.csv"))
# del data5["Unnamed: 0"]

# Model analysing...
# data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\Analysis_of_AVO.csv"))
# del data["Unnamed: 0"]

# temp
data = copy.copy(data3)

############################################################################
# 地震子波人工理想的雷克子波 （不同角度）
theta_max, theta_nums, Fmax, t_samples, dt_space = 40, 40, 30, np.arange(-50, 50, 1), 0.002 #采样间隔2ms,采样点数100
# ricker = (1-2*(np.pi*Fmax*t_samples*dt_space)**2)*np.exp(-(np.pi*Fmax*t_samples*dt_space)**2)
# wavelet = ricker # 此处使用ricker子波

############################################################################
# 选取角度范围
theta = np.linspace(1, theta_max, theta_nums)*np.pi/180  #弧度
theta = theta + np.zeros((data.shape[0], theta_nums))

############################################################################
# L1
# wavelet0 = np.loadtxt("E:\\PROJECT\\L1_pack\\wave0.txt", skiprows=42) # 此处使用提取的地震子波
# wavelet1 = np.loadtxt("E:\\PROJECT\\L1_pack\\wave1.txt", skiprows=42) # 此处使用提取的地震子波
# wavelet2 = np.loadtxt("E:\\PROJECT\\L1_pack\\wave2.txt", skiprows=42) # 此处使用提取的地震子波
# wavelet3 = np.loadtxt("E:\\PROJECT\\L1_pack\\wave3.txt", skiprows=42) # 此处使用提取的地震子波
# wavelet4 = np.loadtxt("E:\\PROJECT\\L1_pack\\wave4.txt", skiprows=42) # 此处使用提取的地震子波
wavelet0 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_01_08.txt", skiprows=56)
wavelet1 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_08_15.txt", skiprows=56)
wavelet2 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_15_22.txt", skiprows=56)
wavelet3 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_22_29.txt", skiprows=56)
wavelet4 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_29_36.txt", skiprows=56)

# # L2
# wavelet0 = np.loadtxt("E:\\PROJECT\\L2_pack\\wavelet_use_wells10.txt", skiprows=138) # 此处使用提取的地震子波
# wavelet1 = np.loadtxt("E:\\PROJECT\\L2_pack\\wavelet_use_wells20.txt", skiprows=138) # 此处使用提取的地震子波
# wavelet2 = np.loadtxt("E:\\PROJECT\\L2_pack\\wavelet_use_wells30.txt", skiprows=138) # 此处使用提取的地震子波
# wavelet3 = np.loadtxt("E:\\PROJECT\\L2_pack\\wavelet_use_wells40.txt", skiprows=138) # 此处使用提取的地震子波
# wavelet4 = np.loadtxt("E:\\PROJECT\\L2_pack\\wavelet_use_wells45.txt", skiprows=53) # 此处使用提取的地震子波

############################################################################
# 8 Zhang_3 横向各向同性
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# _, _, _, _, _, _, reflect_8 = vti_appr.ani_zhang()
# reflect_8[np.isnan(reflect_8)] = 0
# 6 Zoeppritz 精确解表达式　全部时窗　
# zoeppritz = forward.Zoeppritz(data["Vp"], data["Vs"], data["Rho"], theta)
# reflect_6 = zoeppritz.zoeppritz_exact_all()
# reflect_6[np.isnan(reflect_6)] = 0
# reflect_iso = reflect_6[0]
# 10 Graebner 精确解表达式　全部时窗
vti_exact_reflct = forward.Graebner(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
reflect_11 = vti_exact_reflct.vti_exact_all()
reflect_11[np.isnan(reflect_11)] = 0
reflect_ani = reflect_11[0]
############################################################################
# def convmtx(s, n):
#     """
#     Toeplitz方阵建立
#     s为序列，n为长和宽
#     """
#     t = int(s.shape[0]/2)
#     col_1 = np.r_[s[0], np.zeros(n-1+t)]
#     row_1 = np.r_[s, np.zeros(n-len(s)+t)]
#     temp = linalg.toeplitz(col_1, row_1)
#     return temp[:-t, t:]

# # 传入不同角度子波数据
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_02_10.txt", skiprows=56)

# # 反演单个参数的图像背景大小
# background = np.zeros((real_0.shape)) 

# # 子波矩阵不同角度
# W0 = convmtx(wave_0, background.shape[0])

############################################################################
#　合成地震记录 （不同角度）
def synthetic(reflect, wavelet):
	Syn = np.zeros((reflect.shape[0], reflect.shape[1]))
	for i in range(Syn.shape[1]):
	    Syn[:, i] = np.convolve(reflect[:, i], wavelet, "same").flatten()
	return Syn
Syn0 = synthetic(reflect_ani, wavelet0)
# Syn0 = synthetic(reflect_iso, wavelet0)
# Syn1 = synthetic(reflect_8, wavelet1)
# Syn2 = synthetic(reflect_8, wavelet2)
# Syn3 = synthetic(reflect_8, wavelet3)
# Syn4 = synthetic(reflect_8, wavelet4)
# Syn = np.empty_like(Syn0)
# Syn[:, 0:10] = Syn0[:, 0:10]
# Syn[:, 10:20] = Syn1[:, 10:20]
# Syn[:, 20:30] = Syn2[:, 20:30]
# Syn[:, 30:40] = Syn3[:, 30:40]
# Syn[:, 40:50] = Syn4[:, 40:50]
Syn = Syn0
############################################################################
# 绘制振幅随角度变化（真实地震数据）
top, bottom, cdp_beg, cdp_end, cdp_select = 1500, 1885, 0, 2191, 1200 # 500/730/980/1200/1460/1700//0-2191
time_windows = 1606
theta_select_range = 40
shift_samples = int(len(t_samples)/2)
real_seismic = segyio.tools.cube("F:\\Physic_Model_Data\\1.prj\\seismic.dir\\angle_mute.sgy")[0,:,:,:].T
# real_seismic = segyio.tools.cube("E:\\PROJECT\\2.sgy")[0,:,:,:].T
# real_seismic = segyio.tools.cube("F:\\Physic_Model_Data\\L1.prj\\seismic.dir\\angle_gather.sgy")[0,:,:,:].T
real = real_seismic[top:bottom, :, cdp_beg:cdp_end]
real = real/abs(real[time_windows-top-shift_samples:time_windows-top+shift_samples, 0:theta_select_range, cdp_select]).max() # range cdp
# real = real/abs(real).max() # all cdp
real_min_amp_value = (real[time_windows-top-shift_samples:time_windows-top+shift_samples, 0:theta_select_range, :]).min()
real_max_amp_value = (real[time_windows-top-shift_samples:time_windows-top+shift_samples, 0:theta_select_range, :]).max()
min_amp_index = int(np.where(real[time_windows-top-shift_samples:time_windows-top+shift_samples]==real_min_amp_value)[0])
max_amp_index = int(np.where(real[time_windows-top-shift_samples:time_windows-top+shift_samples]==real_max_amp_value)[0])
real_amp_top = real[time_windows-top-shift_samples+max_amp_index, 1:, cdp_select]
real_amp_base = real[time_windows-top-shift_samples+min_amp_index, 1:, cdp_select]
real_theta = np.arange(1, theta_nums)
# fig = plt.figure()
# ax = plt.subplot()
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Amplitude$")
# ax.scatter(real_theta, real_amp_top, color="blue")
# ax.scatter(real_theta, real_amp_base, color="red")
# plt.grid()
# plt.legend()
# plt.show()

############################################################################
# 绘制振幅随角度变化（人工合成地震数据）
Syn = Syn/abs(Syn[time_windows-shift_samples:time_windows+shift_samples, 0:theta_select_range]).max()
syn_min_amp_value = (Syn[time_windows-shift_samples:time_windows+shift_samples, 0:theta_select_range]).min()
syn_max_amp_value = (Syn[time_windows-shift_samples:time_windows+shift_samples, 0:theta_select_range]).max()
min_amp_index = int(np.where(Syn[time_windows-shift_samples:time_windows+shift_samples, :]==syn_min_amp_value)[0][0])
max_amp_index = int(np.where(Syn[time_windows-shift_samples:time_windows+shift_samples, :]==syn_max_amp_value)[0][0])
# fig = plt.figure()
# ax = plt.subplot()
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Amplitude$")
# ax.plot(theta[0]*180/np.pi, Syn[time_windows-int(len(t_samples)/2)+max_amp_index], color="blue", label="amp_top")
# ax.plot(theta[0]*180/np.pi, Syn[time_windows-int(len(t_samples)/2)+min_amp_index], color="red", label="amp_base")
# plt.grid()
# plt.legend()
# plt.show()

############################################################################
# 振幅散点拟合
def ani3_zhang(x, a, b, c):
	'''
	k = (2Vs/Vp)**2
	'''
	theta = x/180*np.pi
	# theta = x
	ani_k = 1.9
	return 1/2*a - ani_k/2*np.sin(theta)**2*b + np.tan(theta)**2/2*c

def iso_aki(x, a, b, c):
	'''
	k = Vs/Vp
	'''
	theta = x/180*np.pi
	# theta = x
	iso_k = 0.69
	return (1+np.tan(theta)**2)/2*a - 8*iso_k**2*np.sin(theta)**2/2*b + (1-4*iso_k**2*np.sin(theta)**2)/2*c

# popt_top_iso, pcov_top_iso = curve_fit(iso_aki, real_theta, real_amp_top)
# popt_base_iso, pcov_base_iso = curve_fit(iso_aki, real_theta, real_amp_base)

popt_top_ani, pcov_top_ani = curve_fit(ani3_zhang, real_theta, real_amp_top)
popt_base_ani, pcov_base_ani = curve_fit(ani3_zhang, real_theta, real_amp_base)

############################################################################
# 测试
# plt.figure()
# plt.imshow(real[:,5,:])
# plt.show()

# plt.figure()
# plt.imshow(real[:,:,10])
# plt.show()

# plt.figure()
# plt.imshow(Syn[top:bottom,::3])
# plt.show()

# plt.figure()
# plt.plot(wavelet)
# plt.show()

############################################################################
fig = plt.figure()
ax = plt.subplot()
plt.xlabel("$Θ(°)$")
plt.ylabel("$Amplitude$")
ax.plot(theta[0]*180/np.pi, Syn[time_windows-int(len(t_samples)/2)+max_amp_index], color="blue", label="amp_top")
ax.plot(theta[0]*180/np.pi, Syn[time_windows-int(len(t_samples)/2)+min_amp_index], color="red", label="amp_base")
# ax.plot(real_theta, iso_aki(real_theta, *popt_top_iso), color="blue", linestyle="-", label="fit:a=%5.3f, b=%5.3f, c=%5.3f"%tuple(popt_top_iso))
# ax.plot(real_theta, iso_aki(real_theta, *popt_base_iso), color="red", linestyle="-", label="fit:a=%5.3f, b=%5.3f, c=%5.3f"%tuple(popt_base_iso))
# ax.plot(real_theta, ani3_zhang(real_theta, *popt_top_ani), color="blue", linestyle="-", label="fit: A=%5.3f, B=%5.3f, C=%5.3f"%tuple(popt_top_ani))
# ax.plot(real_theta, ani3_zhang(real_theta, *popt_base_ani), color="red", linestyle="-", label="fit: A=%5.3f, B=%5.3f, C=%5.3f"%tuple(popt_base_ani))
# ax.scatter(real_theta, real_amp_top/abs(real_amp_top).max(), color="blue")
# ax.scatter(real_theta, real_amp_base/abs(real_amp_base).max(), color="red")
ax.scatter(real_theta, real_amp_top, color="blue")
ax.scatter(real_theta, real_amp_base, color="red")
plt.grid()
plt.legend()
plt.show()
############################################################################

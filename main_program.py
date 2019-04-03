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
# import segyio

#采样间隔 2ms
# # #读取测井数据
data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\Analysis_of_AVO.csv"))
del data["Unnamed: 0"]

# data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_05.csv"))
# del data["Unnamed: 0"]

# data1 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_045.csv"))
# del data1["Unnamed: 0"]

# data2 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_04.csv"))
# del data2["Unnamed: 0"]

# data3 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_035.csv"))
# del data3["Unnamed: 0"]

# data4 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_03.csv"))
# del data4["Unnamed: 0"]

# data5 = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_025.csv"))
# del data5["Unnamed: 0"]

# data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line2_164.csv"))
# del data["Unnamed: 0"]

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

# time windows
time_window = 0 # 划定研究时窗　单位：采样点 （时窗对应反射系数时窗）

# temp
data = copy.copy(data)

# 此处使用提取的地震子波
# wave_0 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_01_08.txt", skiprows=56)
# wave_1 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_08_15.txt", skiprows=56)
# wave_2 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_15_22.txt", skiprows=56)
# wave_3 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_22_29.txt", skiprows=56)
# wave_4 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_29_36.txt", skiprows=56)


# 地震子波人工理想的雷克子波 （不同角度）
# theta_max, theta_spaces, Fmax, t_samples, dt_space = 50, 1, 30, np.arange(-50, 50, 1), 0.002 #采样间隔2ms,采样点数100
theta_max, theta_nums, Fmax, t_samples, dt_space = 40, 40, 30, np.arange(-50, 50, 1), 0.002 #采样间隔2ms,采样点数100
# ricker = (1-2*(np.pi*Fmax*t_samples*dt_space)**2)*np.exp(-(np.pi*Fmax*t_samples*dt_space)**2)
# theta = np.arange(0, theta_max, theta_space)*np.pi/180  #弧度
theta = np.linspace(0.01, theta_max, theta_nums)*np.pi/180  #弧度
theta = theta + np.zeros((data.shape[0], len(theta)))

# # 地震子波真实采集的 （不同角度）
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\wave0.txt", skiprows=42)

# wave_1 = np.loadtxt("D:\\Physic_Model_Data\\wave1.txt", skiprows=42)

# wave_2 = np.loadtxt("D:\\Physic_Model_Data\\wave2.txt", skiprows=42)

# wave_3 = np.loadtxt("D:\\Physic_Model_Data\\wave3.txt", skiprows=42)

# wave_4 = np.loadtxt("D:\\Physic_Model_Data\\wave4.txt", skiprows=42)

# ricker = wave_4

# # wigb绘图
# def plot_vawig(axhdl, data, t, excursion, min_plot_time, max_plot_time):

#     import numpy as np
#     import matplotlib.pyplot as plt

#     data = data.T
#     [ntrc, nsamp] = data.shape
    
#     t = np.hstack([0, t, t.max()])
    
#     for i in range(0, ntrc):
#         tbuf = excursion * data[i,:] / np.max(np.abs(data)) + i
        
#         tbuf = np.hstack([i, tbuf, i])
            
#         axhdl.plot(tbuf, t, color='black', linewidth=0.5)
#         plt.fill_betweenx(t, tbuf, i, where=tbuf>i, facecolor=[0.6,0.6,1.0], linewidth=0)
#         plt.fill_betweenx(t, tbuf, i, where=tbuf<i, facecolor=[1.0,0.7,0.7], linewidth=0)
    
#     axhdl.set_xlim((-excursion, ntrc+excursion))
#     axhdl.set_ylim((min_plot_time, max_plot_time))
#     axhdl.xaxis.tick_top()
#     axhdl.xaxis.set_label_position('top')
#     axhdl.invert_yaxis()
    

# # #划定研究时窗　单位：采样点 （时窗对应相速度时窗）
# time_window = 1606
# # Thomsen 各向异性相速度 Vp Vsv Vsh
# theta_2pi = np.arange(0, 360, 1)*np.pi/180  #弧度
# weak_elastic = forward.Weak_Anisotropy(data["Vp"], data["Vs"], data["Rho"], theta_2pi, data["Epsilon"], data["Delta"], data["Gamma"])
# Vp = weak_elastic.weakVp_phase()
# Vsv = weak_elastic.weakVsv_phase()
# Vsh = weak_elastic.weakVsh_phase()
# Vp[np.isnan(Vp)] = 0
# Vsv[np.isnan(Vsv)] = 0
# Vsh[np.isnan(Vsh)] = 0
# Vp, Vsv, Vsh = Vp[time_window], Vsv[time_window], Vsh[time_window]
# fig = plt.figure()
# ax = fig.add_subplot(111, polar=True)
# ax.plot(theta_2pi, Vp, color="red", label="$Vp$")
# ax.plot(theta_2pi, Vsv, color="blue", label="$Vsv$")
# ax.plot(theta_2pi, Vsh, color="orange", label="$Vsh$")
# ax.set_title("$Clay ontent:0.5$")
# plt.legend()
# plt.show()

# # #划定研究时窗　单位：采样点 （时窗对应相速度时窗）
# time_window = 1606
# # Thomsen 各向异性相速度 Vp Vsv Vsh
# theta_2pi = np.arange(0, 360, 1)*np.pi/180  #弧度
# weak_elastic  = forward.Weak_Anisotropy(data["Vp"], data["Vs"], data["Rho"], theta_2pi, data["Epsilon"], data["Delta"], data["Gamma"])
# weak_elastic1 = forward.Weak_Anisotropy(data1["Vp"], data1["Vs"], data1["Rho"], theta_2pi, data1["Epsilon"], data1["Delta"], data1["Gamma"])
# weak_elastic2 = forward.Weak_Anisotropy(data2["Vp"], data2["Vs"], data2["Rho"], theta_2pi, data2["Epsilon"], data2["Delta"], data2["Gamma"])
# weak_elastic3 = forward.Weak_Anisotropy(data3["Vp"], data3["Vs"], data3["Rho"], theta_2pi, data3["Epsilon"], data3["Delta"], data3["Gamma"])
# weak_elastic4 = forward.Weak_Anisotropy(data4["Vp"], data4["Vs"], data4["Rho"], theta_2pi, data4["Epsilon"], data4["Delta"], data4["Gamma"])
# weak_elastic5 = forward.Weak_Anisotropy(data5["Vp"], data5["Vs"], data5["Rho"], theta_2pi, data5["Epsilon"], data5["Delta"], data5["Gamma"])
# Vp = weak_elastic.weakVp_phase()
# Vp1 = weak_elastic1.weakVp_phase()
# Vp2 = weak_elastic2.weakVp_phase()
# Vp3 = weak_elastic3.weakVp_phase()
# Vp4 = weak_elastic4.weakVp_phase()
# Vp5 = weak_elastic5.weakVp_phase()
# Vp[np.isnan(Vp)] = 0
# Vp1[np.isnan(Vp1)] = 0
# Vp2[np.isnan(Vp2)] = 0
# Vp3[np.isnan(Vp3)] = 0
# Vp4[np.isnan(Vp4)] = 0
# Vp5[np.isnan(Vp5)] = 0
# fig = plt.figure()
# ax = fig.add_subplot(111, polar=True)
# ax.plot(theta_2pi, Vp[time_window], color="#8c564b", label="porosity_16.4%")
# ax.plot(theta_2pi, Vp1[time_window], color="#1f77b4", label="porosity_11.9%")
# ax.plot(theta_2pi, Vp2[time_window], color="#ff7f0e", label="porosity_6.5%")
# ax.plot(theta_2pi, Vp3[time_window], color="#2ca02c", label="porosity_4.7%")
# ax.plot(theta_2pi, Vp4[time_window], color="#d62728", label="porosity_3.8%")
# ax.plot(theta_2pi, Vp5[time_window], color="#9467bd", label="porosity_2.1%")
# # ax.plot(theta_2pi, Vp[time_window], color="#8c564b", label="clay_50%")
# # ax.plot(theta_2pi, Vp1[time_window], color="#1f77b4", label="clay_45%")
# # ax.plot(theta_2pi, Vp2[time_window], color="#ff7f0e", label="clay_40%")
# # ax.plot(theta_2pi, Vp3[time_window], color="#2ca02c", label="clay_35%")
# # ax.plot(theta_2pi, Vp4[time_window], color="#d62728", label="clay_30%")
# # ax.plot(theta_2pi, Vp5[time_window], color="#9467bd", label="clay_25%")
# ax.set_title("$Phase$ $Velocity$")
# plt.legend()
# plt.show()

# # 划定研究时窗　单位：采样点 （时窗对应相速度时窗）
# time_window = 1606
# # Darey_Hron 各向异性相速度 Vp Vsv Vsh
# theta_2pi = np.arange(0, 360, 1)*np.pi/180  #弧度
# weak_elastic = forward.Darey_Hron_trans(data["Vp"], data["Vs"], data["Rho"], theta_2pi, data["Epsilon"], data["Delta"], data["Gamma"])
# Vp = weak_elastic.DH_Vp_phase()
# Vsv = weak_elastic.DH_Vsv_phase()
# Vsh = weak_elastic.DH_Vsh_phase()
# Vp[np.isnan(Vp)] = 0
# Vsv[np.isnan(Vsv)] = 0
# Vsh[np.isnan(Vsh)] = 0
# Vp, Vsv, Vsh = Vp[time_window], Vsv[time_window], Vsh[time_window]
# fig = plt.figure()
# ax = fig.add_subplot(111, polar=True)
# ax.plot(theta_2pi, Vp, color="red", label="$Vp$")
# ax.plot(theta_2pi, Vsv, color="blue", label="$Vsv$")
# ax.plot(theta_2pi, Vsh, color="orange", label="$Vsh$")
# # ax.set_title("$Clay Content(accurate):0.5$")
# plt.legend()
# plt.show()

# #反射系数
# #1 wiggen 三项式 （基于aki_richard）
# # wiggen = forward.Aki_Richard(data["Vp"], data["Vs"], data["Rho"], theta)
# # reflect_1 = wiggen.wiggens()
# # reflect_1[np.isnan(reflect_1)] = 0
# 2 aki_richards
# aki = forward.Aki_Richard(data["Vp"], data["Vs"], data["Rho"], theta)
# reflect_2 = aki.aki_richards()
# reflect_2[np.isnan(reflect_2)] = 0
# 3 Ruger　横向各向同性　
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# reflect_3 = vti_appr.ruger_approximate()
# reflect_3[np.isnan(reflect_3)] = 0
# 4 thomsen 相速度反射系数
# theta_2pi = np.linspace(0.01, theta_max, theta_nums)*np.pi/180  #弧度
# weak_elastic = forward.Darey_Hron_trans(data["Vp"], data["Vs"], data["Rho"], theta_2pi, data["Epsilon"], data["Delta"], data["Gamma"])
# Vp = weak_elastic.DH_Vp_phase()
# Vsv = weak_elastic.DH_Vsv_phase()
# Vsh = weak_elastic.DH_Vsh_phase() # exact
# # weak_elastic = forward.Weak_Anisotropy(data["Vp"], data["Vs"], data["Rho"], theta_2pi, data["Epsilon"], data["Delta"], data["Gamma"])
# # Vp = weak_elastic.weakVp_phase()
# # Vsv = weak_elastic.weakVsv_phase()
# # Vsh = weak_elastic.weakVsh_phase() # approximatly
# Vp[np.isnan(Vp)] = 0
# Vsv[np.isnan(Vsv)] = 0
# Vsh[np.isnan(Vsh)] = 0
# reflect_4 = forward.Normal(Vp, data["Rho"])
# reflect_4 =reflect_4.reflection_p()
# reflect_4[np.isnan(reflect_4)] = 0
# 5 Zoeppritz 精确解表达式　某一时窗
# time_window = 1605 # 划定研究时窗　单位：采样点 （时窗对应反射系数时窗）
# zoeppritz = forward.Zoeppritz(data["Vp"], data["Vs"], data["Rho"], theta)
# reflect_5 = zoeppritz.zoeppritz_exact(time_window)
# reflect_5[np.isnan(reflect_5)] = 0
# 6 Zoeppritz 精确解表达式　全部时窗　
zoeppritz = forward.Zoeppritz(data["Vp"], data["Vs"], data["Rho"], theta)
reflect_6 = zoeppritz.zoeppritz_exact_all()
reflect_6[np.isnan(reflect_6)] = 0
reflect_iso = reflect_6[0]
# 7 Thomsen 横向各向同性
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# _, reflect_7 = vti_appr.iso_approximate()
# reflect_7[np.isnan(reflect_7)] = 0
# 8 Zhang_3 横向各向同性
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# _, _, _, _, _, _, reflect_8 = vti_appr.ani_zhang()
# reflect_8[np.isnan(reflect_8)] = 0
# 9 Zhang_5 横向各向同性
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# reflect_9 = vti_appr.ani_zhang_5() # k = (2Vs/Vp)**2
# reflect_9[np.isnan(reflect_9)] = 0
# 10 Graebner 精确解表达式　某一时窗
# vti_exact_reflct = forward.Graebner(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# reflect_10 = vti_exact_reflct.vti_exact(time_window)
# reflect_10[np.isnan(reflect_10)] = 0
# 11 Graebner 精确解表达式　全部时窗　
vti_exact_reflct = forward.Graebner(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
reflect_11 = vti_exact_reflct.vti_exact_all()
reflect_11[np.isnan(reflect_11)] = 0
reflect_ani = reflect_11[0]

# # 不同泥质含量参数条件下岩石的各向同性与各向异性反射系数计算（包括绘图）
# time_window = 1605 # 划定研究时窗　单位：采样点 （时窗对应相速度时窗）
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
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Reflection Coefficient$")
# ax.plot(theta[0]*180/np.pi, reflect[time_window], color="#8c564b", label="clay_50%")
# ax.plot(theta[0]*180/np.pi, reflect_1[time_window], color="#1f77b4", label="clay_45%")
# ax.plot(theta[0]*180/np.pi, reflect_2[time_window], color="#ff7f0e", label="clay_40%")
# ax.plot(theta[0]*180/np.pi, reflect_3[time_window], color="#2ca02c", label="clay_35%")
# ax.plot(theta[0]*180/np.pi, reflect_4[time_window], color="#d62728", label="clay_30%")
# ax.plot(theta[0]*180/np.pi, reflect_5[time_window], color="#9467bd", label="clay_25%")
# ax.plot(theta[0]*180/np.pi, reflecta[time_window], color="#8c564b", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_1a[time_window], color="#1f77b4", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_2a[time_window], color="#ff7f0e", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_3a[time_window], color="#2ca02c", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_4a[time_window], color="#d62728", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_5a[time_window], color="#9467bd", linestyle="--")
# plt.grid()
# plt.legend()
# plt.show()

# # 不同孔隙度参数条件下岩石的各向同性与各向异性反射系数计算（包括绘图）
# time_window = 1605 # 划定研究时窗　单位：采样点 （时窗对应相速度时窗）
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
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Reflection Coefficient$")
# ax.plot(theta[0]*180/np.pi, reflect[time_window], color="#8c564b", label="porosity_16.4%")
# ax.plot(theta[0]*180/np.pi, reflect_1[time_window], color="#1f77b4", label="porosity_11.5%")
# ax.plot(theta[0]*180/np.pi, reflect_2[time_window], color="#ff7f0e", label="porosity_6.5%")
# ax.plot(theta[0]*180/np.pi, reflect_3[time_window], color="#2ca02c", label="porosity_4.7%")
# ax.plot(theta[0]*180/np.pi, reflect_4[time_window], color="#d62728", label="porosity_3.8%")
# ax.plot(theta[0]*180/np.pi, reflect_5[time_window], color="#9467bd", label="porosity_2.1%")
# ax.plot(theta[0]*180/np.pi, reflecta[time_window], color="#8c564b", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_1a[time_window], color="#1f77b4", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_2a[time_window], color="#ff7f0e", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_3a[time_window], color="#2ca02c", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_4a[time_window], color="#d62728", linestyle="--")
# ax.plot(theta[0]*180/np.pi, reflect_5a[time_window], color="#9467bd", linestyle="--")
# plt.grid()
# plt.legend()
# plt.show()

# # 子波 Toeplitz矩阵(1)
# temp = int(ricker.shape[0]/2)
# row = np.zeros((1, data.shape[0])).flatten()
# column = np.zeros((1, data.shape[0]+temp)).flatten()
# column[0:ricker.shape[0]] = ricker
# toeplitz = (linalg.toeplitz(row, column)[:, temp:])[:-1, :-1]

# # 子波 Toeplitz矩阵(2)
# def convmtx(h, n):
#     """
#     Equivalent of MATLAB's convmtx function, http://www.mathworks.com/help/signal/ref/convmtx.html.
    
#     Makes the convolution matrix, C. The product C.x is the convolution of h and x.
    
#     Args
#         h (ndarray): a 1D array, the kernel.
#         n (int): the number of rows to make.
        
#     Returns
#         ndarray. Size m+n-1
#     """
#     col_1 = np.r_[h[0], np.zeros(n-1)]
#     row_1 = np.r_[h, np.zeros(n-len(h))]
#     return linalg.toeplitz(col_1, row_1)
# toeplitz = convmtx(ricker, data.shape[0])

# #　合成地震记录 （不同角度）
# def synthetic(reflect, wavelet):
# 	Syn = np.zeros((reflect.shape[0], reflect.shape[1]))
# 	for i in range(Syn.shape[1]):
# 	    Syn[:, i] = np.convolve(reflect[:, i], wavelet, "same").flatten()
# 	return Syn
# Syn = synthetic(reflect_8, wavelet)

# 绘制振幅随角度变化（真实地震数据）
# top, bottom, cdp_beg, cdp_end = 1500, 1885, 420, 555 
# time_window = 1606
# shift_samples = int(len(t_samples)/2)
# real_seismic = segyio.tools.cube("D:\\Physic_Model_Data\\L_1.prj\\seismic.dir\\angle_gather_target.sgy")[0,:,:,:].T
# # real_seismic = segyio.tools.cube("F:\\Physic_Model_Data\\L1.prj\\seismic.dir\\angle_gather.sgy")[0,:,:,:].T
# real = real_seismic[top:bottom, :, cdp_beg:cdp_end]
# # real = real/abs(real).max()
# real_min_amp_value = (real[time_window-top-shift_samples:time_window-top+shift_samples, :, :]).min()
# real_max_amp_value = (real[time_window-top-shift_samples:time_window-top+shift_samples, :, :]).max()
# min_amp_index = int(np.where(real[time_window-top-shift_samples:time_window-top+shift_samples]==real_min_amp_value)[0])
# max_amp_index = int(np.where(real[time_window-top-shift_samples:time_window-top+shift_samples]==real_max_amp_value)[0])
# real_amp_top = real[time_window-top-shift_samples+max_amp_index, :, 10]
# real_amp_base = real[time_window-top-shift_samples+min_amp_index, :, 10]
# real_theta = np.arange(1, 16)*3
# fig = plt.figure()
# ax = plt.subplot()
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Amplitude$")
# ax.scatter(real_theta, real_amp_top, color="blue")
# ax.scatter(real_theta, real_amp_base, color="red")
# plt.grid()
# plt.legend()
# plt.show()

# 绘制振幅随角度变化（人工合成地震数据）
# Syn = Syn/abs(Syn[time_window-shift_samples:time_window+shift_samples, :]).max()
# syn_min_amp_value = (Syn[time_window-shift_samples:time_window+shift_samples, 0:20]).min()
# syn_max_amp_value = (Syn[time_window-shift_samples:time_window+shift_samples, 0:20]).max()

# min_amp_index = int(np.where(Syn[time_window-shift_samples:time_window+shift_samples, :]==syn_min_amp_value)[0][0])
# max_amp_index = int(np.where(Syn[time_window-shift_samples:time_window+shift_samples, :]==syn_max_amp_value)[0][0])

# fig = plt.figure()
# ax = plt.subplot()
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Amplitude$")
# ax.plot(theta[0]*180/np.pi, Syn[time_window-int(len(t_samples)/2)+max_amp_index], color="blue", label="amp_top")
# ax.plot(theta[0]*180/np.pi, Syn[time_window-int(len(t_samples)/2)+min_amp_index], color="red", label="amp_base")
# plt.grid()
# plt.legend()
# plt.show()

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

# # 打印子波振幅与人工合成地震记录振幅
# print(ricker.max())
# print(Syn[1500:1850, 4].max())

# #　合成地震记录 （不同角度）
# class Operator(object):
#     """docstring for Operator"""
#     def __init__(self, ricker):
#         super(Operator, self).__init__()
#         self.ricker = ricker

#     def forward(self, reflect):
#         synthetic = np.zeros((reflect.shape[0], reflect.shape[1]))
#         for i in range(synthetic.shape[1]):
#             synthetic[:, i] = np.convolve(reflect[:, i], self.ricker, "same").flatten()
#         return synthetic
        
# Syn = Operator(ricker).forward(reflect)

# 绘制一道的合成地震记录 
# fig = plt.figure() 
# ax = plt.subplot()
# ax.plot(time, Syn[:, 1], color="black")
# ax.fill_between(time, Syn[:, 1], 0, where=Syn[:, 1]>0, facecolor="black")
# plt.show()

# #　反演绘制Rp Rs Rr
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

# 绘制反射系数随角度变化　
# fig = plt.figure()
# ax = plt.subplot()
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Reflection Coefficient$")
# ax.plot(theta[0]*180/np.pi, reflect_2[time_window], color="orange", label="Aki_Richards")
# ax.plot(theta[0]*180/np.pi, reflect_3[time_window], color="green", label="Ruger_1")
# ax.plot(theta_2pi*180/np.pi, reflect_4[time_window], color="brown", label="Darey_Hron")
# ax.plot(theta[0]*180/np.pi, reflect_7[time_window], color="purple", label="Ruger_2")
# ax.plot(theta[0]*180/np.pi, reflect_8[time_window], color="purple", label="Zhang_ani3")
# ax.plot(theta[0]*180/np.pi, reflect_9[time_window], color="pink", label="Zhang_ani5")
# ax.plot(theta[0]*180/np.pi, reflect_1[time_window], color="yellow", label="Wiggen")
# ax.plot(theta[0]*180/np.pi, reflect_5[0], color="red", label="Zoeppritz")
# ax.plot(theta[0]*180/np.pi, reflect_1[0], color="black", alpha=0.2)
# ax.plot(theta[0]*180/np.pi, reflect_10[0], color="blue", label="Graebner")
# ax.plot(theta[0]*180/np.pi, reflect_12[time_window], color="blue", label="Other")
# plt.grid()
# plt.legend()
# plt.show()

# 绘制反射系数随角度变化　
fig = plt.figure()
ax = plt.subplot()
plt.xlabel("$Θ(°)$")
plt.ylabel("$Reflection Coefficient$")
ax.plot(theta[0]*180/np.pi, reflect_iso[0], color="red", label="Zoeppritz")
ax.plot(theta[0]*180/np.pi, reflect_iso[2], color="red", label="Zoeppritz")
ax.plot(theta[0]*180/np.pi, reflect_iso[4], color="red", label="Zoeppritz")
ax.plot(theta[0]*180/np.pi, reflect_ani[0], color="blue", label="Graebner")
ax.plot(theta[0]*180/np.pi, reflect_ani[2], color="blue", label="Graebner")
ax.plot(theta[0]*180/np.pi, reflect_ani[4], color="blue", label="Graebner")
plt.ylim(-0.4, 0.2)
plt.grid()
plt.legend()
plt.show()

# #反射系数随角度变化（残差）
# fig = plt.figure()
# ax = plt.subplot()
# plt.xlabel("$Θ(°)$")
# plt.ylabel("$Residual Error$")
# ax.plot(theta[0]*180/np.pi, reflect_2[time_window]-reflect_5[0], color="blue", label="Aki-Richards")
# ax.plot(theta[0]*180/np.pi, reflect_3[time_window]-reflect_5[0], color="red", label="Ruger")
# ax.plot(theta[0]*180/np.pi, reflect_7[time_window]-reflect_5[0], color="purple", label="Thomsen_vti")
# #ax.plot(theta[0]*180/np.pi, reflect_1[time_window]-reflect_5[0], color="yellow", label="Wiggen")
# #ax.plot(theta[0]*180/np.pi, reflect_5[0], color="orange", label="Zoeppritz")
# #ax.plot(theta[0]*180/np.pi, reflect_1[0], color="black", alpha=0.2)
# plt.grid()
# plt.legend()
# plt.show()

# segyio 输出人工合成地震数据 
# fig = plt.figure()
# ax = plt.subplot()
# plt.imshow(Syn[1560:1620, :])
# plt.show()

# # wigb绘制人工合成地震数据
# dt = (2*1e-3)  #1s=1000ms (two-way travel time) sample space:2ms
# min_plot_time = 1.5  #units s
# max_plot_time = 4.5  #units s
# excursion = 2  # spacing
# tmax = data.shape[0]*dt  #nsamples*dt,  units: s
# t = np.arange(0, tmax, dt)

# # Vp,Vs,Rho logs:
# fig = plt.figure(figsize=(8, 4), dpi=300)
# fig.set_facecolor('white')
# # fig.suptitle("$ANI$", fontsize=20)
# lyr_times = t

# ax0a = fig.add_subplot(141)
# l_vp_dig, = ax0a.plot(data["Vp"]/1000, t, 'k', lw=1)
# ax0a.set_ylim((min_plot_time,max_plot_time))
# ax0a.set_xlim(1.0, 4.0)
# ax0a.invert_yaxis()
# ax0a.set_ylabel('TWT (sec)')
# ax0a.xaxis.tick_top()
# ax0a.xaxis.set_label_position('top')
# ax0a.set_xlabel("Vp (km/s)")
# ax0a.axhline(lyr_times[1606], color='blue', lw=1, alpha=0.5)
# ax0a.axhline(lyr_times[1751], color='red', lw=1, alpha=0.5)
# ax0a.grid()

# plt.text(2.55,
#         min_plot_time + (lyr_times[985] - min_plot_time)/2.,
#         'Layer 1',
#         fontsize=14,
#         horizontalalignment='right')
# plt.text(2.55,
#         lyr_times[985] + (lyr_times[1175] - lyr_times[985])/2. + 0.002,
#         'Layer 2',
#         fontsize=14,
#         horizontalalignment='right')
# plt.text(2.55,
#         lyr_times[1175] + (lyr_times[1606] - lyr_times[1175])/2. + 0.002,
#         'Layer 3',
#         fontsize=14,
#         horizontalalignment='right')
# plt.text(2.55,
#         lyr_times[1606] + (lyr_times[1752] - lyr_times[1606])/2. + 0.002,
#         'Layer 4',
#         fontsize=14,
#         horizontalalignment='right')
# plt.text(2.55,
#         lyr_times[1752] + (lyr_times[1902] - lyr_times[1752])/2. + 0.002,
#         'Layer 5',
#         fontsize=14,
#         horizontalalignment='right')
# plt.text(2.55,
#         lyr_times[1902] + (max_plot_time - lyr_times[1902])/2.,
#         'Layer 6',
#         fontsize=14,
#         horizontalalignment='right')

# ax0b = fig.add_subplot(142)
# l_vs_dig, = ax0b.plot(data["Vs"]/1000, t, 'k', lw=1)
# ax0b.set_ylim((min_plot_time,max_plot_time))
# ax0b.set_xlim((0.8, 2.0))
# ax0b.invert_yaxis()
# ax0b.xaxis.tick_top()
# ax0b.xaxis.set_label_position('top')
# ax0b.set_xlabel("Vs (km/s)")
# ax0b.set_yticklabels('')
# ax0b.axhline(lyr_times[1606], color='blue', lw=1, alpha=0.5)
# ax0b.axhline(lyr_times[1751], color='red', lw=1, alpha=0.5)
# ax0b.grid()

# ax0c = fig.add_subplot(143)
# l_rho_dig, = ax0c.plot(data["Rho"], t, 'k', lw=1)
# ax0c.set_ylim((min_plot_time,max_plot_time))
# ax0c.set_xlim((0.5, 3.0))
# ax0c.invert_yaxis()
# ax0c.xaxis.tick_top()
# ax0c.xaxis.set_label_position('top')
# ax0c.set_xlabel("Den (g/cm^3)")
# ax0c.set_yticklabels('')
# ax0c.axhline(lyr_times[1606], color='blue', lw=1, alpha=0.5)
# ax0c.axhline(lyr_times[1751], color='red', lw=1, alpha=0.5)
# ax0c.grid()

# ax0c = fig.add_subplot(144)
# l_Epsilon_dig, = ax0c.plot(data["Epsilon"], t, 'k', linestyle="-", lw=1, label="Epsilon")
# l_Delta_dig, = ax0c.plot(data["Delta"], t, 'k', linestyle=":", lw=1, label="Delta")
# l_Gamma_dig, = ax0c.plot(data["Gamma"], t, 'k', linestyle="--", lw=1, label="Gamma")
# ax0c.set_ylim((min_plot_time,max_plot_time))
# ax0c.set_xlim((0.0, 0.25))
# ax0c.invert_yaxis()
# ax0c.xaxis.tick_top()
# ax0c.xaxis.set_label_position('top')
# ax0c.set_xlabel("Thomsen parameters")
# ax0c.set_yticklabels('')
# ax0c.legend()
# ax0c.axhline(lyr_times[1606], color='blue', lw=1, alpha=0.5)
# ax0c.axhline(lyr_times[1751], color='red', lw=1, alpha=0.5)
# ax0c.grid()

# # # 人工合成地震道绘图
# ax1 = fig.add_subplot(122)
# plot_vawig(ax1, Syn, t[1:], excursion, min_plot_time, max_plot_time)
# l_int1, = ax1.plot(lyr_times[1606], color='blue', lw=2)
# l_int2, = ax1.plot(lyr_times[1751], color='red', lw=2)
# plt.legend([l_int1,l_int2], ['Interface 1', 'Interface 2'], loc=4)
# ax1.set_xlabel("$Trace$ $(theta)$")
# ax1.set_yticklabels('')
# # ax1.set_ylabel("$Time (sec)$")
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
        
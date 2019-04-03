# -@- coding: utf-8 -@-
"""
Created on Mon Sep 24 12:13:47 2018

@author: 泳浩
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import forward_problem as forward
import scipy.linalg as linalg
import smooth_well
import copy


def convmtx(s, n):
    """
    Toeplitz方阵建立
    s为序列，n为长和宽
    """
    t = int(s.shape[0]/2)
    col_1 = np.r_[s[0], np.zeros(n-1+t)]
    row_1 = np.r_[s, np.zeros(n-len(s)+t)]
    temp = linalg.toeplitz(col_1, row_1)
    return temp[:-t, t:]

def transform_G(coef1, coef2, coef3):
    '''
    系数矩阵，传入的必须为数组或者向量，不能是Series
    '''
    Coe1, Coe2, Coe3 = np.diag(coef1), np.diag(coef2), np.diag(coef3)
    G = np.hstack((Coe1, Coe2, Coe3)) # 系数矩阵
    return G

def transform_m(para_1, para_2, para_3):
    '''
    模型参数，传入的必须为数组或者向量，不能是Series
    '''
    m = np.vstack((para_1, para_2, para_3)).reshape(-1, 1)
    m[np.isnan(m)] = 0 # 模型参数
    return m

def cov_mat(para_1, para_2, para_3, sigma_2):
    '''
    协方差矩阵，传入的必须为数组或者向量，不能是Series
    '''
    Cov = np.cov(np.hstack((para_1, para_2, para_3)).reshape(3, -1))
    lam = sigma_2*linalg.inv(Cov)
    I_lam = np.kron(lam, np.eye(data.shape[0]-1))
    # I_lam = np.kron(np.eye(data.shape[0]-1), lam)
    return [Cov, I_lam]

def inv_param(first_param, impedance):
    """
    将阻抗IVp,IVs,Ivd转换为所需要的参数Vp,Vs,Vd(迭代)
    """
    parameters = []
    for i in np.array(impedance).flatten():
        first_param = first_param*(i+2)/(2-i)
        parameters.append(first_param)
    return parameters

#读取测井数据
data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_05.csv"))
del data["Unnamed: 0"]
# data = data.rolling(15).mean()

# 井数据范围
# data = data.iloc[990:1901, :] 
# data = data.iloc[985:1901, :]

# 平滑法2: Lowess
smdata = smooth_well.Smooth(data)
data_s = smdata.sm()
data_s.to_csv("data_smooth.csv")

# # 平滑法3: 求平均
# data_s = copy.copy(data)
# data_s["Vp"] = data["Vp"].mean()
# data_s["Vs"] = data["Vs"].mean()
# data_s["Rho"] = data["Rho"].mean()

# # 平滑法4: 求平均
# data_s = data.rolling(200).mean()

##########################################################################################################
# 地震子波 （不同角度）
theta_max, theta_space, Fmax, t_samples, dt_space = 50, 1, 30, np.arange(-50, 50, 1), 0.002 #采样间隔2ms,采样点数100
ricker = (1-2*(np.pi*Fmax*t_samples*dt_space)**2)*np.exp(-(np.pi*Fmax*t_samples*dt_space)**2)
theta = np.arange(0, theta_max, theta_space)*np.pi/180  #弧度
theta = theta + np.zeros((data.shape[0], len(theta)))
theta = theta[:, :30:10]

# Thomsen 横向各向同性
vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, 
                            data["Epsilon"], data["Delta"], data["Gamma"])
k = (data["Vs"].mean()/data["Vp"].mean())
IVp, IVs, IVd, coef1, coef2, coef3, rpp = vti_appr.iso_ani_paras(k)
reflect_7 = vti_appr.iso_approximate()
reflect_7[np.isnan(reflect_7)] = 0 # 原始真实模型（目标）

# 合成记录，子波矩阵
W1 = convmtx(ricker, data.shape[0])[:-1, :-1] # 子波矩阵
W = np.kron(np.eye(3), W1) # 子波矩阵（块3@3）
###########################################################################################################

#############################################################################
# # 反演(1 共轭梯度下降)
# Thomsen 横向各向同性（建立初始模型）
vti_appr = forward.Thomsen(data_s["Vp"], data_s["Vs"], data_s["Rho"],theta, 
                            data_s["Epsilon"], data_s["Delta"], data_s["Gamma"])
k = (data_s["Vs"].mean()/data_s["Vp"].mean())
IVp_init, IVs_init, IVd_init, coef1_init, coef2_init, coef3_init, _ = vti_appr.iso_ani_paras(k) # 建立的初始模型

G = [] # 系数矩阵
for i in range(3):
    temp = transform_G(coef1[:, i], coef2[:, i], coef3[:, i]) # 模型参数
    G.append(temp)
G = np.vstack((G[0],G[1],G[2]))
m_init = transform_m(IVp_init, IVs_init, IVd_init) # 初始模型参数
m_real = transform_m(IVp, IVs, IVd) # 真实型参数
m_init0 = copy.copy(m_init)

syn2 = (W1@reflect_7).reshape((-1, 1), order="F") # 原始真实模型地震记录
syn3 = (W@G@m_init) # 初始模型地震记录

err_ls=[]
epoch=[]
step = 1
_, I_lam = cov_mat(IVp, IVs, IVd, 0.01)
plt.ion()
fig2 = plt.figure()

m = copy.copy(m_init)
d = copy.copy(syn2)
F = W@G
r = d-F@m_init
s = np.zeros_like(d)
beta = 0
for i in range(100):
    g = F.conj().T@r
    if i!=0:
        beta = np.vdot(g, g)/gamma
    gamma = np.vdot(g, g)
    s = g+beta*s
    delta_r = F@s
    alpha = -np.vdot(r, delta_r)/np.vdot(delta_r, delta_r)
    m = m-alpha*s
    r = r+alpha*delta_r

    error = np.sqrt(np.vdot(r, r))
    err_ls.append(error)
    epoch.append(i)   
    print("steps={}, error={}".format(i, error))
    plt.cla()
    plt.plot(epoch, err_ls)
    plt.xlabel("$epoch$")
    plt.ylabel("$error$")
    plt.pause(0.01)
    if err_ls[i]>err_ls[i-1]:
        break
plt.ioff()
m_init = m

# IVp,IVs,Ivd转换为Vp,Vs,Vd (inversion 1)
para_pred = m_init.reshape(-1, 3, order="F")
para_real = m_real.reshape(-1, 3, order="F")
Vp_temp = para_real[:, 1]
Vp_temp2 = para_pred[:, 1]

Vp_pred1 = inv_param(2600, Vp_temp)
Vp_pred2 = inv_param(2600, Vp_temp2)
############################################################################

# #############################################################################
# # # 反演(2)
# # Thomsen 横向各向同性（建立初始模型）
# vti_appr = forward.Thomsen(data_s["Vp"], data_s["Vs"], data_s["Rho"],theta, 
#                             data_s["Epsilon"], data_s["Delta"], data_s["Gamma"])
# k = (data_s["Vs"].mean()/data_s["Vp"].mean())
# IVp_init, IVs_init, IVd_init, coef1_init, coef2_init, coef3_init, _ = vti_appr.iso_ani_paras(k) # 建立的初始模型

# G = [] # 系数矩阵
# for i in range(3):
#     temp = transform_G(coef1[:, i], coef2[:, i], coef3[:, i]) # 模型参数
#     G.append(temp)
# G = np.vstack((G[0],G[1],G[2]))

# m_init = transform_m(IVp_init, IVs_init, IVd_init) # 初始模型参数
# m_real = transform_m(IVp, IVs, IVd) # 真实型参数
# m_init0 = copy.copy(m_init)

# syn2 = (W1@reflect_7).reshape((-1, 1), order="F") # 原始真实模型地震记录
# syn3 = (W@G@m_init) # 初始模型地震记录

# err_ls=[]
# epoch=[]
# step = 1
# _, I_lam = cov_mat(IVp, IVs, IVd, 0.01)
# plt.ion()
# fig2 = plt.figure()
# # r = reflect_7.reshape((-1, 1), order="F") # 差用反射系数
# for i in range(10):
#     delta_d = syn2-W@G@m_init # 差用褶积记录
#     # delta_d = r-G@m_init # 差用反射系数
#     m_init = m_init+linalg.inv(G.T@G+I_lam)@delta_d*step
#     error = np.sqrt((np.array(delta_d)**2).sum())
#     err_ls.append(error)
#     epoch.append(i)   
#     print("steps={}, error={}".format(i, error))
#     plt.cla()
#     plt.plot(epoch, err_ls)
#     plt.xlabel("$epoch$")
#     plt.ylabel("$error$")
#     plt.pause(0.01)
#     if err_ls[i]>err_ls[i-1]:
#         break
# plt.ioff()

# # IVp,IVs,Ivd转换为Vp,Vs,Vd (inversion 1)
# para_pred = m_init.reshape(-1, 3, order="F")
# para_real = m.reshape(-1, 3, order="F")
# Vp_temp = para_real[:, 0]
# Vp_temp2 = para_pred[:, 0]
# def inv_param(first_param, impedance):
#     """
#     将阻抗IVp,IVs,Ivd转换为所需要的参数Vp,Vs,Vd(迭代)
#     """
#     parameters = []
#     for i in np.array(impedance).flatten():
#         first_param = first_param*(i+2)/(2-i)
#         parameters.append(first_param)
#     return parameters
# Vp_pred1 = inv_param(2600, Vp_temp)
# Vp_pred2 = inv_param(2600, Vp_temp2)
# ############################################################################

############################################################################
# # # 反演(3)
# data.fillna(0)
# lnVp = (np.log(np.array(data["Vp"]/1000))[:-1]).reshape(-1, 1)
# lnVs = (np.log(np.array(data["Vs"]/1000))[:-1]).reshape(-1, 1)
# lnRho = (np.log(np.array(data["Rho"]))[:-1]).reshape(-1, 1)
# lnVp_init = (np.log(np.array(data_s["Vp"]/1000))[:-1]).reshape(-1, 1)
# lnVs_init = (np.log(np.array(data_s["Vs"]/1000))[:-1]).reshape(-1, 1)
# lnRho_init = (np.log(np.array(data_s["Rho"]))[:-1]).reshape(-1, 1)

# G = [] # 系数矩阵
# for i in range(3):
#     temp = transform_G(coef1[:, i], coef2[:, i], coef3[:, i]) # 模型参数
#     G.append(temp)
# G = np.vstack((G[0],G[1],G[2]))

# m_init = transform_m(lnVp_init, lnVs_init, lnRho_init) # 初始模型参数
# m_real = transform_m(lnVp, lnVs, lnRho) # 真实型参数
# m_init0 = copy.copy(m_init)

# # D = convmtx(np.array([0, -1, 1]), 3*(data.shape[0]-1)) # 微分算子建立
# D = convmtx(np.array([0, -1, 1]), (data.shape[0]-1)) # 微分算子建立
# D[data.shape[0]-2, :] = 0
# D = np.kron(np.ones((3, 3)), D)

# syn4 = (W1@reflect_7).reshape((-1, 1), order="F") # 原始真实模型地震记录
# syn2 = (W@G@D@m_real)
# syn3 = (W@G@D@m_init) # 初始模型地震记录

# plt.ion()
# fig2 = plt.figure()

# err_ls=[]
# epoch=[]
# _, I_lam = cov_mat(lnVp, lnVs, lnRho, 1)
# m = copy.copy(m_init)
# d = copy.copy(syn4)
# F = W@G@D
# r = d-F@m
# s = np.zeros_like(d)
# beta = 0
# for i in range(40):
#     g = (F.conj().T)@r
#     if i!=0:
#         beta = np.vdot(g, g)/gamma
#     gamma = np.vdot(g, g)
#     s = g+beta*s
#     delta_r = F@s
#     alpha = -np.vdot(r, delta_r)/np.vdot(delta_r, delta_r)
#     m = m-alpha*s
#     r = r+alpha*delta_r
#     error = np.sqrt(np.vdot(r, r))
#     err_ls.append(error)
#     epoch.append(i)   
#     print("steps={}, error={}".format(i, error))
#     plt.cla()
#     plt.plot(epoch, err_ls)
#     plt.xlabel("$epoch$")
#     plt.ylabel("$error$")
#     plt.pause(0.01)
#     if err_ls[i]>err_ls[i-1]:
#         break
# plt.ioff()
# m_init = m
#############################################################################

###########################################################################
# # # 反演(4 高斯牛顿迭代)
# data.fillna(0)
# lnVp = (np.log(np.array(data["Vp"]/1000))[:-1]).reshape(-1, 1)
# lnVs = (np.log(np.array(data["Vs"]/1000))[:-1]).reshape(-1, 1)
# lnRho = (np.log(np.array(data["Rho"]))[:-1]).reshape(-1, 1)
# lnVp_init = (np.log(np.array(data_s["Vp"]/1000))[:-1]).reshape(-1, 1)
# lnVs_init = (np.log(np.array(data_s["Vs"]/1000))[:-1]).reshape(-1, 1)
# lnRho_init = (np.log(np.array(data_s["Rho"]))[:-1]).reshape(-1, 1)

# G = [] # 系数矩阵
# for i in range(3):
#     temp = transform_G(coef1[:, i], coef2[:, i], coef3[:, i]) # 模型参数
#     G.append(temp)
# G = np.vstack((G[0],G[1],G[2]))

# m_init = transform_m(lnVp_init, lnVs_init, lnRho_init) # 初始模型参数
# m_real = transform_m(lnVp, lnVs, lnRho) # 真实型参数
# m_init0 = copy.copy(m_init)

# # D = convmtx(np.array([0, -1, 1]), 3*(data.shape[0]-1)) # 微分算子建立
# D = convmtx(np.array([0, -1, 1]), (data.shape[0]-1)) # 微分算子建立
# D[data.shape[0]-2, :] = 0
# D = np.kron(np.ones((3, 3)), D)

# syn4 = (W1@reflect_7).reshape((-1, 1), order="F") # 原始真实模型地震记录
# syn2 = (W@G@D@m_real)
# syn3 = (W@G@D@m_init) # 初始模型地震记录

# plt.ion()
# fig2 = plt.figure()

# F = W@G@D
# m = copy.copy(m_init)

# err_ls=[]
# epoch=[]
# _, I_lam = cov_mat(lnVp, lnVs, lnRho, 0.0001)
# step = 1

# for i in range(10):
#     delta_d = syn4-F@m # 差用褶积记录
#     m = m+linalg.inv(F.T@F+I_lam)@F.T@delta_d*step
#     error = np.sqrt(np.vdot(delta_d, delta_d))
#     err_ls.append(error)
#     epoch.append(i)   
#     print("steps={}, error={}".format(i, error))
#     plt.cla()
#     plt.plot(epoch, err_ls)
#     plt.xlabel("$epoch$")
#     plt.ylabel("$error$")
#     plt.pause(0.01)
#     if err_ls[i]>err_ls[i-1]:
#         break
# plt.ioff()
# m_init = m
############################################################################

############################################################################
# # 反演(5 zhang)
# #########################################################################################################
# # 地震子波 （不同角度）
# theta_max, theta_space, Fmax, t_samples, dt_space = 50, 1, 30, np.arange(-50, 50, 1), 0.002 #采样间隔2ms,采样点数100
# ricker = (1-2*(np.pi*Fmax*t_samples*dt_space)**2)*np.exp(-(np.pi*Fmax*t_samples*dt_space)**2)
# theta = np.arange(0, theta_max, theta_space)*np.pi/180  #弧度
# theta = theta + np.zeros((data.shape[0], len(theta)))
# theta = theta[:, :30:10]

# # Thomsen 横向各向同性
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, 
#                             data["Epsilon"], data["Delta"], data["Gamma"])
# lnVp, lnVs, lnVd, coef1, coef2, coef3, rpp = vti_appr.ani_zhang(0.3)
# vti_appr = forward.Thomsen(data_s["Vp"], data_s["Vs"], data_s["Rho"], theta, 
#                             data_s["Epsilon"], data_s["Delta"], data_s["Gamma"])
# lnVp_init, lnVs_init, lnVd_init, coef1, coef2, coef3, rpp = vti_appr.ani_zhang(0.3)
# reflect_7 = vti_appr.iso_approximate()
# reflect_7[np.isnan(reflect_7)] = 0 # 原始真实模型（目标）

# # 合成记录，子波矩阵
# W1 = convmtx(ricker, data.shape[0])[:-1, :-1] # 子波矩阵
# W = np.kron(np.eye(3), W1) # 子波矩阵（块3@3）
# ###########################################################################################################
# data.fillna(0)
# lnVp = lnVp.reshape(-1, 1)
# lnVs = lnVs.reshape(-1, 1)
# lnRho = lnVd.reshape(-1, 1)
# lnVp_init = lnVp_init.reshape(-1, 1)
# lnVs_init = lnVs_init.reshape(-1, 1)
# lnRho_init = lnVd_init.reshape(-1, 1)

# G = [] # 系数矩阵
# for i in range(3):
#     temp = transform_G(coef1[:, i], coef2[:, i], coef3[:, i]) # 模型参数
#     G.append(temp)
# G = np.vstack((G[0],G[1],G[2]))

# m_init = transform_m(lnVp_init, lnVs_init, lnRho_init) # 初始模型参数
# m_real = transform_m(lnVp, lnVs, lnRho) # 真实型参数
# m_init0 = copy.copy(m_init)

# # D = convmtx(np.array([0, -1, 1]), 3*(data.shape[0]-1)) # 微分算子建立
# D = convmtx(np.array([0, -1, 1]), (data.shape[0]-1)) # 微分算子建立
# D[data.shape[0]-2, :] = 0
# D = np.kron(np.ones((3, 3)), D)

# syn4 = (W1@reflect_7).reshape((-1, 1), order="F") # 原始真实模型地震记录
# syn2 = (W@G@D@m_real)
# syn3 = (W@G@D@m_init) # 初始模型地震记录

# plt.ion()
# fig2 = plt.figure()

# err_ls=[]
# epoch=[]
# _, I_lam = cov_mat(lnVp, lnVs, lnRho, 0.0001)
# m = copy.copy(m_init)
# d = copy.copy(syn4)
# F = W@G@D
# r = d-F@m_init
# s = np.zeros_like(d)
# beta = 0
# for i in range(100):
#     g = F.conj().T@r
#     if i!=0:
#         beta = np.vdot(g, g)/gamma
#     gamma = np.vdot(g, g)
#     s = g+beta*s
#     delta_r = F@s
#     alpha = -np.vdot(r, delta_r)/np.vdot(delta_r, delta_r)
#     m = m-alpha*s
#     r = r+alpha*delta_r
#     error = np.sqrt(np.vdot(r, r))
#     err_ls.append(error)
#     epoch.append(i)   
#     print("steps={}, error={}".format(i, error))
#     plt.cla()
#     plt.plot(epoch, err_ls)
#     plt.xlabel("$epoch$")
#     plt.ylabel("$error$")
#     plt.pause(0.01)
#     if err_ls[i]>err_ls[i-1]:
#         break
# plt.ioff()
# m_init = m
#############################################################################

syn = (W@G@D@m_init) # inversion results

# # 绘图
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(syn, color="red", label="inversion") # inversion
ax1.set_xticks([])
ax2 = fig.add_subplot(312)
ax2.plot(syn4, color="blue", label="real") # real
ax2.set_xticks([])
ax3 = fig.add_subplot(313)
ax3.plot(syn3, color="orange", label="initial") # initial
# plt.show()
# 绘图
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(m_real, color="red", label="inversion") # inversion
ax1.set_xticks([])
ax2 = fig.add_subplot(312)
ax2.plot(m_init, color="blue", label="real") # real
ax2.set_xticks([])
ax3 = fig.add_subplot(313)
ax3.plot(m_init0, color="orange", label="initial") # initial
# plt.show()
# 绘图
fig = plt.figure()
plt.subplot(111)
plt.plot(m_real, color="red", label="inversion") # inversion
plt.plot(m_init, color="blue", label="real") # real
plt.plot(m_init0, color="orange", label="initial") # initial
plt.show()

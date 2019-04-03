# -@- coding: utf-8 -@-
"""
Created on Mon Sep 24 12:13:47 2018

@author: 泳浩
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as linalg
import scipy.signal as signal
import sys
import time
import segyio


##########################################################################################
# 参数修改在此处
# top, bottom, trace_beg, trace_end = 1500, 1850, 0, 2191
# top, bottom, trace_beg, trace_end = 1300, 1900, 425, 426
top, bottom, trace_beg, trace_end = 1300, 1900, 1695, 1696
method = 0 # method{0:"conjugate", 1:"gauss_newton"}
cov_lambda_method = 1 # method{0:"const_number", 1:"3well_data", 2:"3initial_data", 3:"3other_data"}
print_process = 0 # method{0:"don't display", 1:"display"}
sigma = np.array([1, 1, 1])*0.0001*0
iso_k_given = 0 # 为固定值时情况(取值不为0代表使用背景速度)
gaussian_well = 50 # 高斯平滑时窗（背景速度平滑）
standard = 20 # 高斯分布标准差
trc = 0
step = 1 # 迭代步长
niter = 50 # iterative numbers
normal_method = 0 # 0表示不地震记录与子波标准化，1表示标准化
ratio = 0.035952 # 真实记录振幅于子波振幅大小之比
data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_025.csv"))
del data["Unnamed: 0"]

###########################################################################################
# 反演的角度
# theta = np.array([6, 14, 23])*np.pi/180 # 因为角度有三个
# theta = np.array([10, 20, 30, 40, 45])*np.pi/180 # 因为角度有五个
# theta = np.array([5, 14, 23, 32, 41])*np.pi/180 # 因为角度有五个
theta = np.array([5, 12, 19, 26, 33])*np.pi/180 # 因为角度有五个

##########################################################################################
print("Loading Data...") # L1

#　传入真实地震数据 0
# with segyio.open("D:\\Physic_Model_Data\\cdp_stack0.sgy", "r", ignore_geometry=False) as f:
#     real_0 = np.array([i for i in f.trace[:]]).T

# with segyio.open("D:\\Physic_Model_Data\\cdp_stack1.sgy", "r", ignore_geometry=False) as f:
#     real_1 = np.array([i for i in f.trace[:]]).T

# with segyio.open("D:\\Physic_Model_Data\\cdp_stack2.sgy", "r", ignore_geometry=False) as f:
#     real_2 = np.array([i for i in f.trace[:]]).T

# # # 传入真实地震数据 0
# real_0_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack10.sgy")[0,:,:].T
# real_1_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack20.sgy")[0,:,:].T
# real_2_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack30.sgy")[0,:,:].T
# real_3_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack40.sgy")[0,:,:].T
# real_4_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack45.sgy")[0,:,:].T

# # # 传入真实地震数据 1
# real_0_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack5.sgy")[0,:,:].T
# real_1_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack14.sgy")[0,:,:].T
# real_2_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack23.sgy")[0,:,:].T
# real_3_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack32.sgy")[0,:,:].T
# real_4_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack41.sgy")[0,:,:].T

# # 传入真实地震数据 0
real_0_ori = segyio.tools.cube("D:\\Physic_Model_Data\\01-08.sgy")[0,:,:].T
real_1_ori = segyio.tools.cube("D:\\Physic_Model_Data\\08-15.sgy")[0,:,:].T
real_2_ori = segyio.tools.cube("D:\\Physic_Model_Data\\15-22.sgy")[0,:,:].T
real_3_ori = segyio.tools.cube("D:\\Physic_Model_Data\\22-29.sgy")[0,:,:].T
real_4_ori = segyio.tools.cube("D:\\Physic_Model_Data\\29-36.sgy")[0,:,:].T

# # 传入真实地震数据 2
# real_0_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L_1.prj\\seismic.dir\\L1_cdp_stack_02_10.sgy")[0,:,:].T
# real_1_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L_1.prj\\seismic.dir\\L1_cdp_stack_10_19.sgy")[0,:,:].T
# real_2_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L_1.prj\\seismic.dir\\L1_cdp_stack_19_27.sgy")[0,:,:].T
# real_3_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L_1.prj\\seismic.dir\\L1_cdp_stack_27_36.sgy")[0,:,:].T
# real_4_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L_1.prj\\seismic.dir\\L1_cdp_stack_36_44.sgy")[0,:,:].T

# 传入初始模型数据 0
# with segyio.open("D:\\Physic_Model_Data\\geo3line1_vp.sgy", "r", ignore_geometry=True) as f:
#     model_0_ori = np.array([i for i in f.trace[:]]).T

# with segyio.open("D:\\Physic_Model_Data\\geo3line1_vs.sgy", "r", ignore_geometry=True) as f:
#     model_1_ori = np.array([i for i in f.trace[:]]).T

# with segyio.open("D:\\Physic_Model_Data\\geo3line1_Rho.sgy", "r", ignore_geometry=True) as f:
#     model_2_ori = np.array([i for i in f.trace[:]]).T

# 传入初始模型数据 1
# with segyio.open("D:\\Physic_Model_Data\\model_P-wave.sgy", "r", ignore_geometry=True) as f:
#     model_0_ori = np.array([i for i in f.trace[:]]).T

# with segyio.open("D:\\Physic_Model_Data\\model_S-wave.sgy", "r", ignore_geometry=True) as f:
#     model_1_ori = np.array([i for i in f.trace[:]]).T

# with segyio.open("D:\\Physic_Model_Data\\model_Density.sgy", "r", ignore_geometry=True) as f:
#     model_2_ori = np.array([i for i in f.trace[:]]).T

# # 传入初始模型数据 2
# model_0_ori = np.load("D:\\Physic_Model_Data\\line1_Vp_exactly.npy")

# model_1_ori = np.load("D:\\Physic_Model_Data\\line1_Vs_exactly.npy")

# model_2_ori = np.load("D:\\Physic_Model_Data\\line1_Rho_exactly.npy")

# model_3_ori = np.load("D:\\Physic_Model_Data\\line1_Epsilon_exactly.npy")

# model_4_ori = np.load("D:\\Physic_Model_Data\\line1_Delta_exactly.npy")

# model_5_ori = np.load("D:\\Physic_Model_Data\\line1_Gamma_exactly.npy")

# # 传入初始模型数据 3
model_0_ori = np.load("D:\\Physic_Model_Data\\line1_Vp.npy")

model_1_ori = np.load("D:\\Physic_Model_Data\\line1_Vs.npy")

model_2_ori = np.load("D:\\Physic_Model_Data\\line1_Rho.npy")

# model_3_ori = np.load("D:\\Physic_Model_Data\\line1_Epsilon.npy")

# model_4_ori = np.load("D:\\Physic_Model_Data\\line1_Delta.npy")

# model_5_ori = np.load("D:\\Physic_Model_Data\\line1_Gamma.npy")

# # 传入不同角度子波数据
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\statistical_target_02_10.txt", skiprows=31)

# wave_1 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\statistical_target_10_19.txt", skiprows=31)

# wave_2 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\statistical_target_19_27.txt", skiprows=31)

# wave_3 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\statistical_target_27_36.txt", skiprows=31)

# wave_4 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\statistical_target_36_44.txt", skiprows=31)

# 传入不同角度子波数据
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_02_10.txt", skiprows=56)

# wave_1 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_10_19.txt", skiprows=56)

# wave_2 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_19_27.txt", skiprows=56)

# wave_3 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_27_36.txt", skiprows=56)

# wave_4 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_36_44.txt", skiprows=56)

# # 传入不同角度子波数据
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\wave0.txt", skiprows=42)

# wave_1 = np.loadtxt("D:\\Physic_Model_Data\\wave1.txt", skiprows=42)

# wave_2 = np.loadtxt("D:\\Physic_Model_Data\\wave2.txt", skiprows=42)

# wave_3 = np.loadtxt("D:\\Physic_Model_Data\\wave3.txt", skiprows=42)

# wave_4 = np.loadtxt("D:\\Physic_Model_Data\\wave4.txt", skiprows=42)

# # # 传入不同角度子波数据
wave_0 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_01_08.txt", skiprows=56)
wave_1 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_08_15.txt", skiprows=56)
wave_2 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_15_22.txt", skiprows=56)
wave_3 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_22_29.txt", skiprows=56)
wave_4 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_29_36.txt", skiprows=56)

# 读取测井数据
# data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line2_065.csv"))
# del data["Unnamed: 0"]
# data = data.rolling(15).mean()

###########################################################################################
# print("Loading Data...") # L2

# # # # 传入真实地震数据 0
# real_0_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack10.sgy")[0,:,:].T
# real_1_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack20.sgy")[0,:,:].T
# real_2_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack30.sgy")[0,:,:].T
# real_3_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack40.sgy")[0,:,:].T
# real_4_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack45.sgy")[0,:,:].T

# # # 传入初始模型数据 1
# # model_0_ori = np.load("D:\\Physic_Model_Data\\line2_Vp_exactly.npy")

# # model_1_ori = np.load("D:\\Physic_Model_Data\\line2_Vs_exactly.npy")

# # model_2_ori = np.load("D:\\Physic_Model_Data\\line2_Rho_exactly.npy")

# # model_3_ori = np.load("D:\\Physic_Model_Data\\line2_Epsilon_exactly.npy")

# # model_4_ori = np.load("D:\\Physic_Model_Data\\line2_Delta_exactly.npy")

# # model_5_ori = np.load("D:\\Physic_Model_Data\\line2_Gamma_exactly.npy")

# # # 传入初始模型数据 3
# model_0_ori = np.load("D:\\Physic_Model_Data\\line2_Vp.npy")

# model_1_ori = np.load("D:\\Physic_Model_Data\\line2_Vs.npy")

# model_2_ori = np.load("D:\\Physic_Model_Data\\line2_Rho.npy")

# # model_3_ori = np.load("D:\\Physic_Model_Data\\line2_Epsilon.npy")

# # model_4_ori = np.load("D:\\Physic_Model_Data\\line2_Delta.npy")

# # model_5_ori = np.load("D:\\Physic_Model_Data\\line2_Gamma.npy")

# # 传入不同角度子波数据
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells10.txt", skiprows=138)

# wave_1 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells20.txt", skiprows=138)

# wave_2 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells30.txt", skiprows=138)

# wave_3 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells40.txt", skiprows=138)

# wave_4 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells45.txt", skiprows=53)

# # 读取测井数据
# # data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line2_065.csv"))
# # del data["Unnamed: 0"]
# # data = data.rolling(15).mean()

##########################################################################################
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

# def cov_mat(sigma):
#     '''
#     协方差矩阵，传入的必须为列表
#     '''
#     lam = np.array(sigma)*np.ones((3, 1))
#     I_lam = np.kron(lam, np.eye(background.shape[0]))
#     return I_lam

def cov_mat(sigma):
    '''
    协方差矩阵，传入的必须为列表
    '''
    lam = np.diag(sigma)
    I_lam = np.kron(lam, np.eye(background.shape[0]))
    return I_lam

def cov_mat1(para_1, para_2, para_3, sigma_2):
    '''
    协方差矩阵，传入的必须为数组或者向量，不能是Series，此为第一种正则化系数搭配
    传入的sigma_2为常数
    '''
    Cov = np.cov(np.hstack((para_1, para_2, para_3)).reshape(3, -1)) # 3*3 matrix
    lam = sigma_2@linalg.inv(Cov)
    I_lam = np.kron(lam, np.eye(background.shape[0]))
    return [Cov, I_lam]

def cov_mat2(para_1, para_2, para_3, sigma_2):
    '''
    协方差矩阵，传入的必须为数组或者向量，不能是Series，此为第二种正则化系数搭配
    传入的sigma_2为对角矩阵 3*3
    '''
    sigma_2 = np.diag(sigma_2) # 3*3 diag matrix
    Cov = np.cov(np.hstack((para_1, para_2, para_3)).reshape(3, -1)) # 3*3 matrix
    lam = sigma_2@linalg.inv(Cov)
    I_lam = np.kron(lam, np.eye(background.shape[0]))
    return [Cov, I_lam]

def cov_mat3(sigma):
    '''
    协方差矩阵，传入的必须为3@3矩阵
    '''
    I_lam = np.kron(sigma, np.eye(background.shape[0]))
    return I_lam

def cov_mat4(sigma, cov_mat):
    '''
    协方差矩阵，传入的必须为列表和3@3矩阵
    '''
    lam = np.diag(sigma)
    I_lam = np.kron(lam@linalg.inv(cov_mat), np.eye(background.shape[0]))
    return I_lam

###########################################################################################
# 反演范围
# top, bottom, trace_beg, trace_end = 1500, 1850, 200, 1900
# top, bottom, trace_beg, trace_end = 985, 1901, 200, 1900
real_0, real_1, real_2 = real_0_ori[top:bottom, trace_beg:trace_end], real_1_ori[top:bottom, trace_beg:trace_end], real_2_ori[top:bottom, trace_beg:trace_end]
real_3, real_4 = real_3_ori[top:bottom, trace_beg:trace_end], real_4_ori[top:bottom, trace_beg:trace_end]
model_0, model_1, model_2 = model_0_ori[top:bottom, trace_beg:trace_end], model_1_ori[top:bottom, trace_beg:trace_end], model_2_ori[top:bottom, trace_beg:trace_end]
# model_3, model_4 = model_3_ori[top:bottom, trace_beg:trace_end], model_4_ori[top:bottom, trace_beg:trace_end]

# 井数据范围
data = data.iloc[top:bottom, :]
dx, dy = data.shape

t1 = np.random.normal(0, 1, dx)
t2 = np.random.normal(0, 1, dx)
t3 = np.random.normal(0, 1, dx)
data["Vp"] = data["Vp"]+t1*1
data["Vs"] = data["Vs"]+t2*1
data["Rho"] = data["Rho"]+t3*0.01

# 井测得真实值 m/s->km/s
well_real = np.hstack((np.log(data["Vp"]/1000), np.log(data["Vs"]/1000), np.log(data["Rho"])))

############################################################################################
# 反演单个参数的图像背景大小
background = np.zeros((real_0.shape)) 

# # 地震记录振幅标准化
# real_0 = real_0/abs(real_0).max()
# real_1 = real_1/abs(real_1).max()
# real_2 = real_2/abs(real_2).max()
# real_3 = real_3/abs(real_3).max()
# real_4 = real_4/abs(real_4).max()

# # 建立不同角度的真实地震数据矩阵 三个角度
# real = np.vstack((real_0, real_1, real_2))

# # 建立不同角度的真实地震数据矩阵 五个角度
real = np.vstack((real_0[:-1], real_1[:-1], real_2[:-1], real_3[:-1], real_4[:-1]))

# # 地震记录振幅标准化
if normal_method == 1:
    real = ratio*real/abs(real).max()
    
# 子波振幅标准化
# wave_0 = wave_0/abs(wave_0).max()*4.4
# wave_1 = wave_1/abs(wave_1).max()*3
# wave_2 = wave_2/abs(wave_2).max()*4.2
# wave_3 = wave_3/abs(wave_3).max()*7
# wave_4 = wave_4/abs(wave_4).max()*23

# 子波矩阵不同角度
W0 = convmtx(wave_0, background.shape[0]-1)
W1 = convmtx(wave_1, background.shape[0]-1)
W2 = convmtx(wave_2, background.shape[0]-1)
W3 = convmtx(wave_3, background.shape[0]-1)
W4 = convmtx(wave_4, background.shape[0]-1)
W_zero = np.zeros_like(W0)

# # 不同角度子波组成矩阵（块3@3）
# W0 = np.hstack((W0, W_zero, W_zero))
# W1 = np.hstack((W_zero, W1, W_zero))
# W2 = np.hstack((W_zero, W_zero, W2))
# W = np.vstack((W0, W1, W2))

# 不同角度子波组成矩阵（块5@5）
W0 = np.hstack((W0, W_zero, W_zero, W_zero, W_zero))
W1 = np.hstack((W_zero, W1, W_zero, W_zero, W_zero))
W2 = np.hstack((W_zero, W_zero, W2, W_zero, W_zero))
W3 = np.hstack((W_zero, W_zero, W_zero, W3, W_zero))
W4 = np.hstack((W_zero, W_zero, W_zero, W_zero, W4))
W = np.vstack((W0, W1, W2, W3, W4))

# 子波振幅标准化
if normal_method == 1:
    W = W/abs(W).max()

# 建立微分算子矩阵
D = convmtx(np.array([0, -1, 1]), (background.shape[0]))
D = D[:-1, :]
D = np.kron(np.eye(3), D)

# 建立初始模型数据矩阵（平滑过的）
model = np.vstack((model_0, model_1, model_2))

# 建立初始模型数据矩阵（未平滑最精确的）
model_ori = np.vstack((model_0_ori[top:bottom, trace_beg:trace_end], model_1_ori[top:bottom, trace_beg:trace_end], model_2_ori[top:bottom, trace_beg:trace_end]))

# lambda正则化系数(sigma^2)
# sigma = 50 # 法一
# sigma2 = [6, 6, 6] # 法二 个人认为法一正确

# 纵波横波密度的协方差矩阵与正则化大矩阵
if cov_lambda_method == 0:
    cov, I_lam = cov_mat1(np.log(data["Vp"]/1000), np.log(data["Vs"]/1000), np.log(data["Rho"]), sigma=1) # 法一
elif cov_lambda_method == 1:
    cov, I_lam = cov_mat2(np.log(data["Vp"]/1000), np.log(data["Vs"]/1000), np.log(data["Rho"]), sigma) # 法二 个人认为法一正确
elif cov_lambda_method == 2:
    I_lam = cov_mat4(sigma, cov_mat_iso)
elif cov_lambda_method == 3:
    I_lam = cov_mat(sigma)
else:
    print("wrong prameters!")
    time.sleep(3)
    sys.exit(0)
    
##########################################################################################################
# Thomsen 横向各向同性（得到A,B,C系数也就是coef1,coef2,coef3） 三个角度（共轭梯度求解用此）
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, 
#                             data["Epsilon"], data["Delta"], data["Gamma"])
# k = (data["Vs"].mean()/data["Vp"].mean())
# _, _, _, coef1, coef2, coef3, _ = vti_appr.iso_ani_paras(k)
# temp = np.ones((background.shape[0], 1))
# coef1, coef2, coef3 = coef1[0, :]*temp, coef2[0, :]*temp, coef3[0, :]*temp

# Thomsen 横向各向同性（得到A,B,C系数也就是coef1,coef2,coef3） 五个角度（贝叶斯用此）
if iso_k_given==0:
    iso_k = (data["Vs"].mean())/(data["Vp"].mean())
elif iso_k_given==1:
    iso_k = (np.array(((data["Vs"])/(data["Vp"])))[:-1]).reshape(-1, 1)
else:
    iso_k = np.array(((data["Vs"])/(data["Vp"])))[:-1]
    gaussian_window = signal.gaussian(gaussian_well, std=standard)
    iso_k = (np.convolve(iso_k, gaussian_window, "same")/gaussian_window.sum()).reshape(-1, 1) # background velocity
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(iso_k, np.arange(len(iso_k)))
    ax1.invert_yaxis()
    ax2 = fig.add_subplot(122)
    ax2.plot(gaussian_window)
    plt.show()
coef1, coef2, coef3 = (1+np.tan(theta)**2)/2, -8*iso_k**2*np.sin(theta)**2/2, (1-4*iso_k**2*np.sin(theta)**2)/2
temp = np.ones((background.shape[0]-1, 1))
coef1, coef2, coef3 = coef1*temp, coef2*temp, coef3*temp

# # 建立系数矩阵（系数A,B,C的）三个角度
# G = []
# for i in range(3):
#     temp = transform_G(coef1[:, i], coef2[:, i], coef3[:, i]) # 模型参数
#     G.append(temp)
# G = np.vstack((G[0],G[1],G[2]))

# 建立系数矩阵（系数A,B,C的）五个角度
G = []
for i in range(5):
    temp = transform_G(coef1[:, i], coef2[:, i], coef3[:, i]) # 模型参数
    G.append(temp)
G = np.vstack((G[0],G[1],G[2],G[3],G[4]))

print("Start Inversion... ")
##############################################################################################################
if method == 0:
    # 反演(1 共轭梯度下降)
    m_init = np.log(model) # 初始模型取对数后的参数

    # 反演所有参数的图像背景大小(3参数)
    background = np.vstack((background, background, background)) 

    # 迭代的剖面总道数
    total = background.shape[1]

    errorlist=[]
    # 反演主程序 y=Gx
    F_inner = W@G@D
    F = F_inner.T@F_inner+I_lam #G
    m_init = m_init
    real = F_inner.T@real
    F_H = F.conj().T
    for progress in range(total):
        m = m_init[:, progress] # 模型记录
        d = real[:, progress] # 真实地震记录
        r = d-F@m #x 
        s = np.zeros_like(d)
        beta = 0
        for i in range(niter):
            g = F_H@r
            if i!=0:
                beta = np.vdot(g, g)/gamma
            gamma = np.vdot(g, g)
            s = g+beta*s
            delta_r = F@s
            alpha = -np.vdot(r, delta_r)/np.vdot(delta_r, delta_r)
            m = m-alpha*s
            r = r+alpha*delta_r
            error = np.sqrt(np.vdot(r, r))
            errorlist.append(error)
            # print("total progress={:2.2f}%, steps={:2d}, error={:.4f}".format(progress*100/total, i, error))
        background[:, progress] = m.flatten()
        print("total progress={:2.2f}%".format(progress*100/total))
#############################################################################################################

##############################################################################################################
# elif method == 1:
#     # 反演(1 共轭梯度下降)
#     m_init = np.log(model) # 初始模型取对数后的参数

#     # 反演所有参数的图像背景大小(3参数)
#     background = np.vstack((background, background, background)) 

#     # 迭代的剖面总道数
#     total = background.shape[1]

#     # 反演主程序
#     F = W@G@D
#     F_H = F.conj().T
#     for progress in range(total):
#         m = m_init[:, progress] # 模型记录
#         d = real[:, progress] # 真实地震记录
#         r = d-F@m
#         s = np.zeros_like(d)
#         beta = 0
#         for i in range(niter):
#             g = F_H@r
#             if i!=0:
#                 beta = np.vdot(g, g)/gamma
#             gamma = np.vdot(g, g)
#             s = g+beta*s
#             delta_r = F@s
#             alpha = -np.vdot(r, delta_r)/np.vdot(delta_r, delta_r)
#             m = m-alpha*s
#             r = r+alpha*delta_r
#             error = np.sqrt(np.vdot(r, r))
#             print("total progress={:2.2f}%, steps={:2d}, error={:.4f}".format(progress*100/total, i, error))
#         background[:, progress] = m.flatten()
#         # print("total progress={:2.2f}%".format(progress*100/total))
#############################################################################################################

#############################################################################################################
elif method == 1:
    # 反演(2 高斯牛顿迭代)
    m_init = np.log(model) # 初始模型取对数后的参数

    # # 迭代步长
    # step = 1

    # 反演所有参数的图像背景大小(3参数)
    background = np.vstack((background, background, background)) 

    # 迭代的剖面总道数
    total = background.shape[1]

    # 反演主程序
    F = W@G@D
    F_T = F.T
    temp = F_T@F+I_lam
    for progress in range(total):
        m = m_init[:, progress] # 模型记录
        d = real[:, progress] # 真实地震记录
        for i in range(niter):
            r = d-F@m # 差用褶积记录
            m = m+linalg.solve(temp, F_T@r)*step
            # m = m+linalg.inv(temp)@F_T@r*step
            # m = m+linalg.inv(F_T@F+I_lam)@F_T@r*step
            # m = m+linalg.inv(F_T@F+I_lam2)@F_T@r*step
            error = np.sqrt(np.vdot(r, r))
            print("total progress={:2.2f}%, steps={:2d}, error={:.4f}".format(progress*100/total, i, error))
        background[:, progress] = m.flatten()
        print("total progress={:2.2f}%".format(progress*100/total))   

else:
    print("wrong prameters!")
    time.sleep(3)
    sys.exit(0)
    
#############################################################################################################
# 反演的垂向采样点数
nsamples = np.arange(top, bottom, 1)

# 研究第几道的结果
# trc = 0

# 矩阵拆分为三部分
nmat = int(background.shape[0]/3)

# 保存反演结果
inv1 = np.exp(background[0:nmat, :])
inv2 = np.exp(background[nmat:2*nmat, :])
inv3 = np.exp(background[2*nmat:3*nmat, :])
np.save("D:\\Physic_Model_Data\\iso_inv1", inv1)
np.save("D:\\Physic_Model_Data\\iso_inv2", inv2)
np.save("D:\\Physic_Model_Data\\iso_inv3", inv3)
sigma = ((inv1/inv2)**2-2)/((inv1/inv2)**2-1)/2 # poisson
np.save("D:\\Physic_Model_Data\\iso_sigma.npy", sigma)
yang = 2*inv3*(1+sigma)*inv2**2 # yang
np.save("D:\\Physic_Model_Data\\iso_yang.npy", yang)
zp = inv1*inv3 # Zp
np.save("D:\\Physic_Model_Data\\iso_zp.npy", zp)
zs = inv2*inv3 # Zs
np.save("D:\\Physic_Model_Data\\iso_zs.npy", zs)

#############################################################################################################
# 绘图 各向同性
fig = plt.figure()
ax0a = fig.add_subplot(131)
# l_vp_well, = ax0a.plot(np.exp(well_real[0:nmat]), nsamples, 'k', lw=1, label="well_data")
l_vp_well, = ax0a.plot(data["Vp"]/1000, nsamples, 'k', lw=1, label="well_data")
l_vp_inv, = ax0a.plot(np.exp(background[0:nmat, trc]), nsamples, 'k', color="red", lw=1, label="inversion")
l_vp_init, = ax0a.plot(model[0:nmat, trc], nsamples, 'k', color="blue", lw=1, label="initial_model") #注意单位km/s
ax0a.set_ylim((top,bottom))
# ax0a.set_xlim(2.4, 3.2)
ax0a.invert_yaxis()
ax0a.set_ylabel('Samples')
ax0a.xaxis.tick_top()
ax0a.xaxis.set_label_position('top')
ax0a.set_xlabel("Vp (km/s)")
plt.legend()
ax0a.grid()

ax0b = fig.add_subplot(132)
# l_vs_well, = ax0b.plot(np.exp(well_real[nmat:2*nmat]), nsamples, 'k', lw=1, label="well_data")
l_vs_well, = ax0b.plot(data["Vs"]/1000, nsamples, 'k', lw=1, label="well_data")
l_vs_inv, = ax0b.plot(np.exp(background[nmat:2*nmat, trc]), nsamples, 'k', color="red", lw=1, label="inversion")
l_vs_init, = ax0b.plot(model[nmat:2*nmat, trc], nsamples, 'k', color="blue", lw=1, label="initial_model") #注意单位km/s
ax0b.set_ylim((top,bottom))
# ax0b.set_xlim((1.0, 2.5))
ax0b.invert_yaxis()
ax0b.xaxis.tick_top()
ax0b.xaxis.set_label_position('top')
ax0b.set_xlabel("Vs (km/s)")
ax0b.set_yticklabels('')
plt.legend()
ax0b.grid()

ax0c = fig.add_subplot(133)
# l_rho_well, = ax0c.plot(np.exp(well_real[2*nmat:3*nmat]), nsamples, 'k', lw=1, label="well_data")
l_rho_well, = ax0c.plot(data["Rho"], nsamples, 'k', lw=1, label="well_data")
l_rho_inv, = ax0c.plot(np.exp(background[2*nmat:3*nmat, trc]), nsamples, 'k', color="red", lw=1, label="inversion")
l_rho_init, = ax0c.plot(model[2*nmat:3*nmat, trc], nsamples, 'k', color="blue", lw=1, label="initial_model")
ax0c.set_ylim((top,bottom))
# ax0c.set_xlim((0.5, 3.0))
ax0c.invert_yaxis()
ax0c.xaxis.tick_top()
ax0c.xaxis.set_label_position('top')
ax0c.set_xlabel("Den (g/cm^3)")
ax0c.set_yticklabels('')
ax0c.grid()
plt.legend()
plt.show()

#############################################################################################################
# # 绘图
# plt.figure()
# # plt.set_cmap("cubehelix_r")
# background = background[0:nmat]
# vmin = np.percentile(background[background>0], 1)
# vmax = np.percentile(background[background>0], 99)
# print("min={}, max={}".format(vmin, vmax))
# plt.imshow(background, aspect=0.3, vmin=vmin, vmax=vmax)
# plt.colorbar(shrink=0.75)
# plt.show()

# plt.figure()
# # plt.set_cmap("cubehelix_r")
# model = model[0:nmat]
# vmin = np.percentile(model[model>0], 1)
# vmax = np.percentile(model[model>0], 99)
# print("min={}, max={}".format(vmin, vmax))
# plt.imshow(model, aspect=0.3, vmin=vmin, vmax=vmax)
# plt.colorbar(shrink=0.75)
# plt.show()

# #############################################################################################################
# # 绘图
# plt.figure()
# plt.plot(errorlist)
# plt.show()
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
# top, bottom, trace_beg, trace_end = 1300, 1885, 0, 2191
# top, bottom, trace_beg, trace_end = 1300, 1885, 425, 426
top, bottom, trace_beg, trace_end = 1300, 1885, 1695, 1696
method = 1 # method{0:"conjugate", 1:"gauss_newton"}
cov_lambda_method = 3 # method{0:"const_number", 1:"3well_data", 2:"3initial_data", 3:"3other_data"}
cov_mat_ani = np.array([[7.6455076e-03, 7.3234964e-03, 5.0318266e-05],
                        [7.3234964e-03, 8.2397042e-03, 1.5111740e-04],
                        [5.0318266e-05, 1.5111740e-04, 1.0689176e-03]])
print_process = 0 # method{0:"don't display", 1:"display"}
sigma = np.array([0.006, 0.006, 0.0002])
ani_k_given = 0 # 为固定值时情况(取值不为0代表使用背景速度)
gaussian_well = 50 # 高斯平滑时窗（背景速度平滑）
standard = 5 # 高斯分布标准差
trc = 0
step = 1 # 迭代步长
niter = 10 # iterative numbers
normal_method = 0 # 0表示不地震记录与子波标准化，1表示标准化
ratio = 0.031088 # 真实记录振幅于子波振幅大小之比
w_amp = 1 # 1为子波正常子波，-1为子波反转
data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_025.csv"))
# data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_05.csv"))
del data["Unnamed: 0"]

###########################################################################################
# 反演的角度
# theta = np.array([6, 14, 23])*np.pi/180 # 因为角度有三个
# theta = np.array([10, 20, 30, 40, 45])*np.pi/180 # 因为角度有五个
# theta = np.array([5, 14, 23, 32, 41])*np.pi/180 # 因为角度有五个
theta = np.array([5, 12, 19, 26, 33])*np.pi/180 # 因为角度有五个

##########################################################################################
print("Loading Data...") # L1

# # 传入真实地震数据 0
real_0_ori = segyio.tools.cube("D:\\Physic_Model_Data\\stack01-08.sgy")[0,:,:].T
real_1_ori = segyio.tools.cube("D:\\Physic_Model_Data\\stack08-15.sgy")[0,:,:].T
real_2_ori = segyio.tools.cube("D:\\Physic_Model_Data\\stack15-22.sgy")[0,:,:].T
real_3_ori = segyio.tools.cube("D:\\Physic_Model_Data\\stack22-29.sgy")[0,:,:].T
real_4_ori = segyio.tools.cube("D:\\Physic_Model_Data\\stack29-36.sgy")[0,:,:].T

# # # 传入真实地震数据 0
# real_0_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack10.sgy")[0,:,:].T
# real_1_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack20.sgy")[0,:,:].T
# real_2_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack30.sgy")[0,:,:].T
# real_3_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack40.sgy")[0,:,:].T
# real_4_ori = segyio.tools.cube("D:\\Physic_Model_Data\\cdp_stack45.sgy")[0,:,:].T

# # 传入初始模型数据 0 各向异性
model_0_ori = np.load("D:\\Physic_Model_Data\\line1_Vp.npy")
model_1_ori = np.load("D:\\Physic_Model_Data\\line1_Vs.npy")
model_2_ori = np.load("D:\\Physic_Model_Data\\line1_Rho.npy")

# # # 传入不同角度子波数据
wave_0 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_01_08.txt", skiprows=56)
wave_1 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_08_15.txt", skiprows=56)
wave_2 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_15_22.txt", skiprows=56)
wave_3 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_22_29.txt", skiprows=56)
wave_4 = np.loadtxt("F:\\Physic_Model_Data\\0_New Folder\\0_29_36.txt", skiprows=56)

# # 传入不同角度子波数据
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\1_01_08.txt", skiprows=56)
# wave_1 = np.loadtxt("D:\\Physic_Model_Data\\1_08_15.txt", skiprows=56)
# wave_2 = np.loadtxt("D:\\Physic_Model_Data\\1_15_21.txt", skiprows=56)
# wave_3 = np.loadtxt("D:\\Physic_Model_Data\\1_21_28.txt", skiprows=56)
# wave_4 = np.loadtxt("D:\\Physic_Model_Data\\1_28_35.txt", skiprows=56)

# # # 传入不同角度子波数据
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_02_10.txt", skiprows=56)
# wave_1 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_10_19.txt", skiprows=56)
# wave_2 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_19_27.txt", skiprows=56)
# wave_3 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_27_36.txt", skiprows=56)
# wave_4 = np.loadtxt("D:\\Physic_Model_Data\\wavelet\\wavelet_use_wells_target_36_44.txt", skiprows=56)

# 传入不同角度子波数据
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\wave0.txt", skiprows=42)
# wave_1 = np.loadtxt("D:\\Physic_Model_Data\\wave1.txt", skiprows=42)
# wave_2 = np.loadtxt("D:\\Physic_Model_Data\\wave2.txt", skiprows=42)
# wave_3 = np.loadtxt("D:\\Physic_Model_Data\\wave3.txt", skiprows=42)
# wave_4 = np.loadtxt("D:\\Physic_Model_Data\\wave4.txt", skiprows=42)

###########################################################################################
# print("Loading Data...") # L2

# # # # 传入真实地震数据 0
# real_0_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack10.sgy")[0,:,:].T
# real_1_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack20.sgy")[0,:,:].T
# real_2_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack30.sgy")[0,:,:].T
# real_3_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack40.sgy")[0,:,:].T
# real_4_ori = segyio.tools.cube("D:\\Physic_Model_Data\\L2_cdp_stack45.sgy")[0,:,:].T

# # # 传入初始模型数据 0 各向异性
# model_0_ori = np.load("D:\\Physic_Model_Data\\L2_coe0_blur.npy")
# model_1_ori = np.load("D:\\Physic_Model_Data\\L2_coe1_blur.npy")
# model_2_ori = np.load("D:\\Physic_Model_Data\\L2_coe2_blur.npy")

# # 传入不同角度子波数据
# wave_0 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells10.txt", skiprows=138)
# wave_1 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells20.txt", skiprows=138)
# wave_2 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells30.txt", skiprows=138)
# wave_3 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells40.txt", skiprows=138)
# wave_4 = np.loadtxt("D:\\Physic_Model_Data\\wavelet_use_wells45.txt", skiprows=53)

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
    '''
    # Cov = np.cov(np.hstack((para_1, para_2, para_3)).T)
    Cov = np.cov(np.hstack((para_1, para_2, para_3)).reshape(3,-1))
    lam = np.diag(sigma_2)@linalg.inv(Cov)
    I_lam = np.kron(lam, np.eye(background.shape[0]))
    return [Cov, I_lam]

def cov_mat2(sigma, cov_mat):
    '''
    协方差矩阵，传入的必须为列表和3@3矩阵
    '''
    lam = np.diag(sigma)
    I_lam = np.kron(lam@linalg.inv(cov_mat), np.eye(background.shape[0]))
    return I_lam

###########################################################################################
# 反演范围
real_0, real_1, real_2 = real_0_ori[top:bottom, trace_beg:trace_end], real_1_ori[top:bottom, trace_beg:trace_end], real_2_ori[top:bottom, trace_beg:trace_end]
real_3, real_4 = real_3_ori[top:bottom, trace_beg:trace_end], real_4_ori[top:bottom, trace_beg:trace_end]
model_0, model_1, model_2 = model_0_ori[top:bottom, trace_beg:trace_end], model_1_ori[top:bottom, trace_beg:trace_end], model_2_ori[top:bottom, trace_beg:trace_end]

# 井数据范围
data = data.iloc[top:bottom, :]
dx, dy = data.shape

# 井测得真实值 m/s->km/s 各向异性
t1 = np.random.normal(0, 1, dx)*0
t2 = np.random.normal(0, 1, dx)*0
t3 = np.random.normal(0, 1, dx)*0
ani_Vp = data["Vp"]/1000+t1
ani_Vs = data["Vs"]/1000+t2
ani_rho = data["Rho"]+t3
ani_k = (ani_Vs/ani_Vp)
coe0 = ani_Vp
coe1 = ani_Vs
coe2 = ani_rho

############################################################################################
# 反演单个参数的图像背景大小
background = np.zeros((model_0.shape)) 

# # 建立不同角度的真实地震数据矩阵 五个角度
real = np.vstack((real_0[:-1], real_1[:-1], real_2[:-1], real_3[:-1], real_4[:-1]))

# # 地震记录振幅标准化
if normal_method == 1:
    real = ratio*real/abs(real).max()

# 子波矩阵不同角度
W0 = convmtx(wave_0, background.shape[0]-1)
W1 = convmtx(wave_1, background.shape[0]-1)
W2 = convmtx(wave_2, background.shape[0]-1)
W3 = convmtx(wave_3, background.shape[0]-1)
W4 = convmtx(wave_4, background.shape[0]-1)
W_zero = np.zeros_like(W0)

# 不同角度子波组成矩阵（块5@5）
W0 = np.hstack((W0, W_zero, W_zero, W_zero, W_zero))
W1 = np.hstack((W_zero, W1, W_zero, W_zero, W_zero))
W2 = np.hstack((W_zero, W_zero, W2, W_zero, W_zero))
W3 = np.hstack((W_zero, W_zero, W_zero, W3, W_zero))
W4 = np.hstack((W_zero, W_zero, W_zero, W_zero, W4))
W = np.vstack((W0, W1, W2, W3, W4))
if w_amp==-1:
    W = W*w_amp
else:
    pass
    
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

# 纵波横波密度的协方差矩阵与正则化大矩阵
if cov_lambda_method == 0:
    cov, I_lam = cov_mat1(np.log(coe0), np.log(coe1), np.log(coe2), sigma=1) # 法一
elif cov_lambda_method == 1:
    cov, I_lam = cov_mat1(np.log(coe0), np.log(coe1), np.log(coe2), sigma) # 法二
elif cov_lambda_method == 2:
    cov, I_lam = cov_mat1(np.log(model_0), np.log(model_1), np.log(model_2), sigma) # 法三
elif cov_lambda_method == 3:
    I_lam = cov_mat2(sigma, cov_mat_ani) # 法四
elif cov_lambda_method == 4:
    I_lam = cov_mat(sigma) # 法五
else:
    print("wrong parameters!")
    time.sleep(3)
    sys.exit(0)

##########################################################################################################
# Thomsen_zhang 横向各向同性 五个角度（贝叶斯用此）
if ani_k_given==0:
    ani_k = ani_k.mean()
elif ani_k_given==1:
    ani_k = (np.array(ani_k)).reshape(-1, 1)[:-1]
else:
    ani_k = np.array(ani_k)[:-1]
    gaussian_window = signal.gaussian(gaussian_well, std=standard)
    ani_k = (np.convolve(ani_k, gaussian_window, "same")/gaussian_window.sum()).reshape(-1, 1) # background velocity
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax1.plot(ani_k, np.arange(len(ani_k)))
    # ax1.invert_yaxis()
    # ax2 = fig.add_subplot(122)
    # ax2.plot(gaussian_window)
    # plt.show()
coef1, coef2, coef3 = (1+np.tan(theta)**2)/2, -8*iso_k**2*np.sin(theta)**2/2, (1-4*iso_k**2*np.sin(theta)**2)/2
temp = np.ones((background.shape[0]-1, 1))
coef1, coef2, coef3 = coef1*temp, coef2*temp, coef3*temp

# 建立系数矩阵（系数A,B,C的）五个角度
G = []
for i in range(5):
    temp = transform_G(coef1[:, i], coef2[:, i], coef3[:, i]) # 模型参数
    G.append(temp)
G = np.vstack((G[0],G[1],G[2],G[3],G[4]))

print("Start Inversion... ")

##############################################################################################################
# 反演(1 共轭梯度下降)
if method == 0:
    m_init = np.log(model) # 初始模型取对数后的参数

    # 反演所有参数的图像背景大小(3参数)
    background = np.vstack((background, background, background)) 

    # 迭代的剖面总道数
    total = background.shape[1]

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
            print("total progress={:2.2f}%, steps={:2d}, error={:.4f}".format(progress*100/total, i, error))
        background[:, progress] = m.flatten()
        print("total progress={:2.2f}%".format(progress*100/total))
#############################################################################################################

#############################################################################################################
# 反演(2 高斯牛顿迭代)
elif method == 1:
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
    temp_test = F_T@F
    temp = temp_test+I_lam
    for progress in range(total):
        m = m_init[:, progress] # 模型记录
        d = real[:, progress] # 真实地震记录
        for i in range(niter):
            r = d-F@m # 差用褶积记录
            # m = m+linalg.inv(temp)@(F_T@r)*step
            m = m+linalg.solve(temp, (F_T@r))*step
            error = np.sqrt(np.vdot(r, r))
            print("total progress={:2.2f}%, steps={:2d}, error={:.4f}".format(progress*100/total, i, error))
        background[:, progress] = m.flatten()
        print("total progress={:2.2f}%".format(progress*100/total))   

else:
    print("wrong parameters!")
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
np.save("D:\\Physic_Model_Data\\ani_inv1", np.exp(background[0:nmat, :]))
np.save("D:\\Physic_Model_Data\\ani_inv2", np.exp(background[nmat:2*nmat, :]))
np.save("D:\\Physic_Model_Data\\ani_inv3", np.exp(background[2*nmat:3*nmat, :]))

#############################################################################################################
# 绘图 各向异性
fig = plt.figure()
ax0a = fig.add_subplot(131)
l_vp_well, = ax0a.plot(coe0, nsamples, 'k', lw=1, label="well_data")
l_vp_inv, = ax0a.plot(np.exp(background[0:nmat, trc]), nsamples, 'k', color="red", lw=1, label="inversion")
l_vp_init, = ax0a.plot(model[0:nmat, trc], nsamples, 'k', color="blue", lw=1, label="initial_model") #注意单位km/s
ax0a.set_ylim((top,bottom))
# ax0a.set_xlim(2, 10)
ax0a.invert_yaxis()
ax0a.set_ylabel('Samples')
ax0a.xaxis.tick_top()
ax0a.xaxis.set_label_position('top')
ax0a.set_xlabel("AVO 0")
plt.legend()
ax0a.grid()

ax0b = fig.add_subplot(132)
l_vs_well, = ax0b.plot(coe1, nsamples, 'k', lw=1, label="well_data")
l_vs_inv, = ax0b.plot(np.exp(background[nmat:2*nmat, trc]), nsamples, 'k', color="red", lw=1, label="inversion")
l_vs_init, = ax0b.plot(model[nmat:2*nmat, trc], nsamples, 'k', color="blue", lw=1, label="initial_model") #注意单位km/s
ax0b.set_ylim((top,bottom))
# ax0b.set_xlim((2, 10))
ax0b.invert_yaxis()
ax0b.xaxis.tick_top()
ax0b.xaxis.set_label_position('top')
ax0b.set_xlabel("AVO 1")
ax0b.set_yticklabels('')
plt.legend()
ax0b.grid()

ax0c = fig.add_subplot(133)
l_rho_well, = ax0c.plot(coe2, nsamples, 'k', lw=1, label="well_data")
l_rho_inv, = ax0c.plot(np.exp(background[2*nmat:3*nmat, trc]), nsamples, 'k', color="red", lw=1, label="inversion")
l_rho_init, = ax0c.plot(model[2*nmat:3*nmat, trc], nsamples, 'k', color="blue", lw=1, label="initial_model")
ax0c.set_ylim((top,bottom))
# ax0c.set_xlim((2, 4))
ax0c.invert_yaxis()
ax0c.xaxis.tick_top()
ax0c.xaxis.set_label_position('top')
ax0c.set_xlabel("AVO 2")
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

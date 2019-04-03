# -@- coding: utf-8 -@-
"""
Created on Mon Sep 24 12:13:47 2018

@author: 泳浩
"""

import numpy as np
import pandas as pd


##########################################################################################
# 传入初始模型数据
# Vp = np.load("D:\\Physic_Model_Data\\line1_Vp_exactly.npy")
# Vs = np.load("D:\\Physic_Model_Data\\line1_Vs_exactly.npy")
# rho = np.load("D:\\Physic_Model_Data\\line1_Rho_exactly.npy")
# epsilon = np.load("D:\\Physic_Model_Data\\line1_Epsilon_exactly.npy")
# delta = np.load("D:\\Physic_Model_Data\\line1_Delta_exactly.npy")
# delta_new = np.load("D:\\Physic_Model_Data\\line1_Delta_exactly_new.npy")
# gamma = np.load("D:\\Physic_Model_Data\\line1_Gamma_exactly.npy")
# eta = epsilon-delta
# eta_new = epsilon-delta_new
# k = (2*Vs/Vp)**2

# 传入初始模型数据
Vp = np.load("D:\\Physic_Model_Data\\line2_Vp_exactly.npy")
Vs = np.load("D:\\Physic_Model_Data\\line2_Vs_exactly.npy")
rho = np.load("D:\\Physic_Model_Data\\line2_Rho_exactly.npy")
epsilon = np.load("D:\\Physic_Model_Data\\line2_Epsilon_exactly.npy")
delta = np.load("D:\\Physic_Model_Data\\line2_Delta_exactly.npy")
delta_new = np.load("D:\\Physic_Model_Data\\line2_Delta_exactly_new.npy")
gamma = np.load("D:\\Physic_Model_Data\\line2_Gamma_exactly.npy")
eta = epsilon-delta
eta_new = epsilon-delta_new
k = (2*Vs/Vp)**2

##########################################################################################
# 数据范围
top, bottom = 985, 1901

##########################################################################################
# 计算zhang各向异性
coe0 = rho*Vp
coe1 = np.zeros_like(coe0)
coe1_new = np.zeros_like(coe0)
coe1[top:bottom, :] = rho[top:bottom, :]*Vs[top:bottom, :]**2*np.exp(eta[top:bottom, :]/k[top:bottom, :])
coe1_new[top:bottom, :] = rho[top:bottom, :]*Vs[top:bottom, :]**2*np.exp(eta_new[top:bottom, :]/k[top:bottom, :])
coe2 = Vp*np.exp(epsilon)

##########################################################################################
# 模型输出
# np.save("D:\\Physic_Model_Data\\L1_coe0.npy", coe0)
# np.save("D:\\Physic_Model_Data\\L1_coe1.npy", coe1)
# np.save("D:\\Physic_Model_Data\\L1_coe1_new.npy", coe1_new)
# np.save("D:\\Physic_Model_Data\\L1_coe2.npy", coe2)

np.save("D:\\Physic_Model_Data\\L2_coe0.npy", coe0)
np.save("D:\\Physic_Model_Data\\L2_coe1.npy", coe1)
np.save("D:\\Physic_Model_Data\\L2_coe1_new.npy", coe1_new)
np.save("D:\\Physic_Model_Data\\L2_coe2.npy", coe2)

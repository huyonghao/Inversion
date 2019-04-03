# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 20:38:12 2018

@author: W7
"""

import numpy as np
import pandas as pd

# double-travel 2ms_sample
well = np.zeros((4096, 6), dtype=float)
#"Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"
#   0           1       2       3     4     5

# model_1 top
well[1300:1893, 0] = np.loadtxt("D:\\Physic_Model_Data\\CN_THREE_SYN\\epsilon0.txt")
well[1300:1893, 1] = np.loadtxt("D:\\Physic_Model_Data\\CN_THREE_SYN\\delta0.txt")
well[1300:1893, 2] = np.loadtxt("D:\\Physic_Model_Data\\CN_THREE_SYN\\delta0.txt")
well[1300:1893, 3] = np.loadtxt("D:\\Physic_Model_Data\\CN_THREE_SYN\\rho0.txt")
well[1300:1893, 4] = np.loadtxt("D:\\Physic_Model_Data\\CN_THREE_SYN\\vp0.txt")
well[1300:1893, 5] = np.loadtxt("D:\\Physic_Model_Data\\CN_THREE_SYN\\vs0.txt")

#
top, bottom = 1649, 1892
Vp = well[top:bottom, 4]
Vs = well[top:bottom, 5]
Rho = well[top:bottom, 3]
Delta = well[top:bottom, 1]
Epsilon = well[top:bottom, 0]

# Cov Mat iso
cov_iso = np.cov(np.hstack((np.log(Vp), np.log(Vs), np.log(Rho))).reshape(3, -1))

# Cov Mat ani
eta = Epsilon-Delta
k = (2*Vs/Vp)**2
coe0 = Rho*Vp
coe1 = Rho*Vs**2*np.exp(eta/k)
coe2 = Vp*np.exp(Epsilon)
cov_ani = np.cov(np.hstack((np.log(coe0), np.log(coe1), np.log(coe2))).reshape(3, -1))

wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("SYN_real_n203.csv")

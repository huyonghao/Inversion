# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:44:39 2019

@author: carro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size":22})

filepath = "D://Physic_Model_Data//n203.las"
well_n203 = pd.read_fwf(filepath, header=None, skiprows=80)
well_selected = well_n203.iloc[:, [0, 9, 16, 18, 23, 24, 35, 38]]
well_selected.columns = ["Dpeth", "YoungsModulus", "BulkModulus", "ShearModulus", "Poisson's_Ratio",
                         "Density", "P_Wave_Velocity", "S_Wave_Velocity"]
well_selected = well_selected.where(well_selected > 0)
well_selected.loc[:, ["P_Wave_Velocity", "S_Wave_Velocity"]] = well_selected.loc[:, ["P_Wave_Velocity", "S_Wave_Velocity"]]/1000
well_selected.index = well_selected.loc[:, "Dpeth"]
well_selected = well_selected.drop(columns="Dpeth")
# well_selected = well_selected.fillna(0)
para_1 = well_selected.P_Wave_Velocity.rolling(100).mean()
para_2 = well_selected.S_Wave_Velocity.rolling(100).mean()
para_3 = well_selected.Density.rolling(100).mean()
# dt = 0.1524/np.nan_to_num(para_1)/1e3
dt = 0.1524/para_1/1e3
dt_sum = 2*np.cumsum(dt)
cov = np.cov(np.hstack((np.log(para_1), np.log(para_2), np.log(para_3))).reshape(3, -1))

data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_025.csv"))
data_1 = data["Vp"]/1000
data_2 = data["Vs"]/1000
data_3 = data["Rho"]

plt.subplot(4, 1, 1)
plt.plot(para_1)
plt.plot(data_1)
plt.subplot(4, 1, 2)
plt.plot(para_2)
plt.plot(data_2)
plt.subplot(4, 1, 3)
plt.plot(para_3)
plt.plot(data_3)
plt.subplot(4, 1, 4)
plt.plot(dt_sum)
plt.show()
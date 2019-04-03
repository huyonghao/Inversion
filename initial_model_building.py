# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 11:58:52 2018

@author: DELL-Workstaion
"""

import numpy as np
import pandas as pd


# Geo_L1
# 0-8192ms double-travel 2ms_sample
# 2191 trace
model = np.zeros((4097, 2191))

# model trace Line_1
trc1, trc2, trc3, trc4, trc5, trc6 = 420, 569, 660, 809, 907, 1056, 
trc7, trc8, trc9, trc10, trc11, trc12 = 1148, 1297, 1380, 1530, 1627, 1773


# Vp
model[:, :] = 2800
# 0
model[0:985, :] = 1480
# 1
model[985:1175, :] = 2600
# 2
model[1175:1606, :] = 2800   
# 3 研究块体
# [1600:1743, trc1:trc2]
# [1600:1741, trc3:trc4]
# [1600:1733, trc5:trc6]
# [1600:1733, trc7:trc8]
# [1600:1729, trc9:trc10]
# [1600:1722, trc11:trc12]
# 3 研究块体
model[1606:1752, trc1:trc2] = 2568
model[1606:1752, trc3:trc4] = 2585
model[1606:1752, trc5:trc6] = 2650
model[1606:1752, trc7:trc8] = 2690
model[1606:1752, trc9:trc10] = 2732
model[1606:1752, trc11:trc12] = 2776
# 4
model[1752:1902, :] = 2800
# 5
model[1902:4097, :] = 1480
np.save("D:\\Physic_Model_Data\\line1_Vp_exactly", model/1000) # 模型单位转化m/s->km/s


# Vs
model[:, :] = 1437
# 0
model[0:985, :] = 0
# 1
model[985:1175, :] = 1203
# 2
model[1175:1606, :] = 1437   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 1766
model[1606:1752, trc3:trc4] = 1790
model[1606:1752, trc5:trc6] = 1837
model[1606:1752, trc7:trc8] = 1878
model[1606:1752, trc9:trc10] = 1917
model[1606:1752, trc11:trc12] = 1950
# 4
model[1752:1902, :] = 1437
# 5
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line1_Vs_exactly", model/1000) # 模型单位转化m/s->km/s


# Rho
model[:, :] = 1.6
# 0
model[0:985, :] = 1
# 1
model[985:1175, :] = 1.18
# 2
model[1175:1606, :] = 1.6   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 2.48
model[1606:1752, trc3:trc4] = 2.52
model[1606:1752, trc5:trc6] = 2.54
model[1606:1752, trc7:trc8] = 2.56
model[1606:1752, trc9:trc10] = 2.61
model[1606:1752, trc11:trc12] = 2.63
# 4
model[1752:1902, :] = 1.6
# 5
model[1902:4097, :] = 1
np.save("D:\\Physic_Model_Data\\line1_Rho_exactly", model)


# Epsilon
# model[:, :] = np.nan
model[:, :] = 0
# 0
# model[0:985, :] = np.nan
model[0:985, :] = 0
# 1
# model[985:1175, :] = np.nan
model[985:1175, :] = 0
# 2
# model[1175:1606, :] = np.nan   
model[1175:1606, :] = 0   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 0.19
model[1606:1752, trc3:trc4] = 0.187
model[1606:1752, trc5:trc6] = 0.164619
model[1606:1752, trc7:trc8] = 0.148907
model[1606:1752, trc9:trc10] = 0.138206
model[1606:1752, trc11:trc12] = 0.124719
# 4
# model[1752:1902, :] = np.nan
model[1752:1902, :] = 0
# 5
# model[1902:4097, :] = np.nan
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line1_Epsilon_exactly", model)


# Delta 0
# model[:, :] = np.nan
model[:, :] = 0
# 0
# model[0:985, :] = np.nan
model[0:985, :] = 0
# 1
# model[985:1175, :] = np.nan
model[985:1175, :] = 0
# 2
# model[1175:1606, :] = np.nan   
model[1175:1606, :] = 0   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 0.042
model[1606:1752, trc3:trc4] = 0.051
model[1606:1752, trc5:trc6] = 0.038
model[1606:1752, trc7:trc8] = 0.050
model[1606:1752, trc9:trc10] = 0.046
model[1606:1752, trc11:trc12] = 0.048
# 4
# model[1752:1902, :] = np.nan
model[1752:1902, :] = 0
# 5
# model[1902:4097, :] = np.nan
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line1_Delta_exactly", model)


# Delta 1
# model[:, :] = np.nan
model[:, :] = 0
# 0
# model[0:985, :] = np.nan
model[0:985, :] = 0
# 1
# model[985:1175, :] = np.nan
model[985:1175, :] = 0
# 2
# model[1175:1606, :] = np.nan   
model[1175:1606, :] = 0   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 0.395828645
model[1606:1752, trc3:trc4] = 0.401833919
model[1606:1752, trc5:trc6] = 0.353119732
model[1606:1752, trc7:trc8] = 0.361189324
model[1606:1752, trc9:trc10] = 0.327268067
model[1606:1752, trc11:trc12] = 0.324737185
# 4
# model[1752:1902, :] = np.nan
model[1752:1902, :] = 0
# 5
# model[1902:4097, :] = np.nan
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line1_Delta_exactly_new", model)


# Gamma
# model[:, :] = np.nan
model[:, :] = 0
# 0
# model[0:985, :] = np.nan
model[0:985, :] = 0
# 1
# model[985:1175, :] = np.nan
model[985:1175, :] = 0
# 2
# model[1175:1606, :] = np.nan   
model[1175:1606, :] = 0   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 0.166
model[1606:1752, trc3:trc4] = 0.158
model[1606:1752, trc5:trc6] = 0.148
model[1606:1752, trc7:trc8] = 0.120999
model[1606:1752, trc9:trc10] = 0.112838
model[1606:1752, trc11:trc12] = 0.098615
# 4
# model[1752:1902, :] = np.nan
model[1752:1902, :] = 0
# 5
# model[1902:4097, :] = np.nan
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line1_Gamma_exactly", model)

####################################################################################
# Geo_L2
# 0-8192ms double-travel 2ms_sample
# 2191 trace
model = np.zeros((4097, 2191))

# model trace Line_1
trc1, trc2, trc3, trc4, trc5, trc6 = 420, 569, 660, 809, 907, 1056, 
trc7, trc8, trc9, trc10, trc11, trc12 = 1148, 1297, 1380, 1530, 1627, 1773


# Vp
model[:, :] = 2800
# 0
model[0:985, :] = 1480
# 1
model[985:1175, :] = 2600
# 2
model[1175:1606, :] = 2800   
# 3 研究块体
# [1600:1743, trc1:trc2]
# [1600:1741, trc3:trc4]
# [1600:1733, trc5:trc6]
# [1600:1733, trc7:trc8]
# [1600:1729, trc9:trc10]
# [1600:1722, trc11:trc12]
# 3 研究块体
model[1606:1752, trc1:trc2] = 2200
model[1606:1752, trc3:trc4] = 2350
model[1606:1752, trc5:trc6] = 2503
model[1606:1752, trc7:trc8] = 2672
model[1606:1752, trc9:trc10] = 2830
model[1606:1752, trc11:trc12] = 2992
# 4
model[1752:1902, :] = 2800
# 5
model[1902:4097, :] = 1480
np.save("D:\\Physic_Model_Data\\line2_Vp_exactly", model/1000) # 模型单位转化m/s->km/s


# Vs
model[:, :] = 1437
# 0
model[0:985, :] = 0
# 1
model[985:1175, :] = 1203
# 2
model[1175:1606, :] = 1437   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 1389
model[1606:1752, trc3:trc4] = 1512
model[1606:1752, trc5:trc6] = 1655
model[1606:1752, trc7:trc8] = 1780
model[1606:1752, trc9:trc10] = 1898
model[1606:1752, trc11:trc12] = 2023
# 4
model[1752:1902, :] = 1437
# 5
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line2_Vs_exactly", model/1000) # 模型单位转化m/s->km/s


# Rho
model[:, :] = 1.6
# 0
model[0:985, :] = 1
# 1
model[985:1175, :] = 1.18
# 2
model[1175:1606, :] = 1.6   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 2.38
model[1606:1752, trc3:trc4] = 2.41
model[1606:1752, trc5:trc6] = 2.45
model[1606:1752, trc7:trc8] = 2.49
model[1606:1752, trc9:trc10] = 2.54
model[1606:1752, trc11:trc12] = 2.56
# 4
model[1752:1902, :] = 1.6
# 5
model[1902:4097, :] = 1
np.save("D:\\Physic_Model_Data\\line2_Rho_exactly", model)


# Epsilon
# model[:, :] = np.nan
model[:, :] = 0
# 0
# model[0:985, :] = np.nan
model[0:985, :] = 0
# 1
# model[985:1175, :] = np.nan
model[985:1175, :] = 0
# 2
# model[1175:1606, :] = np.nan   
model[1175:1606, :] = 0   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 0.139
model[1606:1752, trc3:trc4] = 0.167
model[1606:1752, trc5:trc6] = 0.198
model[1606:1752, trc7:trc8] = 0.223
model[1606:1752, trc9:trc10] = 0.241
model[1606:1752, trc11:trc12] = 0.253
# 4
# model[1752:1902, :] = np.nan
model[1752:1902, :] = 0
# 5
# model[1902:4097, :] = np.nan
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line2_Epsilon_exactly", model)


# Delta 0
# model[:, :] = np.nan
model[:, :] = 0
# 0
# model[0:985, :] = np.nan
model[0:985, :] = 0
# 1
# model[985:1175, :] = np.nan
model[985:1175, :] = 0
# 2
# model[1175:1606, :] = np.nan   
model[1175:1606, :] = 0   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 0.028663781
model[1606:1752, trc3:trc4] = 0.0374093
model[1606:1752, trc5:trc6] = 0.04354269
model[1606:1752, trc7:trc8] = 0.056272568
model[1606:1752, trc9:trc10] = 0.040402293
model[1606:1752, trc11:trc12] = 0.032962607
# 4
# model[1752:1902, :] = np.nan
model[1752:1902, :] = 0
# 5
# model[1902:4097, :] = np.nan
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line2_Delta_exactly", model)


# Delta 1
# model[:, :] = np.nan
model[:, :] = 0
# 0
# model[0:985, :] = np.nan
model[0:985, :] = 0
# 1
# model[985:1175, :] = np.nan
model[985:1175, :] = 0
# 2
# model[1175:1606, :] = np.nan   
model[1175:1606, :] = 0   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 0.424442379
model[1606:1752, trc3:trc4] = 0.451757201
model[1606:1752, trc5:trc6] = 0.465379425
model[1606:1752, trc7:trc8] = 0.499941911
model[1606:1752, trc9:trc10] = 0.466115083
model[1606:1752, trc11:trc12] = 0.439810263
# 4
# model[1752:1902, :] = np.nan
model[1752:1902, :] = 0
# 5
# model[1902:4097, :] = np.nan
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line2_Delta_exactly_new", model)


# Gamma
# model[:, :] = np.nan
model[:, :] = 0
# 0
# model[0:985, :] = np.nan
model[0:985, :] = 0
# 1
# model[985:1175, :] = np.nan
model[985:1175, :] = 0
# 2
# model[1175:1606, :] = np.nan   
model[1175:1606, :] = 0   
# 3 研究块体
# 3 研究块体
model[1606:1752, trc1:trc2] = 0.126
model[1606:1752, trc3:trc4] = 0.148
model[1606:1752, trc5:trc6] = 0.174
model[1606:1752, trc7:trc8] = 0.191
model[1606:1752, trc9:trc10] = 0.218
model[1606:1752, trc11:trc12] = 0.234
# 4
# model[1752:1902, :] = np.nan
model[1752:1902, :] = 0
# 5
# model[1902:4097, :] = np.nan
model[1902:4097, :] = 0
np.save("D:\\Physic_Model_Data\\line2_Gamma_exactly", model)


# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 20:38:12 2018

@author: W7
"""

import numpy as np
import pandas as pd

#0-8192ms double-travel 2ms_sample
well = np.zeros((4096, 6), dtype=float)
#"Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"
#   0           1       2       3     4     5
# Delta 0 lower.
DeltaValue = 1
#0
well[0:985, 3] = 1
well[0:985, 4] = 1480
#1
well[985:1175, 3] = 1.18
well[985:1175, 4] = 2600
well[985:1175, 5] = 1203
#2
well[1175:1606, 3] = 1.6
well[1175:1606, 4] = 2800   
well[1175:1606, 5] = 1437
#3 研究块体：粘土含量0.5
well[1606:1752, 0] = 0.19
if DeltaValue==0:
	well[1606:1752, 1] = 0.042
else:
	well[1606:1752, 1] = 0.395828645
well[1606:1752, 2] = 0.166
well[1606:1752, 3] = 2.48
well[1606:1752, 4] = 2568
well[1606:1752, 5] = 1766
#4
well[1752:1902, 3] = 1.6
well[1752:1902, 4] = 2800
well[1752:1902, 5] = 1437
#5
well[1902:4096, 3] = 1
well[1902:4096, 4] = 1480
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line1_05.csv")

#3 研究块体：粘土含量0.45
well[1606:1752, 0] = 0.187
if DeltaValue==0:
	well[1606:1752, 1] = 0.051
else:
	well[1606:1752, 1] = 0.401833919
well[1606:1752, 2] = 0.158
well[1606:1752, 3] = 2.52
well[1606:1752, 4] = 2585
well[1606:1752, 5] = 1790
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line1_045.csv")

#3 研究块体：粘土含量0.4
well[1606:1752, 0] = 0.164619
if DeltaValue==0:
	well[1606:1752, 1] = 0.038
else:
	well[1606:1752, 1] = 0.353119732
well[1606:1752, 2] = 0.148
well[1606:1752, 3] = 2.54
well[1606:1752, 4] = 2650
well[1606:1752, 5] = 1837
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line1_04.csv")

#3 研究块体：粘土含量0.35
well[1606:1752, 0] = 0.148907
if DeltaValue==0:
	well[1606:1752, 1] = 0.050
else:
	well[1606:1752, 1] = 0.361189324
well[1606:1752, 2] = 0.120999
well[1606:1752, 3] = 2.56
well[1606:1752, 4] = 2690
well[1606:1752, 5] = 1878
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line1_035.csv")

#3 研究块体：粘土含量0.3
well[1606:1752, 0] = 0.138206
if DeltaValue==0:
	well[1606:1752, 1] = 0.046
else:
	well[1606:1752, 1] = 0.327268067
well[1606:1752, 2] = 0.112838
well[1606:1752, 3] = 2.61
well[1606:1752, 4] = 2732
well[1606:1752, 5] = 1917
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line1_03.csv")

#3 研究块体：粘土含量0.25
well[1606:1752, 0] = 0.124719
if DeltaValue==0:
	well[1606:1752, 1] = 0.048
else:
	well[1606:1752, 1] = 0.324737185
well[1606:1752, 2] = 0.098615
well[1606:1752, 3] = 2.63
well[1606:1752, 4] = 2776
well[1606:1752, 5] = 1950
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line1_025.csv")


#3 研究块体：孔隙度16.4
well[1606:1752, 0] = 0.139
if DeltaValue==0:
	well[1606:1752, 1] = 0.028664
else:
	well[1606:1752, 1] = 0.424442379
well[1606:1752, 2] = 0.126
well[1606:1752, 3] = 2.38
well[1606:1752, 4] = 2200
well[1606:1752, 5] = 1389
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line2_164.csv")

#3 研究块体：孔隙度11.5
well[1606:1752, 0] = 0.167
if DeltaValue==0:
	well[1606:1752, 1] = 0.037409
else:
	well[1606:1752, 1] = 0.451757201
well[1606:1752, 2] = 0.148
well[1606:1752, 3] = 2.41
well[1606:1752, 4] = 2350
well[1606:1752, 5] = 1512
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line2_115.csv")

#3 研究块体：孔隙度6.5
well[1606:1752, 0] = 0.198
if DeltaValue==0:
	well[1606:1752, 1] = 0.043543
else:
	well[1606:1752, 1] = 0.465379425
well[1606:1752, 2] = 0.174
well[1606:1752, 3] = 2.45
well[1606:1752, 4] = 2503
well[1606:1752, 5] = 1655
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line2_065.csv")

#3 研究块体：孔隙度4.7
well[1606:1752, 0] = 0.223
if DeltaValue==0:
	well[1606:1752, 1] = 0.056273
else:
	well[1606:1752, 1] = 0.499941911
well[1606:1752, 2] = 0.191
well[1606:1752, 3] = 2.49
well[1606:1752, 4] = 2672
well[1606:1752, 5] = 1780
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line2_047.csv")

#3 研究块体：孔隙度3.8
well[1606:1752, 0] = 0.241
if DeltaValue==0:
	well[1606:1752, 1] = 0.040402
else:
	well[1606:1752, 1] = 0.466115083
well[1606:1752, 2] = 0.218
well[1606:1752, 3] = 2.54
well[1606:1752, 4] = 2830
well[1606:1752, 5] = 1898
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line2_038.csv")

#3 研究块体：孔隙度2.1
well[1606:1752, 0] = 0.253
if DeltaValue==0:
	well[1606:1752, 1] = 0.032963
else:
	well[1606:1752, 1] = 0.439810263
well[1606:1752, 2] = 0.234
well[1606:1752, 3] = 2.56
well[1606:1752, 4] = 2992
well[1606:1752, 5] = 2023
wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\line2_021.csv")

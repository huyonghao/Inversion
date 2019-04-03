# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 20:38:12 2018

@author: W7
"""

import numpy as np
import pandas as pd

# double-travel 2ms_sample
well = np.zeros((6, 6), dtype=float)
#"Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"
#   0           1       2       3     4     5

# # model_1 top
# well[0, 0] = 0.26
# well[0, 1] = 0.12
# well[0, 2] = 0.18
# well[0, 3] = 2.68
# well[0, 4] = 4.345
# well[0, 5] = 2.584

# # model_1 bottom
# well[1, 0] = 0.1
# well[1, 1] = 0.12
# well[1, 2] = 0.1
# well[1, 3] = 2.538
# well[1, 4] = 3.670
# well[1, 5] = 2.149

# # model_2 top
# well[2, 0] = 0.133
# well[2, 1] = 0.12
# well[2, 2] = 0
# well[2, 3] = 2.43
# well[2, 4] = 2.96
# well[2, 5] = 1.38

# # model_2 bottom
# well[3, 0] = 0
# well[3, 1] = 0
# well[3, 2] = 0
# well[3, 3] = 2.14
# well[3, 4] = 3.49
# well[3, 5] = 2.29

# # model_3 top
# well[4, 0] = 0.133
# well[4, 1] = 0.12
# well[4, 2] = 0
# well[4, 3] = 2.35
# well[4, 4] = 2.73
# well[4, 5] = 1.24

# # model_3 bottom
# well[5, 0] = 0
# well[5, 1] = 0
# well[5, 2] = 0
# well[5, 3] = 2.13
# well[5, 4] = 2.02
# well[5, 5] = 1.23

# # model_4 top
# well[6, 0] = 0.133
# well[6, 1] = 0.12
# well[6, 2] = 0
# well[6, 3] = 2.34
# well[6, 4] = 2.24
# well[6, 5] = 1.62

# # model_4 bottom
# well[7, 0] = 0
# well[7, 1] = 0
# well[7, 2] = 0
# well[7, 3] = 2.27
# well[7, 4] = 1.65
# well[7, 5] = 1.06

# ruger_model_1 top
well[0, 0] = 0.133 #Epsilon
well[0, 1] = 0.12 #Delta
well[0, 2] = 0 #Gamma
well[0, 3] = 2.35 #Rho
well[0, 4] = 3.30 #Vp
well[0, 5] = 1.70 #Vs

# ruger_model_1 bottom
well[1, 0] = 0
well[1, 1] = 0
well[1, 2] = 0
well[1, 3] = 2.49
well[1, 4] = 4.20
well[1, 5] = 2.70

# ruger_model_2 top
well[2, 0] = 0.133
well[2, 1] = 0.12
well[2, 2] = 0
well[2, 3] = 2.43
well[2, 4] = 2.96
well[2, 5] = 1.38

# ruger_model_2 bottom
well[3, 0] = 0
well[3, 1] = 0
well[3, 2] = 0
well[3, 3] = 2.14
well[3, 4] = 3.49
well[3, 5] = 2.29

# ruger_model_3 top
well[4, 0] = 0.133
well[4, 1] = 0.12
well[4, 2] = 0
well[4, 3] = 2.35
well[4, 4] = 2.73
well[4, 5] = 1.24

# ruger_model_3 bottom
well[5, 0] = 0
well[5, 1] = 0
well[5, 2] = 0
well[5, 3] = 2.13
well[5, 4] = 2.02
well[5, 5] = 1.23

wd = pd.DataFrame(well, columns=["Epsilon", "Delta", "Gamma", "Rho", "Vp", "Vs"])
wd.to_csv("D:\\Physic_Model_Data\\Analysis_of_AVO.csv")

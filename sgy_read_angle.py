# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:50:34 2018

@author: ROOT
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import segyio

# with segyio.open("D:\\Physic_Model_Data\\L1_angle_gather.sgy", "r") as f:
#     print(f.offsets)
#     print(f.trace)
#     a = np.array([i for i in f.trace[7260:7275]]).T
    # print(abs(f.trace[7269]*1e10).max())
    # print(abs(f.trace[7269]*1e10).min())
    # tmax = len(f.samples)
    # np.save("7260_7275_cdp888", a)

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
    
# # 人工合成地震道绘图
# fig = plt.figure(figsize=(8, 4), dpi=300)
# ax1 = fig.add_subplot(111)
# excursion = 2
# dt = (2*1e-3) # 采样间隔
# t = np.arange(0, tmax*dt, dt) # 采样总时间
# min_plot_time = 0  #units s 时窗范围
# max_plot_time = tmax*dt  #units s 时窗范围
# plot_vawig(ax1, a, t[:], excursion, min_plot_time, max_plot_time)
# ax1.set_xlabel("$Trace$ $(theta)$")
# ax1.set_yticklabels('')
# ax1.set_ylabel("$Time (sec)$")
# plt.show()
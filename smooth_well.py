# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:40:20 2018

@author: 泳浩
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class Smooth(object):
	'''
	Lowess smooth
	'''
	def __init__(self, data):
		# lowess 
		smooth = pd.DataFrame()
		lowess = sm.nonparametric.lowess
		x = np.arange(data.shape[0])
		for i in range(data.shape[1]):
		    y = np.array(data.iloc[:,[i]]).flatten()
		    smooth[i] = lowess(y, x, frac=0.5, it=0)[:, 1].flatten()
		smooth.columns = data.columns
		self.data = data
		self.smooth = smooth

	def sm(self):
		return self.smooth

	def plot_map(self):
		fig = plt.figure()
		ax = plt.subplot()
		ax.plot(self.smooth, label="smooth")
		ax.plot(self.data, label="initial")
		plt.legend()
		plt.show()


class Smooth2(object):
	'''
	Lowess smooth
	'''
	def __init__(self, data):
		# lowess 
		smooth = pd.DataFrame()
		lowess = sm.nonparametric.lowess
		x = np.arange(data.shape[0])
		for i in range(data.shape[1]):
		    y = np.array(data[:, i]).flatten()
		    smooth[i] = lowess(y, x, frac=0.2, it=0)[:, 1].flatten()
		self.smooth = np.array(smooth)

	def sm(self):
		return self.smooth	

	def plot_map(self):
		fig = plt.figure()
		ax = plt.subplot()
		ax.imshow(self.smooth, label="smooth")
		plt.legend()
		plt.show()

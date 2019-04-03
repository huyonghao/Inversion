import pandas as pd 
import numpy as np 
import scipy.signal as signal


class SavGol(object):
	
	def __init__(self):
		pass

	def pandas_savgol(self, data, window_length=15, polyorder=1):
		smooth = pd.DataFrame()
		for i in range(data.shape[1]):
		    data_columns = np.array(data.iloc[:,[i]]).flatten()
		    smooth[i] = signal.savgol_filter(data_columns, window_length, polyorder)
		smooth.columns = data.columns
		return smooth

def pandas_savgol(data, window_length=15, polyorder=1):
	'''
	1-D
	'''
	smooth = pd.DataFrame()
	for i in range(data.shape[1]):
	    data_columns = np.array(data.iloc[:,[i]]).flatten()
	    smooth[i] = signal.savgol_filter(data_columns, window_length, polyorder)
	smooth.columns = data.columns
	return smooth

def initmodel_savgol(model, window_length=101, polyorder=1):
	'''
	2-D
	'''
	for i in range(model.shape[1]):
	    model[:, i] = signal.savgol_filter(model[:, i].flatten(), window_length, polyorder)
	return model
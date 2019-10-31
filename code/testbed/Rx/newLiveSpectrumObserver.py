"""
	Code by Zhujun and Zhijing, modified by Shinan
"""
import numpy as np
import sys
import os
import glob
import multiprocessing as mp
import tensorflow as tf
import pickle
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import mean_squared_error

# set environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.seterr(divide = 'ignore')
AUTOTUNE = tf.data.experimental.AUTOTUNE

# key values
filename = sys.argv[1]
core = 1
power_level = -20
thresh = 5
downsample = 10  #downsample rate
timestamps = 100
predict_len = 25
model_filename = '100_25.h5'
out_filename = './plotData/usrpSameSource0.txt'
batch_size = 64
timesteps = 50
predict_steps = 25
data_dim = 128
epochs = 10
hidden_size = 64
train_num = 500000
valid_num = 5000
test_num = 10000
batch_size = 1
window_size = 100
predict_size = 25
shift_size = 125
ano_count = 0
all_count = 0


def standard(sequence, mean_data, std_data):
	"""
		standard normalization
	"""
	# mean_data = np.mean(sequence)
	# std_data = np.std(sequence)
	if std_data == 0:
		raise Exception('Bad data')
		#ret = [1 for i in range(len(sequence))]
	else:
		ret = []
		for i in range(len(sequence)):
			ret.append((sequence[i] - mean_data)/std_data)
	return ret


def extract(fft_array):
	"""
		extarct method: standard normalization
	"""
	data = []
	ft_array = []
	for line in fft_array:
		a = line.split()
		data.extend([float(a[i]) for i in range(len(a))])
		if len(data) == timestamps*128:
			mean_data = np.mean(data)
			std_data = np.std(data)
		if len(data) == (timestamps + predict_len)*128:
			data = standard(data, mean_data, std_data)
			# print np.shape(data)
			ft_array.append(data)

			"""
				to sample more data
			"""
			data = []

	return ft_array


def multicore_live_monitor(core, index):
	## run faster using multiprocessing on multicores of CPU
	pool = mp.Pool(processes = core)
	pool.map(live_monitor, index)
	return


def live_monitor(index):
	global ano_count
	global all_count
	model = tf.keras.models.load_model(model_filename)
	float_formatter = lambda x: "%6.3f" % x
	# print('{0} starts pre-processing.').format(filename)

	fid = open(filename, 'rb')
	out_fid = open(out_filename, 'a')
	count = 1280000*2 # divide data into blocks. Each block has 1280000 IQ samples
	Slen = 128 # Number of FFT points
	block_count = 0  # Our data volume is large,
	# so only process first 1200 blocks.

	MAX_num_of_block = 1200
	trash_count = 102400 # throw the begining of data, becasue sometimes
	# the begining of collected data are large numbers
	# which don't seem to be real signal measurements.
	dont_care = np.fromfile(fid, np.float32, count=trash_count).reshape((-1, 2))

	data_test = []
	block_num = 0 # actually passed inside
	starttime = time.time()

	while block_count < MAX_num_of_block:
		data_raw = np.fromfile(fid, np.float32, count=count)
		if block_count % core == index:

			data_array = data_raw.reshape((-1, 2))
			(a, _) = data_array.shape
			if a < count//2:
				end = (a//Slen)*Slen
			else:
				end = count//2

			new_array = data_array[:,0] + 1j*data_array[:,1]

			nan_count_a = 0
			nan_count_p = 0

			fft_array = []
			#every downsample*Slen points, compute FFT once
			for start_point in range(0, end, downsample*Slen):
				raw_data = new_array[start_point:(start_point+Slen)]
				Y = np.fft.fft(raw_data, Slen)
				P1 = np.fft.fftshift(Y)
				P2 = np.absolute(P1)
				amp = 10*np.log10(P2/Slen)

			# data clean
			# amplitude is unlikely to exceed 0dB. If amp > 0, could be measurement
			# error in data, so force it to -30dB
				if np.isnan(amp).any():
					nan_count_a += 1
					amp[np.isnan(amp)] = -120
				# if np.isnan(phase).any():
				#         nan_count_p += 1
				amp[amp<-120] = -120
				amp[amp>0] = -30
				if np.any(amp>power_level):
					print('Power alarm triggered.')
				output_data = amp.reshape((-1, Slen))
				# print np.shape(output_data)
				output_string = np.array2string(output_data, formatter={'float_kind':float_formatter})
				fft_array.append(output_string.lstrip('[[').rstrip(']]').replace('\n  ', ' '))
				# output_data = np.column_stack((amp, phase))

			data_test = extract(fft_array)
			# print(np.shape(extract(fft_array)))

			series = np.array(data_test).reshape((-1, 128)).astype('float64')
			# print(np.shape(series))

			# Construct MSE DataFrame
			if np.any(np.equal(series, None)) != True:
				pred_list = [model_forecast(model, series)]
				# print(series)
				true_list = [series[shift_size * i - predict_size: shift_size*i, :].reshape((-1, 25, 128))
							for i in range(1, len(series) // shift_size + 1)]
				mse_list = []
				error_df_list = []

				for i in range(len(pred_list)):
					pred = pred_list[0][i]
					true = true_list[i]
					# if np.isnan(pred).any() or np.isnan(true).any() != True:
					mse = np.mean(np.power(pred.reshape(-1, 128)
				                                - true.reshape(-1, 128), 2), axis=1)
					endtime = time.time()
					deltatime = endtime -starttime
					log = str(np.mean(mse)) +',' + str(deltatime)

					# log = str(np.mean(mse))
					print(log)
					out_fid.write(log + '\n')

			if a < count//2:
				break
		block_count += 1


def model_forecast(model, series):
	ds = tf.data.Dataset.from_tensor_slices(series)
	ds = ds.window(window_size + predict_size, shift=shift_size,
				   drop_remainder=True)
	ds = ds.flat_map(lambda w: w.batch(window_size + predict_size))
	ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
	forecast = model.predict(ds)
	return forecast

if __name__ == "__main__":
	# test_LSTM1(t_filename, model_filename, out_filename, out_filename_cdf)
    multicore_live_monitor(core, range(core))
    # live_monitor(1)

"""
	Code by Zhujun and Zhijing, modified by Shinan
"""
import numpy as np
import sys
import os
import glob
import multiprocessing as mp
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.layers.core import Activation
from keras import backend as K
from keras.optimizers import RMSprop
from keras.layers.pooling import AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.core import Dropout
from keras.models import load_model
import keras
from keras.layers import Input
from sklearn.metrics import mean_squared_error
from keras.layers.normalization import BatchNormalization

# set environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.seterr(divide = 'ignore')

# key values
filename = sys.argv[1]
core = 8
power_level = -20
thresh = 5
downsample = 10  #downsample rate
timestamps = 100
predict_len = 25
t_filename = 'now_882.5M_5m_ft.txt'
model_filename = 'mix_880M_200k.h5'
weights_filename = '880M_mix_weights.h5'
out_filename = 'out.txt'
out_filename_cdf = 'out_cdf.txt'
batch_size = 64
timesteps = 50
predict_steps = 25
data_dim = 128
epochs = 10
hidden_size = 64
train_num = 500000
valid_num = 5000
test_num = 10000
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

			#
			# out.write('\n')

			"""
				to sample more data
			"""
			data = []

	return ft_array


def multicore_downsample_feature(core, index):
    ## run faster using multiprocessing on multicores of CPU
    pool = mp.Pool(processes = core)
    pool.map(downsample_feature, index)
    return


def downsample_feature(index):
        global ano_count
        global all_count
	model = load_model(model_filename)
	model.load_weights(weights_filename)
	for filename in glob.glob('./*.dat'):
		# for IQ data, we compute the FFT amplitude, and save it in txt file
		# w_filename = filename.replace('.dat', '_ds.txt')
		# w_filename = filename.replace('.dat', str(index) + '_ft.txt')
		# out = open(w_filename, 'w')
		float_formatter = lambda x: "%6.3f" % x
		# print('{0} starts pre-processing.').format(filename)

		fid = open(filename, 'rb')
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

					# np.set_printoptions(formatter={'float_kind':float_formatter})
					#phase = np.angle(P1)

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
				# print np.shape(extract(fft_array))
				# data_test.extend(extract(fft_array))
				# print np.shape(data_test)
				# block_num += 1
				# if block_num % 25 == 0:
				# 	# print np.shape(data_test)
				# 	test_LSTM(data_test, model_filename, out_filename, out_filename_cdf)
				# 	data_test = []

				# meaning this is the last block
				x = []
				y = []

				for data in data_test:
					np.reshape(data, (16000,))
					x1_test = [data[i] for i in range(timesteps*data_dim)]
					y1_test = [data[i] for i in range(timesteps*data_dim, (timesteps+predict_steps)*data_dim)]
					x.append(x1_test)
					y.append(y1_test)

				test_x = np.reshape(x, (8, timesteps, data_dim)) # 1 block equals to 8 lines of features
				test_y = np.reshape(y, (8, data_dim*predict_steps))
				predict_y = model.predict(test_x)
				for j in range(len(predict_y)):
				    mse = mean_squared_error(test_y[j], predict_y[j])
                                    all_count += 1
				    if np.any(mse>thresh):
                                            ano_count += 1.0
					    print 'Anomaly detected, rate:', ano_count / all_count * 100 , '%'
				    # mse_list.append(mse)

				if a < count//2:
					break
			block_count += 1


		print('{0} done').format(filename)
		print('nan count: {0}-{1}').format(nan_count_a, nan_count_p)
		print("dataset size: {0}").format(block_count)


def test_LSTM(data_test, model_filename, out_filename, out_filename_cdf):
	mse_list = []
	# out_f = open(out_filename, 'w')

	x = []
	y = []

	for data in data_test:
		np.reshape(data, (16000,))
		x1_test = [data[i] for i in range(timesteps*data_dim)]
		y1_test = [data[i] for i in range(timesteps*data_dim, (timesteps+predict_steps)*data_dim)]
		x.append(x1_test)
		y.append(y1_test)

	test_x = np.reshape(x, (200, timesteps, data_dim))
	test_y = np.reshape(y, (200, data_dim*predict_steps))
	predict_y = model.predict(test_x)
	for j in range(len(predict_y)):
	    mse = mean_squared_error(test_y[j], predict_y[j])
	    print mse #out_f.write('%s\n' %mse)
	    mse_list.append(mse)


if __name__ == "__main__":
	# test_LSTM1(t_filename, model_filename, out_filename, out_filename_cdf)
    multicore_downsample_feature(core, range(core))

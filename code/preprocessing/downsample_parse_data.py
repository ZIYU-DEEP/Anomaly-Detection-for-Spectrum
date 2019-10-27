"""
	Code by Zhujun
"""
import numpy as np
import sys
import os
import glob

path = sys.argv[1]
downsample = 10  #downsample rate
# band = sys.argv[2]
# samp_rate = sys.argv[3]
# loc = sys.argv[4]

# create a folder to save downsampled data
if not os.path.exists(path + '/downsample_' + str(downsample)):
	os.mkdir(path + '/downsample_' + str(downsample))


for filename in glob.glob(path + '/*.dat'):
	# for IQ data, we compute the FFT amplitude, and save it in txt file
	w_filename = filename.replace('.dat', '_ap.txt')
	w_filename = w_filename.replace(path, 
					path + '/downsample_' + str(downsample))
	print('{0} starts pre-processing.').format(filename)



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



	while block_count < MAX_num_of_block:
		data_array = np.fromfile(fid, np.float32, count=count).reshape((-1, 2))
		(a, _) = data_array.shape
		if a < count//2:
			end = (a//Slen)*Slen
		else:
			end = count//2

		block_count += 1
		new_array = data_array[:,0] + 1j*data_array[:,1]

		nan_count_a = 0
		nan_count_p = 0
		#every downsample*Slen points, compute FFT once
		for start_point in range(0, end, downsample*Slen):
			raw_data = new_array[start_point:(start_point+Slen)]
			Y = np.fft.fft(raw_data, Slen)
			P1 = np.fft.fftshift(Y)
			P2 = np.absolute(P1)
			amp = 10*np.log10(P2/Slen)
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
			output_data = amp.reshape((-1, Slen))
			# output_data = np.column_stack((amp, phase))

			# f_r = open(raw_filename,'a')
			# np.savetxt(f_r,raw_data)
			# f_r.close()

			#save fft amplitude 
			f_w = open(w_filename, 'a')
			np.savetxt(f_w, output_data, fmt='%6.3f')
			f_w.close()

		# meaning this is the last block
		if a < count//2:
			break
	print('{0} done').format(filename)
	print('nan count: {0}-{1}').format(nan_count_a, nan_count_p)
	print("dataset size: {0}").format(block_count)

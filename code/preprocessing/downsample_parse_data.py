"""
	Code by Zhujun
"""
import numpy as np
import sys
import os
import glob
import multiprocessing as mp

print('Start downsample...')
path = sys.argv[1] # path to raw data, e.g. /net/adv_spectrum/data/raw/abnormal/0208_anomaly
downsample = int(sys.argv[2])  #downsample rate
core = int(sys.argv[3])
# band = sys.argv[2]
# samp_rate = sys.argv[3]
# loc = sys.argv[4]

# create a folder to save downsampled data
print('Start creating folder...')
downsample_str = 'downsample_' + str(downsample)
output_path = path.replace('raw', 'downsample/' + downsample_str + '/') # e.g. 
if not os.path.exists(output_path):
	os.mkdir(output_path)
	print(output_path + 'is created')

files = glob.glob(path + '/*.dat')
file_num = np.shape(files)[0]
file_per_core = file_num // core

def down_sample(index):
	print('Start downsampling...')

	filenames = [files[i*core + index] for i in range(file_per_core)]
	for filename in filenames:
		# for IQ data, we compute the FFT amplitude, and save it in txt file
		w_filename = filename.replace('.dat', '_ap.txt')
		w_filename = w_filename.replace(path, output_path)
		print('{0} starts pre-processing.'.format(filename))

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
		print('{0} done'.format(filename))
		print('nan count: {0}-{1}'.format(nan_count_a, nan_count_p))
		print("dataset size: {0}".format(block_count))


def multicore_downsample(core, index):
	## run faster using multiprocessing on multicores of CPU
	pool = mp.Pool(processes = core)
	pool.map(down_sample, index)
	return

if __name__ == "__main__":
	# multicore run downsampling
    multicore_downsample(core, range(core))
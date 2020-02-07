"""
Title: featurization.py
Prescription: Process downsampled data into features
"""

import utils
import os
import sys
import glob
import warnings
import numpy as np
import multiprocessing as mp

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##########################################################
# 1. Initialization
##########################################################
# Arguments
downsample_ratio = str(sys.argv[1])  # e.g. 10
window_size = int(sys.argv[2])  # e.g. 100
predict_size = int(sys.argv[3])  # e.g. 25
folder = str(sys.argv[4])  # e.g. ryerson, or 0208_anomaly
data_type = str(sys.argv[5])  # e.g. normal, or abnormal
core = int(sys.argv[6])

# String variables
downsample_str = 'downsample_' + str(downsample_ratio)
window_predict_size = str(sys.argv[2]) + '_' + str(sys.argv[3])

# General path
path = '/net/adv_spectrum/data/'

# Input path of downsampled txt
raw_input_path = path + 'downsample/{}/{}/{}/'\
                 .format(downsample_str, data_type, folder)

# Output path of featurized txt
feature_output_path = raw_input_path.replace('/downsample/', '/feature/') \
                     + window_predict_size + '/'

# Check path existence
if not os.path.exists(feature_output_path):
    os.makedirs(feature_output_path)


##########################################################
# 2. Featurization for normal and abnormal data
##########################################################
print('start processing {} data....'.format(data_type))

files = sorted(glob.glob(raw_input_path + '*.txt'))
file_num = np.shape(files)[0]
file_per_core = file_num // core

def featurization(index):
    filenames = [files[i*core + index] for i in range(file_per_core)]
    for filename in filenames:
        out = feature_output_path + 'feature_' + os.path.basename(filename)
        print(filename, out)
        utils.extract_method3(filename, open(out, 'w'), window_size, predict_size)

    print('Featurization finished! : )')


def multicore_featurization(core, index):
	## run faster using multiprocessing on multicores of CPU
	pool = mp.Pool(processes = core)
	pool.map(featurization, index)
	return

if __name__ == "__main__":
	# multicore run downsampling
    multicore_featurization(core, range(core))
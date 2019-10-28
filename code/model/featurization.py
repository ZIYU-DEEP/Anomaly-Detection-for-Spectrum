"""
Title: featurization.py
Prescription: Process downsampled data into features
"""

import utils
import os
import sys
import glob


##########################################################
# 1. Initialization
##########################################################
# Arguments
downsample_ratio = str(sys.argv[1])  # e.g. 10
window_size = int(sys.argv[2])  # e.g. 100
predict_size = int(sys.argv[3])  # e.g. 25
normal_folder = str(sys.argv[4]) # e.g. ryerson
anomaly_folder = str(sys.argv[5])  # e.g. 0208_anomaly

# String variables
downsample_str = 'downsample_' + str(downsample_ratio)
window_predict_size = str(sys.argv[2]) + '_' + str(sys.argv[3])

# General path
path = '/net/adv_spectrum/data/'

# Input path of downsampled txt
normal_input_path = path + 'downsample/{}/normal/{}/'.format(downsample_str, 
                                                             normal_folder) 
abnormal_input_path = path + 'downsample/{}/abnormal/{}/'.format(downsample_str, 
                                                                 anomaly_folder)

# Output path of featurized txt
normal_output_path = normal_input_path.replace('/downsample/', '/feature/') \
                     + window_predict_size + '/'
abnormal_output_path = abnormal_input_path.replace('/downsample/',
                                                   '/feature/') \
                       + window_predict_size + '/'

# Check path existence
if not os.path.exists(normal_output_path):
    os.makedirs(normal_output_path)
if not os.path.exists(abnormal_output_path):
    os.makedirs(abnormal_output_path)


##########################################################
# 2. Featurization for normal and abnormal data
##########################################################
print('start processing normal data....')
for filename in sorted(glob.glob(normal_input_path + '*.txt')):
    out = normal_output_path + 'feature_' + os.path.basename(filename)
    print(filename, out)
    utils.extract_method3(filename, open(out, 'w'), window_size, predict_size)

print('start processing abnormal data....')
for filename in sorted(glob.glob(abnormal_input_path + '*.txt')):
    out = abnormal_output_path + 'feature_' + os.path.basename(filename)
    print(filename, out)
    utils.extract_method3(filename, open(out, 'w'), window_size, predict_size)

print('Featurization finished! : )')

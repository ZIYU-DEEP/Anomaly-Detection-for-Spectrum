"""
Title: featurization.py
Prescription: Process downsampled data into features
Author: Yeol Ye
"""

import utils
import os
import sys
import pickle
import glob

downsample_ratio = str(sys.argv[1])  # e.g. 10
window_size = int(sys.argv[2])  # e.g. 100
predict_size = int(sys.argv[3])  # e.g. 25
normal_folder = str(sys.argv[4]) # e.g. ryerson
anomaly_folder = str(sys.argv[5])  # e.g. 0208_anomaly
anomaly_power = str(sys,argv[6])
downsample_str = 'downsample_' + str(downsample_ratio)
window_predict_size = str(sys.argv[2]) + '_' + str(sys.argv[3])


path = '/net/adv_spectrum/data/'
normal_input_path = path + 'downsample/{}/normal/{}/'.format(downsample_str, 
                                                             normal_folder) 
abnormal_input_path = path + 'downsample/{}/abnormal/{}/'.format(downsample_str, 
                                                                 anomaly_folder)
normal_output_path = normal_input_path.replace('downsample', 'feature') \
                     + window_predict_size + '/'
 
abnormal_output_path = abnormal_input_path.replace('downsample', 'feature') \
                     + window_predict_size + '/'
normal_series_list_path = normal_output_path \
                          + 'normal_series_list_' + window_predict_size
abnormal_series_list_path = abnormal_input_path \
                            + 'abnormal_series_list_' + window_predict_size



if not os.path.exists(normal_output_path):
    os.makedirs(normal_output_path)
if not os.path.exists(abnormal_output_path):
    os.makedirs(abnormal_output_path)

##########################################################
# Featurization for normal and abnormal data
##########################################################
print('start processing normal data....')
for filename in sorted(glob.glob(normal_input_path + '*.txt')):
    print(filename, out)
    out = normal_output_path + 'feature_' + os.path.basename(filename)
    utils.extract_method3(filename, open(out, 'w'), window_size, predict_size)



print('start processing abnormal data....')
for filename in sorted(glob.glob(abnormal_output_path + '*.txt')):
    print(filename, out)
    out = normal_output_path + 'feature_' + os.path.basename(filename)
    utils.extract_method3(filename, open(out, 'w'), window_size, predict_size)



##########################################################
# Construct normal series and abnormal series
##########################################################
# normal_series_list = []
# abnormal_series_list = []

# for file in os.listdir(normal_output_path):
#     if file == 'feature_.DS_Store' or file == 'feature_._.DS_Store':
#         continue
#     print('Processing: ', file)
#     filename = normal_output_path + file
#     series = utils.txt_to_series(filename)
#     normal_series_list.append(series)


# for file in sorted(os.listdir(abnormal_output_path)):
#     if file == 'feature_.DS_Store' or file == 'feature_._.DS_Store':
#         continue
#     print('Processing: ', file)
#     filename = abnormal_output_path + file
#     series = utils.txt_to_series(filename)
#     abnormal_series_list.append(series)

# temp = abnormal_series_list.pop(0)
# abnormal_series_list.append(temp)


# ##########################################################
# # Save normal series list and abnormal series list
# ##########################################################
# with open(normal_series_list_path, 'wb') as f:
#     pickle.dump(normal_series_list, f)

# with open(abnormal_series_list_path, 'wb') as f:
#     pickle.dump(abnormal_series_list, f)

print('Featurization finished! : )')

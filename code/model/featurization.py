"""
Title: featurization.py
Prescription: Process downsampled data into features
Author: Yeol Ye
"""

import utils
import os
import sys
import pickle

downsample_ratio = str(sys.argv[1])  # e.g. downsample_10
window_size = int(sys.argv[2])  # e.g. 100
predict_size = int(sys.argv[3])  # e.g. 25
anomaly_folder = str(sys.argv[4])  # e.g. 0208_anomaly

downsample_str = 'downsample_' + str(downsample_ratio)
window_predict_size = str(sys.argv[2]) + '_' + str(sys.argv[3])
normal_input_path = '../../data/source/normal/' + downsample_str + '/'
normal_output_path = '../../data/processed/normal/' + downsample_str + '/' \
                     + window_predict_size + '/'
abnormal_input_path = '../../data/source/anomaly/' + anomaly_folder + '/' \
                      + downsample_str + '/'
abnormal_output_path = '../../data/processed/anomaly/' + anomaly_folder + '/' \
                       + downsample_str + '/' + window_predict_size + '/'
normal_series_list_path = '../../data/dataset/' + downsample_str + '/' \
                          + 'normal_series_list_' + window_predict_size
abnormal_series_list_path = '../../data/dataset/' + downsample_str + '/' \
                            + 'abnormal_series_list_' + window_predict_size


##########################################################
# Featurization for normal and abnormal data
##########################################################
for file in os.listdir(normal_input_path):
    if file == '.DS_Store' or file == '._.DS_Store':
        continue
    filename = normal_input_path + file
    out = normal_output_path + 'feature_' + file
    print(out)
    utils.extract_method3(filename, open(out, 'w'), window_size, predict_size)


for file in os.listdir(abnormal_input_path):
    if file == '.DS_Store' or file == '._.DS_Store':
        continue
    filename = abnormal_input_path + file
    out = abnormal_output_path + 'feature_' +  file
    print(out)
    utils.extract_method3(filename, open(out, 'w'), window_size, predict_size)


##########################################################
# Construct normal series and abnormal series
##########################################################
normal_series_list = []
abnormal_series_list = []

for file in os.listdir(normal_output_path):
    if file == 'feature_.DS_Store' or file == 'feature_._.DS_Store':
        continue
    print('Processing: ', file)
    filename = normal_output_path + file
    series = utils.txt_to_series(filename)
    normal_series_list.append(series)


for file in sorted(os.listdir(abnormal_output_path)):
    if file == 'feature_.DS_Store' or file == 'feature_._.DS_Store':
        continue
    print('Processing: ', file)
    filename = abnormal_output_path + file
    series = utils.txt_to_series(filename)
    abnormal_series_list.append(series)

temp = abnormal_series_list.pop(0)
abnormal_series_list.append(temp)


##########################################################
# Save normal series list and abnormal series list
##########################################################
with open(normal_series_list_path, 'wb') as f:
    pickle.dump(normal_series_list, f)

with open(abnormal_series_list_path, 'wb') as f:
    pickle.dump(abnormal_series_list, f)

print('Featurization finished! : )')

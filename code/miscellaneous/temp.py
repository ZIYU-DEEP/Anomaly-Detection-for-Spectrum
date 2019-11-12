"""
Title: evaluation.py
Prescription: Evaluate the model's anomaly detection performance on different
              anomaly inputs.
"""
import warnings

warnings.filterwarnings('ignore')

from timeit import default_timer as timer
import utils
import os
import glob
import pickle
import sys
import matplotlib
import joblib
from functools import reduce
import operator
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##########################################################
# 1. Initialization
##########################################################
# Arguments
downsample_ratio = int(sys.argv[1])
window_size = int(sys.argv[2])
predict_size = int(sys.argv[3])
normal_folder = str(sys.argv[4])  # e.g. ryerson
anomaly_folder = str(sys.argv[5])  # e.g. 0208_anomaly
shift_eval = int(sys.argv[6])
batch_size = int(sys.argv[7])
gpu_no = str(sys.argv[8])

# Interval values
all_samp = 200000000 * 2  # all samp per file
samp_sec = 5000000 * 2  # sample rate, each sample has I/Q 2 values
interval = 5  # in seconds
inter_samp = samp_sec * interval / downsample_ratio  # in samp number
trash_count = 102400  # begining samples being throwed

# calculate the intervals in prediction window
ini_anom = int((((inter_samp - trash_count) / 256) - window_size)
               // shift_eval)
anom_interval = int(((inter_samp / 256) - window_size) // shift_eval)
up_down_interval = 10 #int(window_size // shift_eval)

# Set gpu environment
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

# String variables
downsample_str = 'downsample_' + str(downsample_ratio)
window_predict_size = str(sys.argv[2]) + '_' + str(sys.argv[3])

# General path
path = '/net/adv_spectrum/data/'

# Path to read featurized txt
abnormal_output_path = path + 'feature/{}/abnormal/{}/{}/' \
    .format(downsample_str,
            anomaly_folder,
            window_predict_size)

error_df_path = '/net/adv_spectrum/result/error_df/anomaly/{}/{}/' \
    .format(downsample_str, anomaly_folder)
full_anom_error_df_list_filename = error_df_path + \
                                   'full_anom_error_df_list_{}_{}_{}.pkl' \
                                       .format(normal_folder,
                                               window_predict_size,
                                               shift_eval)
model_path = '/net/adv_spectrum/model/{}/{}/' \
    .format(downsample_str, normal_folder)
model_filename = model_path + '{}_{}.h5' \
    .format(downsample_ratio, window_predict_size)

anom_seq_filename = '/net/adv_spectrum/miscellaneous/anom_seq_{}.pkl'\
                    .format(window_predict_size)

##########################################################
# 2. Load Model and Data
##########################################################
model = tf.keras.models.load_model(model_filename)

# Change the path if you need other abnormal series. Be sure the series are
# stored in a format of list of arrays (shape = [n, 128]).
abnormal_series_list = []

print('Start retrieving abnormal series....')
for filename in sorted(glob.glob(abnormal_output_path + '*.txt')):
    print(filename)
    series = utils.txt_to_series(filename)
    abnormal_series_list.append(series)

##########################################################
# 4. Construct MSE DataFrame for Full Anom Data
##########################################################
# Construct MSE DataFrame
anom_hat_list = [utils.model_forecast(model, i, batch_size, window_size,
                                      predict_size, shift_eval)
                                      .reshape(-1, shift_eval, 128)
                 for i in abnormal_series_list]

if shift_eval == predict_size + window_size:
    anom_true_list = [utils.windowed_true(i, shift_eval, predict_size)
                      for i in abnormal_series_list]
else:
    anom_true_list = [i[window_size:, :].reshape((-1, shift_eval, 128))
                      for i in abnormal_series_list]

##########################################################
# 5. Construct Normal and Abnormal MSE from combined data
##########################################################
# MSE of full data
full_anom_error_df_list = []

for i in range(len(anom_hat_list)):
    print('Processing the {} th anom hat list!'.format(i))
    anom_hat = anom_hat_list[i]
    anom_true = anom_true_list[i]
    mse = np.mean(np.power(anom_hat - anom_true, 2), axis=(1,2))
    print(np.shape(mse))

    # Get full anom error
    full_anom_error_df = pd.DataFrame({'full_anom_error ' + str(i): mse})
    full_anom_error_df_list.append(full_anom_error_df)

    # Get nom, anom, anom_up, anom_down
    cycle = int(all_samp / (2 * samp_sec * interval))
    an_interval = (np.shape(mse)[0] - ini_anom) // 7 + 1

    anom_seq = [[5] * ini_anom]
    for k in range(cycle):
        anom_seq.append([6] * an_interval)
        if k != cycle - 1:
            anom_seq.append([5] * an_interval)
    anom_seq = reduce(operator.add, anom_seq)
    if len(anom_seq) > len(full_anom_error_df):
        anom_seq = anom_seq[0:len(full_anom_error_df)]
    else:
        anom_seq = [anom_seq, [6] * (len(full_anom_error_df) - len(anom_seq))]
        anom_seq = reduce(operator.add, anom_seq)

    with open(anom_seq_filename, 'wb') as f:
        joblib.dump(anom_seq, anom_seq_filename)

    print('Done!')




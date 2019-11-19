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
trash_count = 102400 / downsample_ratio  # begining samples being throwed

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

# Path to save model and full_x_valid
full_x_valid_path = '/net/adv_spectrum/result/x_valid/'
full_x_valid_filename = full_x_valid_path + 'full_x_valid_{}_{}_{}.pkl' \
    .format(downsample_str,
            normal_folder,
            window_predict_size)
model_path = '/net/adv_spectrum/model/{}/{}/' \
    .format(downsample_str, normal_folder)
model_filename = model_path + '{}_{}.h5' \
    .format(downsample_ratio, window_predict_size)
model_info_filename = model_path + '{}_{}_{}_info.txt' \
    .format(anomaly_folder,
            downsample_ratio,
            window_predict_size)
model_size = os.path.getsize(model_filename)

# Path to save valid error df and list of anomaly error df
valid_error_df_path = '/net/adv_spectrum/result/error_df/valid/' \
                      '{}/{}/'.format(downsample_str, normal_folder)
valid_error_df_filename = valid_error_df_path + 'valid_error_df_{}_{}.pkl' \
    .format(normal_folder, window_predict_size)

error_df_path = '/net/adv_spectrum/result/error_df/anomaly/{}/{}/' \
    .format(downsample_str, anomaly_folder)
full_anom_error_df_list_filename = error_df_path + \
                                   'full_anom_error_df_list_{}_{}_{}.pkl' \
                                       .format(normal_folder,
                                               window_predict_size,
                                               shift_eval)
anom_error_df_filename = error_df_path + \
                         'anom_error_df_{}_{}_{}.pkl' \
                             .format(normal_folder,
                                     window_predict_size,
                                     shift_eval)
nom_error_df_filename = anom_error_df_filename.replace('anom_', 'nom_')
anom_up_error_df_filename = anom_error_df_filename.replace('anom_', 'anom_up_')
anom_down_error_df_filename = anom_up_error_df_filename.replace('up', 'down')

# Path of figure
figure_CDF_name = '[Anomaly v.s. Valid] CDF Plot for Prediction Error ' \
                  '(norm = {}, anom = {}, ds_ratio={}, w_p_size={})' \
    .format(normal_folder, anomaly_folder,
            downsample_ratio, window_predict_size)
figure_CDF_path = '/net/adv_spectrum/result/plot/CDF/'
figure_time_path = '/net/adv_spectrum/result/plot/time_mse/'
figure_CDF_filename = figure_CDF_path + 'CDF_plot_{}_{}_{}_{}_{}.png' \
    .format(normal_folder,
            anomaly_folder,
            downsample_ratio,
            window_predict_size,
            shift_eval)

# Check path existence
if not os.path.exists(full_x_valid_path):
    os.makedirs(full_x_valid_path)
if not os.path.exists(valid_error_df_path):
    os.makedirs(valid_error_df_path)
if not os.path.exists(error_df_path):
    os.makedirs(error_df_path)
if not os.path.exists(figure_CDF_path):
    os.makedirs(figure_CDF_path)
if not os.path.exists(figure_time_path):
    os.makedirs(figure_time_path)

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

# Comment out the following operation if you do not need validation data
with open(full_x_valid_filename, 'rb') as f:
    full_x_valid = joblib.load(f)

# Comment out the next section if you do not need validation data
##########################################################
# 3. Construct MSE DataFrame for Validation Data
##########################################################
print('Start constructing mse DataFrame...')
# Get valid_hat and evaluate time
start = timer()
valid_hat = utils.model_forecast(model, full_x_valid, batch_size, window_size,
                                 predict_size, shift_eval).reshape(-1, shift_eval, 128)
end = timer()
validation_time = (start - end) / (len(full_x_valid) // shift_eval)
print('Validation spends {} seconds! Hmm...'.format(validation_time))

# Get valid true
if shift_eval == predict_size + window_size:
    valid_true = utils.windowed_true(full_x_valid, shift_eval, predict_size)
else:
    valid_true = full_x_valid[window_size:, :].reshape((-1, shift_eval, 128))

# Create mse DataFrame
valid_mse = np.mean(np.power(valid_hat - valid_true, 2), axis=(1, 2))
valid_error_df = pd.DataFrame({'valid_error': valid_mse})

# Save MSE DataFrame
valid_error_df.to_pickle(valid_error_df_filename)

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
# MSE of intervals with only BS data in it
nom_mse = []
nom_error_df_pd = pd.DataFrame()

# MSE of intervals with BS + 'FBS' data in it
anom_mse = []
anom_error_df_pd = pd.DataFrame()

# MSE of moments when 'FBS' is just on
anom_up_mse = []
anom_up_error_df_pd = pd.DataFrame()

# MSE of moments when 'FBS' is just off
anom_down_mse = []
anom_down_error_df_pd = pd.DataFrame()

# MSE of full data
full_anom_error_df_list = []

anom_seq_list = []
mse_list = []

for i in range(len(anom_hat_list)):
    print('Processing the {} th anom hat list!'.format(i))
    nom_mse = []
    anom_mse = []
    anom_up_mse = []
    anom_down_mse = []
    anom_hat = anom_hat_list[i]
    anom_true = anom_true_list[i]
    mse = np.mean(np.power(anom_hat - anom_true, 2), axis=(1,2))
    #mse_list.append(mse)
    #print(np.shape(mse))
    #print(np.shape(mse_list), i)

    # Get full anom error
    full_anom_error_df = pd.DataFrame({'full_anom_error ' + str(i): mse})
    full_anom_error_df_list.append(full_anom_error_df)

    # Get nom, anom, anom_up, anom_down
    nom_mse = [mse[0: (ini_anom - up_down_interval)]]
    print('nom_mse:', 0, ini_anom - up_down_interval)
    cycle = int(all_samp / (2 * samp_sec * interval))
    an_interval = (np.shape(mse)[0] - ini_anom) // 7 + 1

    for j in range(4):
        if j != 0:
            nom_mse.append(mse[ini_anom + an_interval * (2 * j - 1)
                               + up_down_interval:
                               ini_anom + an_interval * (2 * j)
                               - up_down_interval])
            print('nom_mse:', ini_anom + an_interval * (2 * j - 1)+ up_down_interval, ini_anom + an_interval * (2 * j) - up_down_interval)
        anom_mse.append(mse[ini_anom + (2 * j) * an_interval
                            + up_down_interval:
                            ini_anom + (2 * j + 1) * an_interval
                            - up_down_interval])
        print('anom_mse:', ini_anom + an_interval * (2 * j) + up_down_interval, ini_anom + an_interval * (2*j+1) - up_down_interval)
        anom_up_mse.append(mse[ini_anom + an_interval * (2 * j)
                               - up_down_interval:
                               ini_anom + (2 * j) * an_interval + up_down_interval])
        print('anom_up_mse', ini_anom + an_interval * (2 * j)- up_down_interval, ini_anom + (2 * j) * an_interval + up_down_interval)
        anom_down_mse.append(mse[ini_anom + (2 * j + 1) * an_interval
                                 - up_down_interval:
                                 ini_anom + (2 * j + 1) * an_interval + up_down_interval])
        print('anom_down_mse', ini_anom + an_interval * (2 * j+1)- up_down_interval, ini_anom + (2 * j +1) * an_interval + up_down_interval)

    nom_mse = [l.tolist() for l in nom_mse]
    nom_mse = reduce(operator.add, nom_mse)
    nom_error_df = pd.DataFrame({'nom_error ': nom_mse})
    nom_error_df_pd = nom_error_df_pd.append(nom_error_df)

    anom_mse = [l.tolist() for l in anom_mse]
    anom_mse = reduce(operator.add, anom_mse)
    anom_error_df = pd.DataFrame({'anom_error ': anom_mse})
    anom_error_df_pd = anom_error_df_pd.append(anom_error_df)

    anom_up_mse = [l.tolist() for l in anom_up_mse]
    anom_up_mse = reduce(operator.add, anom_up_mse)
    anom_up_error_df = pd.DataFrame({'anom_up_error ': anom_up_mse})
    anom_up_error_df_pd = anom_up_error_df_pd.append(anom_up_error_df)

    anom_down_mse = [l.tolist() for l in anom_down_mse]
    anom_down_mse = reduce(operator.add, anom_down_mse)
    anom_down_error_df = pd.DataFrame({'anom_down_error ': anom_down_mse})
    anom_down_error_df_pd = anom_down_error_df_pd.append(anom_down_error_df)

    print(np.shape(nom_mse), np.shape(anom_mse), np.shape(anom_up_mse), np.shape(anom_down_mse))

    anom_seq = [[2] * ini_anom]
    #anom_seq = [[5] * int((inter_samp - trash_count) / 256)]
    #an_interval = (np.shape(mse)[0] - ini_anom) // 7
    for k in range(cycle):
        anom_seq.append([2.5] * an_interval)
        if k != cycle - 1:
            anom_seq.append([2] * an_interval)
    anom_seq = reduce(operator.add, anom_seq)
    if len(anom_seq) > len(full_anom_error_df):
        anom_seq = anom_seq[0:len(full_anom_error_df)]
    else:
        anom_seq = [anom_seq, [2.5]* (len(full_anom_error_df) - len(anom_seq))]
        anom_seq = reduce(operator.add, anom_seq)

    #print(np.shape(anom_seq))
    anom_seq_list.append(anom_seq)
    mse_list.append(mse.tolist())
    #print(np.shape(mse))
    print('mse_list:', np.shape(mse_list), i)
    print('anom_seq_list', np.shape(anom_seq_list), i)
    #mse_list.append(mse)
    # Draw the i th time mse of full anom error
print('Drawing the full anom time mse plot!')
anom_seq_list = reduce(operator.add, anom_seq_list)
mse_list = reduce(operator.add, mse_list)
print('anom_seq_list:', np.shape(anom_seq_list))
print('mse_list shape:', np.shape(mse_list))
chunk = int(len(mse_list) / 4)
plt.figure()
for i  in range(4):
    plt.subplot(4, 1, i+1)
    ax = sns.lineplot(x=range(chunk),
                      y=anom_seq_list[chunk * i:chunk * (i+1)])
    ax = sns.lineplot(x=range(chunk),
                      y=mse_list[chunk * i:chunk * (i+1)], color='orange')
    #plt.title('Block' + str(i*23) + ' to block' + str((i+1)*23))
    plt.ylim(top=3)
    plt.ylim(bottom=0)

plt.xlabel('Time')
plt.ylabel('MSE')
sns.despine()
figure_time_name = 'time_mse_{}_{}_{}_{}_{}.png' \
    .format(normal_folder, anomaly_folder,
            downsample_ratio, window_predict_size,
            shift_eval)
figure_time_filename = figure_time_path + figure_time_name
ax.get_figure().savefig(figure_time_filename, dpi =1200)
print(figure_time_filename, 'is saved')

# Save MSE DataFrame
print('Saving the strange mse DataFrames!')
with open(full_anom_error_df_list_filename, 'wb') as f:
    joblib.dump(full_anom_error_df_list, f)

with open(nom_error_df_filename, 'wb') as f:
    joblib.dump(nom_error_df_pd, f)

with open(anom_error_df_filename, 'wb') as f:
    joblib.dump(anom_error_df_pd, f)

with open(anom_up_error_df_filename, 'wb') as f:
    joblib.dump(anom_up_error_df_pd, f)

with open(anom_down_error_df_filename, 'wb') as f:
    joblib.dump(anom_down_error_df_pd, f)

# Comment out the following section if you do not need visualization
##########################################################
# 5. Plot CDF for Anomaly and Validation Set
##########################################################
print('Start plotting CDF...')
# If you need other normal data as baseline, be sure to change the following
# line to the path you desired.
with open(valid_error_df_filename, 'rb') as f:
    valid_error_df = pickle.load(f)

# Draw plot
sns.set_style('white')
plt.figure(figsize=(23, 6))
ax = sns.kdeplot(valid_error_df['valid_error'], cumulative=True, shade=False,
                 color='r')

ax = sns.kdeplot(nom_error_df_pd['nom_error '],
                 cumulative=True, shade=False, color='g')
ax = sns.kdeplot(anom_error_df_pd['anom_error '],
                 cumulative=True, shade=False, color='b')
ax = sns.kdeplot(anom_up_error_df_pd['anom_up_error '],
                 cumulative=True, shade=False, color='y')
ax = sns.kdeplot(anom_down_error_df_pd['anom_down_error '],
                 cumulative=True, shade=False, color='k')

sns.despine()
ax.hlines(0.9, ax.get_xlim()[0], ax.get_xlim()[1], colors="blue", zorder=100,
          label='10% FP Threshold', linestyles='dashdot')
ax.hlines(0.8, ax.get_xlim()[0], ax.get_xlim()[1], colors="purple", zorder=100,
          label='20% FP Threshold', linestyles='dotted')
ax.vlines(valid_error_df.quantile(0.8)[0], ymin=0, ymax=0.8, color='purple',
          linestyles='dotted')
ax.vlines(valid_error_df.quantile(0.9)[0], ymin=0, ymax=0.9, color='blue',
          linestyles='dashdot')
ax.set_title(figure_CDF_name)
ax.set_xlim(left=0, right=3)

plt.legend(loc=4)
plt.xlabel('Value of prediction error')
plt.ylabel('Cumulative probability')
ax.get_figure().savefig(figure_CDF_filename)
print(figure_CDF_filename)

tf.keras.backend.clear_session()
print('Evaluation finished!')

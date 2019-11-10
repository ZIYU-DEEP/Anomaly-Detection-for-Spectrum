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
import joblib
import sys
import matplotlib
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
gpu_no = int(sys.argv[8])

# Set gpu environment
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[gpu_no - 1], 'GPU')
logical_devices = tf.config.experimental.list_logical_devices('GPU')

# String variables
downsample_str = 'downsample_' + str(downsample_ratio)
window_predict_size = str(sys.argv[2]) + '_' + str(sys.argv[3])

# General path
path = '/net/adv_spectrum/data/'

# Path to read featurized txt
abnormal_output_path = path + 'feature/{}/abnormal/{}/{}/'\
                    .format(downsample_str, anomaly_folder, window_predict_size)

# Path to save model and full_x_valid
full_x_valid_path = '/net/adv_spectrum/result/x_valid/'
full_x_valid_filename = full_x_valid_path + 'full_x_valid_{}_{}_{}.pkl'\
                        .format(downsample_str,
                                normal_folder,
                                window_predict_size)
model_path = '/net/adv_spectrum/model/{}/{}/'\
             .format(downsample_str, normal_folder)
model_filename = model_path + '{}_{}.h5'\
                 .format(downsample_ratio, window_predict_size)
model_info_filename = model_path + '{}_{}_{}_info.txt'\
                      .format(anomaly_folder,
                              downsample_ratio,
                              window_predict_size)
model_size = os.path.getsize(model_filename)

# Path to save valid error df and list of anomaly error df
valid_error_df_path = '/net/adv_spectrum/result/error_df/valid/' \
                      '{}/{}/'.format(downsample_str, normal_folder)
valid_error_df_filename = valid_error_df_path + 'valid_error_df_{}_{}.pkl'\
                          .format(normal_folder, window_predict_size)

anom_error_df_list_path = '/net/adv_spectrum/result/error_df/anomaly/' \
                          '{}/{}/'.format(downsample_str, anomaly_folder)
anom_error_df_list_filename = anom_error_df_list_path + \
                              'anom_error_df_{}_{}.pkl'\
                              .format(normal_folder, window_predict_size)

# Path of figure
figure_name = '[Anomaly v.s. Valid] CDF Plot for Prediction Error ' \
              '(ds_ratio={}, w_p_size={})'\
              .format(downsample_ratio, window_predict_size)
figure_path = '/net/adv_spectrum/result/plot/'
figure_filename = figure_path + 'CDF_plot_{}_{}_{}_{}.png'\
                  .format(normal_folder,
                          anomaly_folder,
                          downsample_ratio,
                          window_predict_size)

# Check path existence
if not os.path.exists(full_x_valid_path):
    os.makedirs(full_x_valid_path)
if not os.path.exists(valid_error_df_path):
    os.makedirs(valid_error_df_path)
if not os.path.exists(anom_error_df_list_path):
    os.makedirs(anom_error_df_list_path)
if not os.path.exists(figure_path):
    os.makedirs(figure_path)


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
                                 predict_size, shift_eval).reshape(-1, 128)
end = timer()
validation_time = (start - end) / (len(full_x_valid) // shift_eval)
print('Validation spends {} seconds! Hmm...'.format(validation_time))

# Get valid true
valid_true = utils.windowed_true(full_x_valid, shift_eval, predict_size)

# Create mse DataFrame
valid_mse = np.mean(np.power(valid_hat - valid_true, 2), axis=1)
valid_error_df = pd.DataFrame({'valid_error': valid_mse})

# Save MSE DataFrame
valid_error_df.to_pickle(valid_error_df_filename)


##########################################################
# 4. Construct Abnormal MSE
##########################################################
# Construct MSE DataFrame
anom_hat_list = [utils.model_forecast(model, i, batch_size, window_size,
                                      predict_size, shift_eval).reshape(-1, 128)
                 for i in abnormal_series_list]
anom_true_list = [utils.windowed_true(i, shift_eval, predict_size)
                  for i in abnormal_series_list]

anom_mse_list = []
anom_error_df_list = []

for i in range(len(anom_hat_list)):
    anom_hat = anom_hat_list[i]
    anom_true = anom_true_list[i]
    anom_mse = np.mean(np.power(anom_hat - anom_true, 2), axis=1)
    anom_error_df = pd.DataFrame({'anom_error ' + str(i): anom_mse})
    anom_mse_list.append(anom_mse)
    anom_error_df_list.append(anom_error_df)

# Save MSE DataFrame
with open(anom_error_df_list_filename, 'wb') as f:
    pickle.dump(anom_error_df_list, f)


# Comment out the following section if you do not need visualization
##########################################################
# 5. Plot CDF for Anomaly and Validation Set
##########################################################
print('Start plotting...')
# If you need other normal data as baseline, be sure to change the following
# line to the path you desired.
with open(valid_error_df_filename, 'rb') as f:
    valid_error_df = pickle.load(f)

# Draw plot
sns.set_style('white')
plt.figure(figsize=(23, 6))
ax = sns.kdeplot(valid_error_df['valid_error'], cumulative=True, shade=False,
                 color='r')
color_list = list(matplotlib.colors.cnames.items())
for i in range(len(anom_hat_list)):
    ax = sns.kdeplot(anom_error_df_list[i]['anom_error ' + str(i)],
                     cumulative=True, shade=False, color=color_list[i][0])

sns.despine()
ax.hlines(0.9, ax.get_xlim()[0], ax.get_xlim()[1], colors="blue", zorder=100,
          label='10% FP Threshold', linestyles='dashdot')
ax.hlines(0.8, ax.get_xlim()[0], ax.get_xlim()[1], colors="purple", zorder=100,
          label='20% FP Threshold', linestyles='dotted')
ax.vlines(valid_error_df.quantile(0.8)[0], ymin=0, ymax=0.8, color='purple',
          linestyles='dotted')
ax.vlines(valid_error_df.quantile(0.9)[0], ymin=0, ymax=0.9, color='blue',
          linestyles='dashdot')
ax.set_title(figure_name)
ax.set_xlim(left=0, right=4)

plt.legend(loc=4)
plt.xlabel('Value of prediction error')
plt.ylabel('Cumulative probability')
ax.get_figure().savefig(figure_filename)


##########################################################
# 6. Print FP rate v.s. Detection rate
##########################################################
# Modify the following quantile to see different outcomes.
cut = valid_error_df.quantile(0.9)[0]
i = 0
print('False Positive Rate: 10%')

# Write relevant information
f = open(model_info_filename, 'w')
f.write('Model Info filename: {}\n'.format(model_info_filename))
f.write('Model size: {}\n'.format(model_size))
f.write('Validation time: {}'.format(validation_time))
for df in anom_error_df_list:
    y = [1 if e > cut else 0 for e in df['anom_error ' + str(i)].values]
    detect_rate = sum(y) / len(y)
    detect_str = 'Detection rate for anom_error_{} (FP rate = 0.1): {}\n'\
                 .format(i, detect_rate)
    print(detect_str)
    f.write(detect_str)
    i += 1
f.close()

tf.keras.backend.clear_session()
print('Evaluation finished!')

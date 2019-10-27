"""
Title: training.py
Prescription: Training the rnn model.
Author: Yeol Ye
"""

import utils
import pickle
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

downsample_ratio = int(sys.argv[1])
window_size = int(sys.argv[2])
shift_size = int(sys.argv[3])
batch_size = int(sys.argv[4])
anomaly_folder = str(sys.argv[5])
model_filename = str(sys.argv[6])

downsample_str = 'downsample_' + str(downsample_ratio)
window_shift_size = str(sys.argv[2]) + '_' + str(sys.argv[3])
normal_series_list_path = '../../data/dataset/' + downsample_str + '/' \
                          + 'normal_series_list_' + window_shift_size
abnormal_series_list_path = '../../data/dataset/' + downsample_str + '/' \
                            + 'abnormal_series_list_' + window_shift_size
full_x_valid_path = '../../data/dataset/' + downsample_str + '/' \
                    + 'full_x_valid_' + window_shift_size
valid_error_df_path = '../../data/evaluation/' + downsample_str + '/' \
                      + 'valid_error_df_' + window_shift_size
anom_error_df_list_path = '../../data/evaluation/' + anomaly_folder + '/' \
                          + downsample_str + '/' + 'anom_error_df_list_' \
                          + window_shift_size

figure_name = '[Anomaly v.s. Valid] CDF Plot for Prediction Error (ds_ratio=' \
              + str(downsample_ratio) + ', window_size=' + str(window_size) \
              + ', shift_size=' + str(shift_size) + ')'
figure_path = '../../plot/' + figure_name


##########################################################
# Load Model and Data
##########################################################
model = tf.keras.models.load_model(model_filename)

# Change the path if you need other abnormal series. Be sure the series are
# stored in a format of list of arrays (shape = [n, 128]).
with open(abnormal_series_list_path, 'rb') as f:
    abnormal_series_list = pickle.load(f)

# Comment out the following operation if you do not need validation data
with open(full_x_valid_path, 'rb') as f:
    full_x_valid = pickle.load(f)

# Comment out the next section if you do not need validation data
##########################################################
# Construct MSE DataFrame for Validation Data
##########################################################
# Construct MSE DataFrame
valid_hat = utils.model_forecast(model, full_x_valid, batch_size, window_size,
                           shift_size)
valid_true = full_x_valid[window_size:, :].reshape((-1, 25, 128))
valid_mse = np.mean(np.power(valid_hat.reshape(-1, 128) -
                             valid_true.reshape(-1, 128), 2), axis=1)
valid_error_df = pd.DataFrame({'valid_error': valid_mse})

# Save MSE DataFrame
valid_error_df.to_pickle(valid_error_df_path)


##########################################################
# Construct Abnormal MSE
##########################################################
# Construct MSE DataFrame
anom_hat_list = [utils.model_forecast(model, i, batch_size, window_size,
                                      shift_size)
                 for i in abnormal_series_list]
anom_true_list = [i[window_size:, :].reshape((-1, 25, 128))
                  for i in abnormal_series_list]
anom_mse_list = []
anom_error_df_list = []

for i in range(len(anom_hat_list)):
    anom_hat = anom_hat_list[i]
    anom_true = anom_true_list[i]
    anom_mse = np.mean(np.power(anom_hat.reshape(-1, 128)
                                - anom_true.reshape(-1, 128), 2), axis=1)
    anom_error_df = pd.DataFrame({'anom_error ' + str(i): anom_mse})
    anom_mse_list.append(anom_mse)
    anom_error_df_list.append(anom_error_df)

# Save MSE DataFrame
with open(anom_error_df_list_path, 'wb') as f:
    pickle.dump(anom_error_df_list, f)


# Comment out the following section if you do not need visualization
##########################################################
# Plot CDF for Anomaly and Validation Set
##########################################################
# If you need other normal data as baseline, be sure to change the following
# line to the path you desired.
with open(valid_error_df_path, 'rb') as f:
    valid_error_df = pickle.load(f)

# Draw plot
sns.set_style('white')
plt.figure(figsize=(23, 6))
ax = sns.kdeplot(valid_error_df['valid_error'], cumulative=True, shade=False,
                 color='r')
color_list = ['g', 'm', 'yellow', 'black']
for i in range(len(anom_hat_list)):
    ax = sns.kdeplot(anom_error_df_list[i]['anom_error ' + str(i)],
                     cumulative=True, shade=False, color=color_list[i])

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
ax.get_figure().savefig(figure_path)


##########################################################
# Print FP rate v.s. Detection rate
##########################################################
# Modify the following quantile to see different outcomes.
cut = valid_error_df.quantile(0.9)[0]
i = 0
print('False Positive Rate: 10%')

for df in anom_error_df_list:
    y = [1 if e > cut else 0 for e in df['anom_error ' + str(i)].values]
    print('Detection rate for anom_error ' + str(i) + ':', sum(y) / len(y))
    i += 1

print('Evaluation finished!')

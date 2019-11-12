"""
Title: single_time_mse.py
Prescription: This file will give a time_mse plot for a single raw data file.
              You will need to specify the path of the file, and the model you
              would like to use.
"""

import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Downsample and featurization


print('Drawing time mse plot!')
plt.figure(figsize=(23, 6))
ax = sns.lineplot(x=error_df.index,
                  y=error_df.iloc[:, 0], color='orange')

plt.xlabel('Time')
plt.ylabel('MSE')
sns.despine()

figure_time_name = 'time_mse_{}_{}_{}_{}_{}_{}.png' \
    .format(normal_folder, anomaly_folder, i,
            downsample_ratio, window_predict_size,
            shift_eval)
figure_time_filename = figure_time_path + figure_time_name
ax.get_figure().savefig(figure_time_filename)

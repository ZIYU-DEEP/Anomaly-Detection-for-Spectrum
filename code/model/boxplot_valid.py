"""
Title: boxplot_valid.py
Prescription: Draw the boxplot for multiple valid data.
"""

import os
import sys
import glob
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

##########################################################
# 1. Construct the List of Valid Error DataFrames
##########################################################
path = '/net/adv_spectrum/result/error_df/valid/'
base_df = pd.DataFrame()
i = 0
for filename in glob.glob(path + '**/**/*.pkl'):
    with open(filename, 'rb') as f:
        df = pickle.load(f)
        basename = os.path.basename(filename)\
                   .replace('valid_error_df_', '').replace('.pkl', '')
        df.rename(columns={'valid_error': basename}, inplace=True)
        if i == 0:
            base_df = df
        else:
            base_df = pd.concat((base_df, df))
        i += 1

##########################################################
# 2. Draw the Figure
##########################################################
sns.set_style('whitegrid')
plt.figure(figsize=(20, 10))
ax = sns.boxplot(data=base_df, showfliers=False, palette='Set2')
ax.set_ylim(bottom=0, top=1.5)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
sns.despine()
ax.get_figure().savefig(path + 'boxplot_valid.png')

print('Finished drawing the figure!')

import numpy as np
import glob
import pickle
import math
from functools import reduce
import operator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


lambda_path = '/home/shinanliu/lambda/'
ry_path = '/net/adv_spectrum/data/downsample/downsample_10/normal/ryerson_all/'
jcl_path = '/net/adv_spectrum/data/downsample/downsample_10/normal/JCL/'
dt_path = '/net/adv_spectrum/data/downsample/downsample_10/normal/downtown/'
mb_path = '/net/adv_spectrum/data/downsample/downsample_10/normal/Mobile/'



def folder_to_series(file_path, label, n_channels=1):
    features = []
    for filename in sorted(glob.glob(file_path + '*.txt')):
        print('Beginning to process ' + filename)
        with open(filename, 'r') as f:
            for line in f:
                x = line.split()
                po = np.mean([float(i) for i in x])
                if po > -50:
                    features.append(po)
    print(np.shape(features))
    print('Series of ' + file_path + ' is already constructed.')
    return pd.DataFrame({'Power of ' + label: features})


def folder_to_control_series(file_path, label, n_channels=1):
    features = []
    i = 0 
    for filename in sorted(glob.glob(file_path + '*.txt')):
        print('Beginning to process ' + filename)
        with open(filename, 'r') as f:
            for line in f:
                x = line.split()
                i += 1
                if i % 40 == 0:
                    features.append(np.mean([float(k) for k in x]))
                # rm = i % select
                # if rm == 0:
                #     rand_select = np.random.randint(1, select)
                #     flag = 1
                # elif rm % rand_select == 0 and flag:
                #     x = [float(i) for i in x]
                #     features.append(x)
                #     flag = 0
    # features = reduce(operator.add, features)
    print(np.shape(features))
    print('Series of ' + file_path + ' is already constructed.')
    return pd.DataFrame({'Power of ' + label: features})


def plot_power_cdf():
    sns.set_style('white')
    plt.figure()

    # label = 'JCL'
    # print('Start processing ' + label)
    # power_df = folder_to_series(jcl_path, label)
    # print('Start plotting power cdf of ' + label)
    # ax = sns.kdeplot(power_df['Power of ' + label], cumulative=True, shade=False,
    #                 color='b')
    # ax.hlines(0.95, ax.get_xlim()[0], ax.get_xlim()[1], colors="blue", zorder=100, 
    #                 linestyles='dashdot')
    # print(power_df.quantile(0.95))
    # ax.vlines(power_df.quantile(0.95)[0], ymin=0, ymax=0.95, color='blue', 
    #                 linestyles='dashdot')

    label = 'Stable' # Ryerson
    power_df = folder_to_series(ry_path, label)
    print('Start plotting power cdf of ' + label)
    ax = sns.kdeplot(power_df['Power of ' + label], cumulative=True, shade=False,
                    color='r')
    ax.hlines(0.95, ax.get_xlim()[0], ax.get_xlim()[1], colors="red", zorder=100, 
                    linestyles='dashdot')
    print(power_df.quantile(0.95))
    ax.vlines(power_df.quantile(0.95)[0], ymin=0, ymax=0.9, color='red', 
                    linestyles='dashdot')

    label = 'Semi-stable' # Downtown
    power_df = folder_to_series(dt_path, label)
    ax = sns.kdeplot(power_df['Power of ' + label], cumulative=True, shade=False,
                    color='g')
    ax.hlines(0.95, ax.get_xlim()[0], ax.get_xlim()[1], colors="green", zorder=100, 
                    linestyles='dashdot')
    print(power_df.quantile(0.95))
    ax.vlines(power_df.quantile(0.95)[0], ymin=0, ymax=0.95, color='green', 
                    linestyles='dashdot')

    label = 'Dynamic' # Mobile, campus_driving
    power_df = folder_to_series(mb_path, label)
    print('Start plotting power cdf of ' + label)
    ax = sns.kdeplot(power_df['Power of ' + label], cumulative=True, shade=False,
                    color='b')
    ax.hlines(0.95, ax.get_xlim()[0], ax.get_xlim()[1], colors="blue", zorder=100, 
                    linestyles='dashdot')
    print(power_df.quantile(0.95))
    ax.vlines(power_df.quantile(0.95)[0], ymin=0, ymax=0.9, color='blue', 
                    linestyles='dashdot')

    ax.set_title('Power CDF')
    plt.legend(loc=2)
    plt.xlabel('Power(in dB)')
    plt.ylabel('CDF')

    ax.get_figure().savefig(lambda_path + '3ob.png', ppi = 1200)
    print(label + 'power_all.png is successfully generated')

plot_power_cdf()
# plot_power_cdf(dt_path, 'downtown')
# plot_power_cdf(jcl_path, 'jcl')
# plot_power_cdf(ry_path, 'ryerson')

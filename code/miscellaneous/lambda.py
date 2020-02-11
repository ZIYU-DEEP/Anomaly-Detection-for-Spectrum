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
all_normal_path = glob.glob('/net/adv_spectrum/data/downsample/downsample_10/normal/*/')
all_abnormal_path = glob.glob('/net/adv_spectrum/data/downsample/downsample_10/abnormal/*/')

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


def plot_power_cdf(path_list, fig_name):
    sns.set_style('white')
    plt.figure()

    for path in path_list:
        label = path.split('/')[-2] # Ryerson
        power_df = folder_to_series(path, label)
        print('Start plotting power cdf of ' + label)
        ax = sns.kdeplot(power_df['Power of ' + label], cumulative=False, shade=False)
        print(power_df.quantile(0.95))

    ax.set_title('Power CDF')
    plt.legend(loc=0)
    plt.xlabel('Power(in dB)')
    plt.ylabel('CDF')

    ax.get_figure().savefig(lambda_path + fig_name + '.png', ppi = 1200)
    print(fig_name + '.png is successfully generated')

abnor_path = ['LOS-5M-USRP1', 'LOS-5M-USRP2', 'LOS-5M-USRP3', 'NLOS-5M-USRP1', 'Dynamics-5M-USRP1']
abnor_path = ['/net/adv_spectrum/data/downsample/downsample_10/abnormal/' + i + '/' for i in abnor_path]
# print(abnor_path)
plot_power_cdf(abnor_path, 'FBS_PDF')
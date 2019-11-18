import numpy as np
import glob
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

figure_XCorr_path = '/home/shinanliu/XCorr/'
ry_path = '/net/adv_spectrum/data/feature/downsample_10/normal/ryerson_all/100_25/'
jcl_path = '/net/adv_spectrum/data/feature/downsample_10/normal/JCL/100_25/'
dt_path = '/net/adv_spectrum/data/feature/downsample_10/normal/downtown/100_25/'

def txt_to_series(file_path, select=100, n_channels=128):
    features = []
    i = 0

    with open(file_path, 'r') as f:
        for line in f:
            x = line.split()
            if i % select == 0:
                features.append(x)
            i += 1

    series = np.array(features).reshape((-1, n_channels)).astype('float64')
    return series

# Set the reference file
ry = txt_to_series(ry_path + 'feature_1518554457_880M_5m_ap.txt')
jcl = txt_to_series(jcl_path + 'feature_1572728951_880M_5m_ap.txt')
dt = txt_to_series(dt_path + 'feature_1519678799_880M_5m_ap.txt')

def all_xcorr(ref, file_path, label):
    for filename in sorted(glob.glob(file_path + '*.txt')):
        xcorr = []
        if filename != ref:
            cc = np.corrcoef(ref, txt_to_series(filename))
            for i in range(np.shape(ref)[0]):
                for j in range(np.shape(ref)[0], np.shape(cc)[0]):
                    xcorr.append(cc[i][j])
        return pd.DataFrame({'XCorr: ' + label: xcorr})

def plot_xcorr_cdf(ref, loc):
    print('Start processing ' + loc)
    xcorr_df = all_xcorr(ref, ry_path, loc + ' vs Ryerson')
    sns.set_style('white')
    plt.figure()
    print('Start plotting ' + loc + ' vs Ryerson')
    ax = sns.kdeplot(xcorr_df['XCorr: ' + loc + ' vs Ryerson'], cumulative=True, shade=False,
                    color='r')
    xcorr_df = all_xcorr(ref, jcl_path, loc + ' vs JCL')
    print('Start plotting ' + loc + ' vs JCL')
    ax = sns.kdeplot(xcorr_df['XCorr: ' + loc + ' vs JCL'], cumulative=True, shade=False,
                    color='b')
    print('Start plotting ' + loc + ' vs Downtown')
    xcorr_df = all_xcorr(ref, dt_path, loc + ' vs Downtown')
    ax = sns.kdeplot(xcorr_df['XCorr: ' + loc + ' vs Downtown'], cumulative=True, shade=False,
                    color='g')

    ax.set_title('Pearson correlation across locations')
    plt.legend(loc=2)
    plt.xlabel('Pearson Correlation Values')
    plt.ylabel('CDF')

    ax.get_figure().savefig(figure_XCorr_path + loc + '_all.png', ppi = 300)
    print(loc + '_all.png is successfully generated')

plot_xcorr_cdf(ry, 'Ryerson')
plot_xcorr_cdf(jcl, 'JCL')
plot_xcorr_cdf(dt, 'Downtown')
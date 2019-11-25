### Code by Shinan

import numpy as np
import glob
import pickle
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


XCorr_path = '/home/shinanliu/XCorr/'
ry_path = '/net/adv_spectrum/data/feature/downsample_10/normal/ryerson_all/100_25/'
jcl_path = '/net/adv_spectrum/data/feature/downsample_10/normal/JCL/100_25/'
dt_path = '/net/adv_spectrum/data/feature/downsample_10/normal/downtown/100_25/'


def txt_to_series(file_path, select=100, n_channels=128):
    features = []
    i = 0

    with open(file_path, 'r') as f:
        for line in f:
            x = line.split()
            rm = i % select
            if rm == 0:
                rand_select = np.random.randint(1, select)
                flag = 1
            elif rm % rand_select == 0 and flag:
                features.append(x)
                flag = 0
            i += 1

    series = np.array(features).reshape((-1, n_channels)).astype('float64')
    return series
    

def folder_to_series(file_path, select=1000, n_channels=128):
    features = []
    i = 0 
    for filename in sorted(glob.glob(file_path + '*.txt')):
        print('Beginning to process ' + filename)
        with open(filename, 'r') as f:
            for line in f:
                x = line.split()
                rm = i % select
                if rm == 0:
                    rand_select = np.random.randint(1, select)
                    flag = 1
                elif rm % rand_select == 0 and flag:
                    features.append(x)
                    flag = 0
                i += 1

    series = np.array(features).reshape((-1, n_channels)).astype('float64')
    print('Series of ' + file_path + ' is already constructed.')
    return series

def all_xcorr(ref, file_path, label):
    xcorr = []
    for filename in sorted(glob.glob(file_path + '*.txt')):
        if filename != ref:
            cc = np.corrcoef(ref, txt_to_series(filename))
            for i in range(np.shape(ref)[0]):
                for j in range(np.shape(ref)[0], np.shape(cc)[0]):
                    xcorr.append(cc[i][j])
    return pd.DataFrame({'XCorr: ' + label: xcorr})


def find_xcorr(ref, file_path):
    xcorr_fft_list = []
    for filename in sorted(glob.glob(file_path + '*.txt')):
        if filename != ref:
            print('Finding XCorr for '+ filename)
            target = txt_to_series(filename)
            cc = np.corrcoef(ref, target)
            for i in range(np.shape(ref)[0]):
                k = 0
                for j in range(np.shape(ref)[0], np.shape(cc)[0]):
                    if cc[i][j] < 0.3:
                        k += 1
                    if k > (np.shape(cc)[0] - np.shape(ref)[0]) * 0.8:
                        xcorr_fft_list.append(target[j - np.shape(ref)[0]])
                        break
    return xcorr_fft_list


def find_xcorr_inside(ref):
    xcorr_fft_list = []
    cc = np.corrcoef(ref)
    for i in range(np.shape(ref)[0]):
        k = 0
        for j in range(np.shape(cc)[0]):
            if cc[i][j] < 0.1:
                k += 1
            if k > np.shape(cc)[0] * 0.8:
                xcorr_fft_list.append(ref[i])
                break
    return xcorr_fft_list


def xcorr_fft_pick(xcorr_fft_list, pick_num):
    print(np.shape(xcorr_fft_list))
    xcorr_pick_list = [list() for i in range(12)]
    for i in range(12):
        if np.shape(xcorr_fft_list[i])[0] > 0:
            if np.shape(xcorr_fft_list[i])[0] > pick_num:
                rand_pick = np.random.randint(np.shape(xcorr_fft_list[i])[0], size = pick_num)
            # print(np.shape(rand_pick))
                for j in rand_pick:
                    xcorr_pick_list[i].append(xcorr_fft_list[i][j])
            else:
                xcorr_pick_list[i] = xcorr_fft_list[i]
    return xcorr_pick_list


def get_xcorr_dist(ref, file_path, pick_num):
    # Find all Xcorrlation data and get a distribution of these data
    xcorr_fft_list = [list() for i in range(12)]
    for filename in sorted(glob.glob(file_path + '*.txt')):
        print('Finding XCorr for '+ filename)
        target = txt_to_series(filename)
        print('Found XCorr for '+ filename)
        print(np.shape(ref), np.shape(target))
        cc = np.corrcoef(ref, target)
        print('Found Cross Coefficients for '+ filename)
        print(np.shape(cc))
        for i in range(np.shape(ref)[0], np.shape(cc)[0]):
            k = [0]*12
            for j in range(np.shape(ref)[0]):
                if cc[i][j] >= -0.2 and cc[i][j] < 1:
                    index = math.floor(cc[i][j] * 10) + 2
                    k[index] += 1
            for q in range(12):
                if k[q] >= np.shape(ref)[0] * 0.5:
                    #print('Found 1 for ', q)
                    xcorr_fft_list[q].append(target[i - np.shape(ref)[0]])
                    break
    print('Picking XCorr for '+ file_path)
    # print(xcorr_fft_list)
    xcorr_pick_list = xcorr_fft_pick(xcorr_fft_list, pick_num)
    return xcorr_pick_list


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

    ax.get_figure().savefig(XCorr_path + loc + '_all.png', ppi = 1200)
    print(loc + '_all.png is successfully generated')


def store_xcorr_dist(ref, file_path, loc, pick_num):
    # store the randomly picked xcorr according to its distribution
    XCorr_file = open(XCorr_path + 'XCorr_ry_' + loc + '.txt', 'a')
    xcorr_pick_list = get_xcorr_dist(ref, file_path, pick_num)
    for i in range(12):
        print('Pickup list size ', i, np.shape(xcorr_pick_list[i])[0])
        if np.shape(xcorr_pick_list[i])[0]:
            XCorr_file.write(str((i - 2)/10.0) + ', ' + str(np.shape(xcorr_pick_list[i])[0]) + '\n')
            print('Saving file ' + XCorr_path + 'XCorr_ry_' + loc + '.txt')
            np.savetxt(XCorr_file, xcorr_pick_list[i])
    XCorr_file.close()

# Set the reference file
ry = folder_to_series(ry_path)
# jcl = folder_to_series(jcl_path)
# dt = folder_to_series(dt_path)

store_xcorr_dist(ry, dt_path, 'dt', 5)
store_xcorr_dist(ry, jcl_path, 'jcl', 5)
# store_xcorr_dist(ry, ry_path, 'ry', 5)

# store_xcorr_dist(dt, dt_path, 'dt', 5)
# store_xcorr_dist(dt, jcl_path, 'jcl', 5)
# store_xcorr_dist(dt, ry_path, 'ry', 5)


# Find FFT points that are not correlated with ryerson BS data
# XCorr_file = open(XCorr_path + 'XCorr.txt', 'a')
# XCorr_inside = open(XCorr_path + 'XCorr_inside.txt', 'a')
# ry_jcl = find_xcorr(ry, jcl_path)
# print(np.shape(ry_jcl))
# np.savetxt(XCorr_file, ry_jcl)
# ry_dt = find_xcorr(ry, dt_path)
# print(np.shape(ry_dt))
# np.savetxt(XCorr_file, ry_dt)
# ry_jcl_dt = ry_jcl.append(rd for rd in ry_dt)
# print(np.shape(ry_jcl_dt))
# ry_xcorr = np.array(find_xcorr_inside(ry_jcl_dt))
# print(np.shape(ry_xcorr))
# np.savetxt(XCorr_inside, ry_xcorr)


# plot_xcorr_cdf(ry, 'Ryerson')
# plot_xcorr_cdf(jcl, 'JCL')
# plot_xcorr_cdf(dt, 'Downtown')

"""
    code by Shinan Liu
    This is to construct raw file from already sampled FFT
"""

import numpy as np
import glob
import pickle
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal


time_interval = 1  ## seconds
all_time = 40  ## seconds
step_sec = 10000000  ## step number per sec, i.e. sample rate
interval = time_interval * step_sec ## step number
power_level = 3  ## power dB
in_path = '/net/adv_spectrum/data/raw/normal/ryerson2/'
out_path = '/net/adv_spectrum/data/raw/abnormal/ry2_xcorr_mix'
xcorr_path = '~/XCorr/'


def txt_to_series(file_path, n_channels=128):
    feature_list = [list() for i in range(12)]
    series_list = [list() for i in range(12)]

    with open(file_path, 'r') as f:
        for line in f:
            x = line.split()
            if np.shape(x)[0] == 2:
                k = int(float(x[0].split(',')[0]) * 10 + 2)
            if np.shape(x)[0] == 128:
                feature_list[k].append(x)
    
    for k in range(12):
        series_list[k] = np.array(feature_list[k]).reshape((-1, n_channels)).astype('float64')

    return series_list


def fft_to_raw(norm_fft, std_data, mean_data):
    # normalization inversion
    fft = norm_fft * std_data + mean_data
    
    # ifft
    ifftshift = np.fft.ifftshift(fft)
    ifft = np.fft.ifft(ifftshift)
    
    # construct raw
    raw_list = []
    for i in ifft:
        raw_list.append(np.real(i))
        raw_list.append(np.imag(i))

    return np.array(raw_list)


def series_to_raw(series_list, std_data, mean_data):
    raw_list = [list() for i in range(12)]
    
    for k in range(12):
        for norm_fft in series_list[k]:
            raw_list[k].append(fft_to_raw(norm_fft, std_data, mean_data))
            
    return raw_list


def raw_to_std_mean(filename, start, interval):
    fid = open(filename, 'rb')
    dont_care = np.fromfile(fid, np.float32, count=start)
    data_array = np.fromfile(fid, np.float32, count=count).reshape((-1, 2))
    (a, _) = data_array.shape
    if a < count//2:
        end = (a//Slen)*Slen
    else:
        end = count//2

    block_count += 1
    new_array = data_array[:,0] + 1j*data_array[:,1]

    nan_count_a = 0
    nan_count_p = 0
    #every downsample*Slen points, compute FFT once
    for start_point in range(0, end, downsample*Slen):
        raw_data = new_array[start_point:(start_point+Slen)]
        Y = np.fft.fft(raw_data, Slen)
        P1 = np.fft.fftshift(Y)
        P2 = np.absolute(P1)
        amp = 10*np.log10(P2/Slen)
        #phase = np.angle(P1)

    # data clean
    # amplitude is unlikely to exceed 0dB. If amp > 0, could be measurement
    # error in data, so force it to -30dB
        if np.isnan(amp).any():
            nan_count_a += 1
            amp[np.isnan(amp)] = -120
        # if np.isnan(phase).any():
        #         nan_count_p += 1
        amp[amp<-120] = -120
        amp[amp>0] = -30
        output_data = amp.reshape((-1, Slen))

    return std_list, mean_list


def add_xcorr_raw(pick, time_interval):
    ry_ry_series = txt_to_series(xcorr_path + 'XCorr_ry_ry.txt')
    ry_jcl_series = txt_to_series(xcorr_path + 'XCorr_ry_jcl.txt')
    ry_dt_series = txt_to_series(xcorr_path + 'XCorr_ry_dt.txt')
    for i in range(6):
        xcorr_path = out_path + '_x' + str(i*0.2 - 0.1) + '/'
        if not os.path.exists(xcorr_path):
            os.mkdir(xcorr_path)
            print(xcorr_path + ' Created')
    files  = random.choices(glob.glob(in_path + '*.dat'), k=pick)
    for file in files:
        
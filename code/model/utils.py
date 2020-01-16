"""
Title: utils.py
Prescription: Provide helper functions
Declaration: Function `standard()` and `extract()` credit to Zhijing Li
"""

import warnings
warnings.filterwarnings('ignore')
import re
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


##########################################################
# Down-sample (txt) to Features (txt)
##########################################################
def standard(sequence, mean_data, std_data):
    if std_data == 0:
        raise Exception('Bad data')
    else:
        ret = []
        for i in range(len(sequence)):
            ret.append((sequence[i] - mean_data)/std_data)
    return ret


def str_to_flt(a):
    ## handle errors like unexpected contacnated data
    ## some downsampled data overflows and cause connected strings
    b = [] # the splitted list of strings
    len(re.findall("-", '-50--47.53'))
    for i in range(len(a)):
        if len(re.findall("-", a[i])) > 1:
            for k in a[i].split('-'):
                if k != '':
                    b.append('-' + k)
        else:
            b.append(a[i])
    c = [float(b[i]) for i in range(len(b))]
    return c


def extract_method3(filename, out, timestamps, predict_len):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            a = line.split()
            data.extend(str_to_flt(a))
            if len(data) == timestamps * 128:
                mean_data = np.mean(data)
                std_data = np.std(data)
            if len(data) == (timestamps + predict_len) * 128:
                data = standard(data, mean_data, std_data)
                for d in data:
                    out.write('%s ' %d)
                out.write('\n')
                data = []


##########################################################
# Features (txt) to Series (np.array)
##########################################################
def txt_to_series(file_path, n_channels=128):
    features = []

    with open(file_path, 'r') as f:
        for line in f:
            x = line.split()
            features.append(x)

    series = np.array(features).reshape((-1, n_channels)).astype('float64')
    return series


##########################################################
# Array (np.array) to Window (np.array)
##########################################################
def array_to_window(X, window_size):
    """
    Inputs:
        X (np.array): Its shape should be (n_time_steps, 128)
        window_size (int): the number of time steps in a window
        
    Return:
        result (np.array): Its shape should be (n_windows, 1, window_size, 128)
    """
    result = []
    ind = np.arange(0, X.shape[0], window_size)
    
    for start, end in zip(ind, np.r_[ind[1:], X.shape[0]]):
        if end - start < window_size:
            # Discard the last few lines
            break
        result.append([X[start:end, :]])
        
    return np.array(result)


##########################################################
# Series (np.array) to Windowed Dataset (tf.data.Dataset)
##########################################################
def windowed_dataset(series, window_size, batch_size, predict_size,
                     shift_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + predict_size, shift=shift_size,
                   drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + predict_size))
    ds = ds.map(lambda w: (w[:window_size, :], w[window_size:, :]))
    ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return ds


##########################################################
# Model Forecast (np.array to tf.data.Dataset)
##########################################################
def model_forecast(model, series, batch_size, window_size, predict_size,
                   shift_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + predict_size, shift=shift_size,
                   drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + predict_size))
    ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    forecast = model.predict(ds)
    return forecast


##########################################################
# Create true list from original list
##########################################################
def windowed_true(series, shift_size, predict_size):
    """
    Input:
          series (numpy array): the array we would like to make prediction on
    Output:
          series_true (numpy array): the windowed series which is comparable
          with series_hat (output of model_forecast)
    """
    return np.array([series[shift_size * i - predict_size: shift_size * i, :]
                    .reshape((-1, predict_size, 128))
                    for i in range(1, (len(series) // shift_size) + 1)])

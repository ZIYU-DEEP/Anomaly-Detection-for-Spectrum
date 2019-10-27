"""
Title: utils.py
Prescription: Provide helper functions
Author: Yeol Ye
Declaration: Function `standard()` and `extract()` credit to Zhijing Li
"""

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


def extract_method3(filename, out, timestamps, predict_len):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            a = line.split()
            data.extend([float(a[i]) for i in range(len(a))])
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
# Series (np.array) to Windowed Dataset (tf.data.Dataset)
##########################################################
def windowed_dataset(series, window_size, batch_size, shift_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + shift_size, shift=shift_size,
                   drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + shift_size))
    ds = ds.map(lambda w: (w[:-shift_size, :], w[window_size:, :]))
    ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return ds


##########################################################
# Model Forecast (np.array to tf.data.Dataset)
##########################################################
def model_forecast(model, series, batch_size, window_size, shift_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + shift_size, shift=shift_size, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + shift_size))
    ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    forecast = model.predict(ds)
    return forecast


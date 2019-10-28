"""
Title: utils.py
Prescription: Provide helper functions
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
                    for i in range(1, (len(series) // shift_size) + 1)])\
           .reshape((-1, 128))


def fix_gpu_memory():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess

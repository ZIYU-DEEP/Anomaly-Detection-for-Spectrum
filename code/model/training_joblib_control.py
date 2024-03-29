"""
Title: training.py
Prescription: Training the rnn model
Declaration: The LSTM structure credits to Zhijing Li
"""


from tensorflow.keras.callbacks import EarlyStopping
import utils
import os
import pickle
import joblib
import sys
import glob
import tensorflow as tf
import numpy as np
import joblib


##########################################################
# 1. Initialization
##########################################################
# Arguments
downsample_ratio = int(sys.argv[1])
window_size = int(sys.argv[2])
predict_size = int(sys.argv[3])
normal_folder = str(sys.argv[4])  # e.g. ryerson
shift_train = int(sys.argv[5])
shift_eval = int(sys.argv[6])
batch_size = int(sys.argv[7])
epochs = int(sys.argv[8])
gpu_no = int(sys.argv[9])

normal_folder_control = normal_folder + '_control'

print('Using GPU:', gpu_no)
# String variables
downsample_str = 'downsample_' + str(downsample_ratio)
window_predict_size = str(sys.argv[2]) + '_' + str(sys.argv[3])
# Set gpu environment
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[gpu_no], 'GPU')
logical_devices = tf.config.experimental.list_logical_devices('GPU')

# General path
path = '/net/adv_spectrum/data/'

# Path to read featurized txt
normal_output_path = path + 'feature/{}/normal/{}/{}/'\
                    .format(downsample_str, normal_folder, window_predict_size)

# Path to save model and full_x_valid
full_x_valid_path = '/net/adv_spectrum/result/x_valid/'
full_x_valid_filename = full_x_valid_path + 'full_x_valid_{}_{}_{}.pkl'\
                        .format(downsample_str,
                                normal_folder_control,
                                window_predict_size)
model_path = '/net/adv_spectrum/model/{}/{}/'\
             .format(downsample_str, normal_folder_control)
model_filename = model_path + '{}_{}.h5'\
                 .format(downsample_ratio, window_predict_size)

# Check path existence
if not os.path.exists(full_x_valid_path):
    os.makedirs(full_x_valid_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)


##########################################################
# 2. Construct normal series and abnormal series
##########################################################
normal_series_list = []

print('Start constructing normal series....')
for filename in sorted(glob.glob(normal_output_path + '*.txt')):
    print(filename)
    series = utils.txt_to_series(filename)[:, 87:123]
    print(series.shape)
    normal_series_list.append(series)


##########################################################
# 3. Load and Process Data
##########################################################
print('Start loading and processing data...')
# Initiate data
temp = normal_series_list[0].copy()
split_time = int(temp.shape[0] * 0.8)
temp_x_train = temp[:split_time]
temp_x_valid = temp[split_time:]
full_x_valid = temp_x_valid.copy()

# Initiate training and valid set
full_train_set = utils.windowed_dataset(temp_x_train, window_size, batch_size,
                                        predict_size, shift_train)
full_valid_set = utils.windowed_dataset(temp_x_valid, window_size, batch_size,
                                        predict_size, shift_eval)

# Create full train set and full valid set
for series in normal_series_list[1:]:
    split_time = int(series.shape[0] * 0.8)
    x_train = series[:split_time]
    x_valid = series[split_time:]
    full_x_valid = np.concatenate((full_x_valid, x_valid))

    train_set = utils.windowed_dataset(x_train, window_size, batch_size,
                                       predict_size, shift_train)
    valid_set = utils.windowed_dataset(x_valid, window_size, batch_size,
                                       predict_size, shift_eval)

    full_train_set = full_train_set.concatenate(train_set)
    full_valid_set = full_valid_set.concatenate(valid_set)


# Save full_x_valid for future threshold use
with open(full_x_valid_filename, 'wb') as f:
    joblib.dump(full_x_valid, f)

##########################################################
# 4. Compile model
##########################################################
print('Start compiling model...')
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model = tf.keras.models.\
    Sequential([tf.keras.layers.LSTM(16, return_sequences=True,
                                     input_shape=[None, 36]),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(16, return_sequences=True),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(16, return_sequences=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(36 * predict_size),
                tf.keras.layers.Reshape((predict_size, 36))])

es = EarlyStopping(monitor='mean_squared_error',
                   min_delta=0.0001,
                   patience=12)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=["mean_squared_error"])


##########################################################
# 5. Fit model
##########################################################
print('Fitting model...')
history = model.fit(full_train_set, epochs=epochs, callbacks=[es])
model.save(model_filename)


##########################################################
# 6. Model Validation and Evaluation
##########################################################
print('Validate model on valid set (using normal data):')
model.evaluate(full_valid_set)

tf.keras.backend.clear_session()
print('Training finished!')

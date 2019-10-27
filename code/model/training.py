"""
Title: training.py
Prescription: Training the rnn model
Author: Yeol Ye
Declaration: The LSTM structure credits to Zhijing Li
"""

from tensorflow.keras.callbacks import EarlyStopping
import utils
import pickle
import sys
import tensorflow as tf
import numpy as np

downsample_ratio = int(sys.argv[1])
window_size = int(sys.argv[2])
predict_size = int(sys.argv[3])
shift_train = int(sys.argv[4])
shift_eval = int(sys.argv[5])
batch_size = int(sys.argv[6])
epochs = int(sys.argv[7])

downsample_str = 'downsample_' + str(downsample_ratio)
window_predict_size = str(sys.argv[2]) + '_' + str(sys.argv[3])
normal_series_list_path = '../../data/dataset/' + downsample_str + '/' \
                          + 'normal_series_list_' + window_predict_size
abnormal_series_list_path = '../../data/dataset/' + downsample_str + '/' \
                            + 'abnormal_series_list_' + window_predict_size
full_x_valid_path = '../../data/dataset/' + downsample_str + '/' \
                    + 'full_x_valid_' + window_predict_size
model_path = '../../model/{}_{}_{}'.format(downsample_ratio, window_size,
                                           predict_size)

##########################################################
# Load and Process Data
##########################################################
# Load normal list and abnormal list of series
with open(normal_series_list_path, 'rb') as f:
    normal_series_list = pickle.load(f)

with open(abnormal_series_list_path, 'rb') as f:
    abnormal_series_list = pickle.load(f)

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


# Create full abnormal series list
abnormal_set_list = []
for series in abnormal_series_list:
    abnormal_set = utils.windowed_dataset(series, window_size, batch_size,
                                          predict_size, shift_eval)
    abnormal_set_list.append(abnormal_set)


# Save full_x_valid for future threshold use
with open(full_x_valid_path, 'wb') as f:
    pickle.dump(full_x_valid, f)


##########################################################
# Compile model
##########################################################
model = tf.keras.models.\
    Sequential([tf.keras.layers.LSTM(64, return_sequences=True,
                                     input_shape=[None, 128]),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(64, return_sequences=False),
                tf.keras.layers.Dense(128 * predict_size),
                tf.keras.layers.Reshape((predict_size, 128))])

es = EarlyStopping(monitor='mse',
                   min_delta=0.0001,
                   patience=5)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=["mse"],
              callbacks=[es])

##########################################################
# Fit model
##########################################################
history = model.fit(full_train_set, epochs=epochs, callbacks=[es])
model.save(model_path)


##########################################################
# Model Validation and Evaluation
##########################################################
print('Validate model on valid set (using normal data):')
model.evaluate(full_valid_set)

print('Evaluate model on test set (using abnormal data):')
for i, abnormal_set in enumerate(abnormal_set_list):
    print('Abnormal set: ', i)
    print(model.evaluate(abnormal_set))

print('Training finished!')

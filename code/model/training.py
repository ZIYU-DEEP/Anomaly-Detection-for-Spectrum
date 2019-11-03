"""
Title: training.py
Prescription: Training the rnn model
Declaration: The LSTM structure credits to Zhijing Li
"""


from tensorflow.keras.callbacks import EarlyStopping
import utils
import os
import pickle
import sys
import glob
import tensorflow as tf
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##########################################################
# 1. Initialization
##########################################################
# Arguments
downsample_ratio = int(sys.argv[1])
window_size = int(sys.argv[2])
predict_size = int(sys.argv[3])
normal_folder = str(sys.argv[4])  # e.g. ryerson
anomaly_folder = str(sys.argv[5])  # e.g. 0208_anomaly
shift_train = int(sys.argv[6])
shift_eval = int(sys.argv[7])
batch_size = int(sys.argv[8])
epochs = int(sys.argv[9])
gpu_no = str(sys.argv[10])

# Set gpu environment
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

# String variables
downsample_str = 'downsample_' + str(downsample_ratio)
window_predict_size = str(sys.argv[2]) + '_' + str(sys.argv[3])

# General path
path = '/net/adv_spectrum/data/'

# Path to read featurized txt
normal_output_path = path + 'feature/{}/normal/{}/{}/'\
                    .format(downsample_str, normal_folder, window_predict_size)
abnormal_output_path = path + 'feature/{}/abnormal/{}/{}/'\
                    .format(downsample_str, anomaly_folder, window_predict_size)

# Path to save model and full_x_valid
full_x_valid_path = '/net/adv_spectrum/result/x_valid/'
full_x_valid_filename = full_x_valid_path + 'full_x_valid_{}_{}_{}.pkl'\
                        .format(downsample_str,
                                normal_folder,
                                window_predict_size)
model_path = '/net/adv_spectrum/model/{}/{}/'\
             .format(downsample_str, normal_folder)
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
abnormal_series_list = []

print('Start constructing normal series....')
for filename in sorted(glob.glob(normal_output_path + '*.txt')):
    print(filename)
    series = utils.txt_to_series(filename)
    print(series.shape)
    normal_series_list.append(series)

print('Start constructing abnormal series....')
for filename in sorted(glob.glob(abnormal_output_path + '*.txt')):
    print(filename)
    series = utils.txt_to_series(filename)
    print(series.shape)
    abnormal_series_list.append(series)


##########################################################
# 3. Load and Process Data
##########################################################
print('Start loading and processing data...')
# Create full_x_valid
# (Note this is based on the assumption that all the normal list are in the
# correct order and are consistent)
full_x = np.array(normal_series_list).reshape((-1, 128)).astype('float64')
split_time = int(full_x.shape[0] * 0.8)
full_x_train = full_x[:split_time]
full_x_valid = full_x[split_time:]

# Create train set and valid set
full_train_set = utils.windowed_dataset(full_x_train, window_size, batch_size,
                                        predict_size, shift_eval)
full_valid_set = utils.windowed_dataset(full_x_valid, window_size, batch_size,
                                        predict_size, shift_eval)

# Create full abnormal series list
abnormal_set_list = []
for series in abnormal_series_list:
    abnormal_set = utils.windowed_dataset(series, window_size, batch_size,
                                          predict_size, shift_eval)
    abnormal_set_list.append(abnormal_set)

# Save full_x_valid for future threshold use
with open(full_x_valid_filename, 'wb') as f:
    pickle.dump(full_x_valid, f)


##########################################################
# 4. Compile model
##########################################################
print('Start compiling model...')
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model = tf.keras.models.\
    Sequential([tf.keras.layers.LSTM(64, return_sequences=True,
                                     input_shape=[None, 128]),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LSTM(64, return_sequences=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128 * predict_size),
                tf.keras.layers.Reshape((predict_size, 128))])

es = EarlyStopping(monitor='mean_squared_error',
                   min_delta=0.0001,
                   patience=5)

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

print('Evaluate model on test set (using abnormal data):')
for i, abnormal_set in enumerate(abnormal_set_list):
    print('Abnormal set: ', i)
    print(model.evaluate(abnormal_set))

tf.keras.backend.clear_session()
print('Training finished!')

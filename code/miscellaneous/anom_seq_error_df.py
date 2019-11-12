#  读进来两个file
#  分别算出他们的anom_seq
#  然后可以开始画图

import pickle

file_1000 = '/net/adv_spectrum/result/error_df/anomaly/downsample_10/ry2_' \
             'jcl_mix_G6/full_anom_error_df_list_ryerson_all_1000_250_250.pkl'
file_100 = '/net/adv_spectrum/result/error_df/anomaly/downsample_10/ry2_jcl_' \
            'mix_G6/full_anom_error_df_list_ryerson_all_100_25_25.pkl'

# Interval values
all_samp = 200000000 * 2  # all samp per file
samp_sec = 5000000 * 2  # sample rate, each sample has I/Q 2 values
interval = 5  # in seconds
inter_samp = samp_sec * interval / downsample_ratio  # in samp number
trash_count = 102400  # begining samples being throwed

# calculate the intervals in prediction window
ini_anom = int((((inter_samp - trash_count) / 256) - window_size)
               // shift_eval)
anom_interval = int(((inter_samp / 256) - window_size) // shift_eval)
up_down_interval = 10 #int(window_size // shift_eval)



with open(file_1000, 'rb') as f:
    error_1000 = pickle.load(f)

with open(file_100, 'rb') as f:
    error_100 = pickle.load(f)

#  读进来两个file
#  分别算出他们的anom_seq
#  然后可以开始画图

import pickle
import seaborn as sns
import matplotlib.pyplot as plt

file_anom_seq_100 = '/net/adv_spectrum/miscellaneous/anom_seq_100_25.pkl'
file_anom_seq_1000 = '/net/adv_spectrum/miscellaneous/anom_seq_1000_250.pkl'
file_error_df_100 = '/net/adv_spectrum/result/error_df/anomaly/' \
                     'downsample_10/ry2_jcl_mix_G6/' \
                     'full_anom_error_df_list_ryerson_all_100_25_25.pkl'
file_error_df_1000  = '/net/adv_spectrum/result/error_df/anomaly/' \
                     'downsample_10/ry2_jcl_mix_G6/' \
                     'full_anom_error_df_list_ryerson_all_1000_250_250.pkl'

with open(file_anom_seq_100, 'rb') as f:
    anom_seq_100 = pickle.load(f)

with open(file_anom_seq_1000, 'rb') as f:
    anom_seq_1000 = pickle.load(f)

with open(file_error_df_100, 'rb') as f:
    error_df_100 = pickle.load(f)

with open(file_error_df_1000, 'rb') as f:
    error_df_1000 = pickle.load(f)


print('Drawing the anom time mse plot!')
fig, (ax, ax1) = plt.subplots(2, 1, figsize=(23, 12), dpi=200)
sns.lineplot(x=error_df_100.index,
             y=anom_seq_100, ax=ax)
sns.lineplot(x=error_df_100.index,
             y=error_df_100.iloc[:, 0], color='orange', ax=ax)

ax.set_ylim(top=7)
ax.set_xlabel('Time')
ax.set_ylabel('MSE')

sns.lineplot(x=error_df_1000.index,
             y=anom_seq_1000, ax=ax1)
sns.lineplot(x=error_df_1000.index,
             y=error_df_1000.iloc[:, 0], color='orange', ax=ax1)

ax1.set_ylim(top=7)
ax1.set_xlabel('Time')
ax1.set_ylabel('MSE')
sns.despine(fig=fig)
fig.save_fig('/net/adv_spectrum/miscellaneous/contrast.png')

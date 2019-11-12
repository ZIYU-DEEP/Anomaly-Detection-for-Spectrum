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
    error_df_100 = pickle.load(f)[0]

with open(file_error_df_1000, 'rb') as f:
    error_df_1000 = pickle.load(f)[0]


print('Drawing the anom time mse plot!')
fig, axes = plt.subplots(2, 1, figsize=(23, 12), dpi=200)
sns.lineplot(x=error_df_100.index,
             y=anom_seq_100, ax=axes[0])
sns.lineplot(x=error_df_100.index,
             y=error_df_100.iloc[:, 0], color='orange', ax=axes[0])

axes[0].set_ylim(top=7)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('MSE')

sns.lineplot(x=error_df_1000.index,
             y=anom_seq_1000, ax=axes[1])
sns.lineplot(x=error_df_1000.index,
             y=error_df_1000.iloc[:, 0], color='orange', ax=axes[1])

axes[1].set_ylim(top=7)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('MSE')
sns.despine(fig=fig)
fig.save_fig('/net/adv_spectrum/miscellaneous/contrast.png')

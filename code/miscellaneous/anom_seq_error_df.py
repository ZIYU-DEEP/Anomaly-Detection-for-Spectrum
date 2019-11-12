import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
fig, (ax1, ax2) = plt.subplots(2, figsize=(23, 12), dpi=200)
sns.lineplot(x=error_df_100.index,
             y=np.array(anom_seq_100) - 1, ax=ax1)
sns.scatterplot(x=error_df_100.index,
             y=error_df_100.iloc[:, 0], color='orange', ax=ax1, linewidth=0, s=6)

ax1.set_ylim(top=5.5, bottom=0)
ax1.set_yticks(np.arange(0, 5.5, 0.5))
ax1.set_xlabel('Time')
ax1.set_ylabel('MSE')
ax1.set_title('norm=100_25, model=100_25')

sns.lineplot(x=error_df_1000.index,
             y=np.array(anom_seq_1000) - 1, ax=ax2)
sns.scatterplot(x=error_df_1000.index,
             y=error_df_1000.iloc[:, 0], color='orange', ax=ax2, linewidth=0, s=6)

ax2.set_ylim(top=5.5, bottom=0)
ax2.set_yticks(np.arange(0, 5.5, 0.5))
ax2.set_xlabel('Time')
ax2.set_ylabel('MSE')
ax2.set_title('norm=1000_250, model=1000_250')
sns.despine(fig=fig)
fig.save_fig('/net/adv_spectrum/miscellaneous/contrast.png')

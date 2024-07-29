import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('train_log.csv')

solver = 'AdamW'
netsize = [128, 128]
additional = 'RELU_L2_1e-5_EVSTEP'
lr = [3e-4, 3e-5]

fig, axes = plt.subplots(nrows=4, sharex=True, figsize=[8,8])

axes[0].plot(df['ep'], df['step'])
axes[1].plot(df['ep'], df['a_loss'], color='orchid')
axes[2].plot(df['ep'], df['c_loss'], color='seagreen')
axes[3].plot(df['ep'], df['rew'], color='lightsalmon', label='Inst.')
axes[3].plot(df['ep'], df['rew'].rolling(10).mean(), color='crimson', label='Avg. (10 eps)')

# Formatting
axes[1].plot(df['ep'], np.zeros_like(df['rew']), 'k:')
axes[3].plot(df['ep'], np.zeros_like(df['rew']), 'k:')
axes[3].plot(df['ep'], 200 * np.ones_like(df['rew']), '--', color='chartreuse')
axes[0].set_ylabel('Ep. length')
axes[1].set_ylabel('Actor Loss')
axes[2].set_ylabel('Critic Losses')
axes[3].set_ylabel('Reward')
axes[3].legend()

axes[0].set_ylim([0, 1000])
axes[1].set_ylim([-10,10])
axes[2].set_ylim([0, 100])
axes[3].set_ylim([-500,300])

axes[3].set_xlim([0, df['ep'].max()])

axes[3].set_xlabel('Episode')

title = f'A2C | A {netsize[0]}, LR {lr[0]} | C {netsize[1]}, LR {lr[1]}, | {solver}, {additional}' 
axes[0].set_title(title)

fig.savefig(f'A2C_{netsize[0]}_{netsize[1]}_{lr[0]}_{lr[1]}_{solver}_{additional}.png')

# fig, axes = plt.subplots(ncols=3)
# axes[0].plot(df['step'], df['rew'], 'r.')
# axes[1].plot(df['step'], df['a_loss'], 'b.')
# axes[3].plot(df['rew'], df['c_loss'], 'g.')
# plt.show()
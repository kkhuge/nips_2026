import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=14)


def plot_learning(result_dir, ax, smooth=10, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    data = data[:500]
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)


fig, ax = plt.subplots(figsize=[5, 4])
#train

plot_learning('result_loss/fedavg5/loss_traincifar10_all_data_1_random_iidresnet18.npy',
              ax, label='FedAvg_train_iid', linestyle='--', color='tab:cyan')
plot_learning('result_loss/fedavg5/loss_testcifar10_all_data_1_random_iidresnet18.npy',
              ax, label='FedAvg_test_iid', linestyle='--', color='tab:purple')
plot_learning('result_loss/fedavg5/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
              ax, label='FedForth_train_niid', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg5/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
              ax, label='FedForth_test_niid', linestyle='-', color='tab:green')


ax.set_xlim([0, 500])
ax.set_ylim([0,3])
ax.set_xticks([0,100,200,300,400,450,500])
ax.set_yticks([0,1,2,3])
ax.set_xlabel('Global Round')
ax.set_ylabel('Loss')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle='-', alpha=0.3)
ax.axvline(x=450, color='#555555', linestyle='--', linewidth=1.5)
ax.text(x=450, y=0.4, s='  Stage 2 Begins',
        color='#555555',
        rotation=90,
        verticalalignment='center',
        fontsize=14,
        transform=ax.get_xaxis_transform())

ax.legend(handlelength=2.3,
          loc='upper right',
          bbox_to_anchor=(0.92, 1))
fig.tight_layout()
plt.show()

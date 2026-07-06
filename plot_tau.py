import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True 
matplotlib.rc('font', size=14)



def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)


def compute_last50_stats(result_dir):
    data = np.load(result_dir, allow_pickle=True)

    last50 = data[-50:]
    mean_val = np.mean(last50)
    std_val = np.std(last50)

    print(f"File: {result_dir}")
    print(f"Last 50 rounds mean: {mean_val:.6f}")
    print(f"Last 50 rounds std: {std_val:.6f}")
    print("-" * 60)



fig, ax = plt.subplots(figsize=[5, 4])
#train

# #0.1
plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_tau1.npy',
              ax, label=r'$\tau_2=1$', linestyle='-', color='tab:cyan')
plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_tau5.npy',
              ax, label=r'$\tau_2=5$', linestyle='-', color='tab:red')
# plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_tau10.npy',
#               ax, label=r'$\tau=10$', linestyle='-', color='tab:green')
plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_tau50.npy',
              ax, label=r'$\tau_2=50$', linestyle='-', color='tab:purple')


ax.set_xlim([0, 100])
ax.set_ylim([0.55, 0.65])
ax.set_xticks(np.arange(0,120,20))
ax.set_yticks([0.55,0.57,0.59,0.61,0.63,0.65])
ax.set_xlabel('Global Round')
ax.set_ylabel('Test Accuracy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle='-', alpha=0.3)
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()




# # #0.5
# plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_freeze_tau1.npy',
#               ax, label=r'$\tau_2=1$', linestyle='-', color='tab:cyan')
# plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_freeze_tau5.npy',
#               ax, label=r'$\tau_2=5$', linestyle='-', color='tab:red')
# # plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_freeze_tau10.npy',
# #               ax, label=r'$\tau=10$', linestyle='-', color='tab:green')
# plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_freeze_tau50.npy',
#               ax, label=r'$\tau_2=50$', linestyle='-', color='tab:purple')
# 
# 
# ax.set_xlim([0, 100])
# ax.set_ylim([0.62, 0.68])
# ax.set_xticks(np.arange(0,120,20))
# ax.set_yticks([0.62,0.64,0.66,0.68])
# ax.set_xlabel('Global Round')
# ax.set_ylabel('Test Accuracy')
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.grid(True, linestyle='-', alpha=0.3)
# ax.legend(handlelength=2.3)
# fig.tight_layout()
# plt.show()


import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True 
matplotlib.rc('font', size=14) 



def plot_learning(result_dir, ax, smooth=10, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)



fig, ax = plt.subplots(figsize=[5, 4])
#train



# #0.1
plot_learning('result_acc/fedavg5/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_acc/fedetf/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_etf.npy',
              ax, label='FedETF', linestyle='-', color='tab:green')
plot_learning('result_acc/feduv/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_uv.npy',
              ax, label='FedUV', linestyle='-', color='tab:purple')



# # 0.5
# plot_learning('result_acc/fedavg5/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
#               ax, label='FedForth', linestyle='-', color='tab:red')
# plot_learning('result_acc/fedetf/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_etf.npy',
#               ax, label='FedETF', linestyle='-', color='tab:green')
# plot_learning('result_acc/feduv/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_uv.npy',
#               ax, label='FedUV', linestyle='-', color='tab:purple')



ax.set_xlim([0, 550])
ax.set_ylim([0, 0.7])
ax.set_xticks([0,100,200,300,400,450,500])
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
ax.set_xlabel('Global Round')
ax.set_ylabel('Test Accuracy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


ax.grid(True, linestyle='-', alpha=0.3)
ax.axvline(x=450, color='#555555', linestyle='--', linewidth=1.5)
ax.text(x=450, y=0.4, s='  Stage 2 Begins',
        color='#555555',
        rotation=90,
        verticalalignment='center',
        fontsize=14,
        transform=ax.get_xaxis_transform())

ax.legend(handlelength=2.3, frameon=True)

fig.tight_layout()
plt.show()

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True 
matplotlib.rc('font', size=12)



def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)

def compute_last15_stats(result_dir):
    data = np.load(result_dir, allow_pickle=True)
    last15 = data[-10:]
    mean_val = np.mean(last15)
    std_val = np.std(last15)

    print(f"File: {result_dir}")
    print(f"Last 15 rounds mean: {mean_val:.6f}")
    print(f"Last 15 rounds std: {std_val:.6f}")
    print("-" * 60)


fig, ax = plt.subplots(figsize=[5, 4])


# #fedavg_0.1
plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.1resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.1resnet18.npy',
               ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last15_stats('result_acc/fedavg4/acc_testtinyimagenet_niid_0.1resnet18_freeze.npy')
compute_last15_stats('result_acc/fedavg4/acc_testtinyimagenet_niid_0.1resnet18.npy')

#fedavg_0.5
plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.5resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last15_stats('result_acc/fedavg4/acc_testtinyimagenet_niid_0.5resnet18_freeze.npy')
compute_last15_stats('result_acc/fedavg4/acc_testtinyimagenet_niid_0.5resnet18.npy')

#fedprox_0.1
plot_learning('result_acc/fedavg6/acc_testtinyimagenet_niid_0.1resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_acc/fedavg6/acc_testtinyimagenet_niid_0.1resnet18.npy',
              ax, label='FedProx', linestyle='-', color='tab:green')
compute_last15_stats('result_acc/fedavg6/acc_testtinyimagenet_niid_0.1resnet18_freeze.npy')
compute_last15_stats('result_acc/fedavg6/acc_testtinyimagenet_niid_0.1resnet18.npy')


#fedprox_0.5
plot_learning('result_acc/fedavg6/acc_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_acc/fedavg6/acc_testtinyimagenet_niid_0.5resnet18.npy',
              ax, label='FedProx', linestyle='-', color='tab:green')
compute_last15_stats('result_acc/fedavg6/acc_testtinyimagenet_niid_0.5resnet18_freeze.npy')
compute_last15_stats('result_acc/fedavg6/acc_testtinyimagenet_niid_0.5resnet18.npy')

#scaffold_0.1
plot_learning('result_acc/scaffold/acc_testtinyimagenet_niid_0.1resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_acc/scaffold/acc_testtinyimagenet_niid_0.1resnet18.npy',
              ax, label='Scaffold', linestyle='-', color='tab:green')
compute_last15_stats('result_acc/scaffold/acc_testtinyimagenet_niid_0.1resnet18_freeze.npy')
compute_last15_stats('result_acc/scaffold/acc_testtinyimagenet_niid_0.1resnet18.npy')

#scaffold_0.5
plot_learning('result_acc/scaffold/acc_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_acc/scaffold/acc_testtinyimagenet_niid_0.5resnet18.npy',
              ax, label='Scaffold', linestyle='-', color='tab:green')
compute_last15_stats('result_acc/scaffold/acc_testtinyimagenet_niid_0.5resnet18_freeze.npy')
compute_last15_stats('result_acc/scaffold/acc_testtinyimagenet_niid_0.5resnet18.npy')


ax.set_xlim([0, 100])
ax.set_ylim([0.1, 1])
ax.set_xticks(np.arange(0,20,5))
ax.set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
ax.set_xlabel('Global Round')
ax.set_ylabel('Accuracy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle='-', alpha=0.3)
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()

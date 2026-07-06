import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True 
matplotlib.rc('font', size=12) 



def plot_learning(result_dir_1, result_dir_2, ax, smooth=1, interval=1, **kwargs):
    data_1 = np.load(result_dir_1,allow_pickle=True)
    data_2 = np.load(result_dir_2, allow_pickle=True)
    data = data_2-data_1
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)

def compute_last15_gen_error(train_file, test_file, smooth=1, interval=1):
    train = np.load(train_file, allow_pickle=True)
    test = np.load(test_file, allow_pickle=True)
    data = test - train 
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    data = data[::interval]
    last15 = data[-10:]
    mean_val = np.mean(last15)
    std_val = np.std(last15)
    print(f"Train File: {train_file}")
    print(f"Test  File: {test_file}")
    print(f"Last 15 Generalization Error Mean : {mean_val:.6f}")
    print(f"Last 15 Generalization Error Std  : {std_val:.6f}")
    print("-" * 60)



fig, ax = plt.subplots(figsize=[5, 4])

#fedavg_0.1
plot_learning('result_loss/fedavg4/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
'result_loss/fedavg4/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg4/loss_traintinyimagenet_niid_0.1resnet18.npy',
'result_loss/fedavg4/loss_testtinyimagenet_niid_0.1resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/fedavg4/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
    'result_loss/fedavg4/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/fedavg4/loss_traintinyimagenet_niid_0.1resnet18.npy',
    'result_loss/fedavg4/loss_testtinyimagenet_niid_0.1resnet18.npy'
)

#fedavg_0.5
plot_learning('result_loss/fedavg4/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
'result_loss/fedavg4/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg4/loss_traintinyimagenet_niid_0.5resnet18.npy',
'result_loss/fedavg4/loss_testtinyimagenet_niid_0.5resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/fedavg4/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
    'result_loss/fedavg4/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/fedavg4/loss_traintinyimagenet_niid_0.5resnet18.npy',
    'result_loss/fedavg4/loss_testtinyimagenet_niid_0.5resnet18.npy'
)
#
#fedprox_0.1
plot_learning('result_loss/fedavg6/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
'result_loss/fedavg6/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg6/loss_traintinyimagenet_niid_0.1resnet18.npy',
'result_loss/fedavg6/loss_testtinyimagenet_niid_0.1resnet18.npy',
              ax, label='FedProx', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/fedavg6/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
    'result_loss/fedavg6/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/fedavg6/loss_traintinyimagenet_niid_0.1resnet18.npy',
    'result_loss/fedavg6/loss_testtinyimagenet_niid_0.1resnet18.npy'
)

#fedprox_0.5
plot_learning('result_loss/fedavg6/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
'result_loss/fedavg6/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg6/loss_traintinyimagenet_niid_0.5resnet18.npy',
'result_loss/fedavg6/loss_testtinyimagenet_niid_0.5resnet18.npy',
              ax, label='FedProx', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/fedavg6/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
    'result_loss/fedavg6/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/fedavg6/loss_traintinyimagenet_niid_0.5resnet18.npy',
    'result_loss/fedavg6/loss_testtinyimagenet_niid_0.5resnet18.npy'
)

#scaffold_0.1
plot_learning('result_loss/scaffold/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
'result_loss/scaffold/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_loss/scaffold/loss_traintinyimagenet_niid_0.1resnet18.npy',
'result_loss/scaffold/loss_testtinyimagenet_niid_0.1resnet18.npy',
              ax, label='Scaffold', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/scaffold/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
    'result_loss/scaffold/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/scaffold/loss_traintinyimagenet_niid_0.1resnet18.npy',
    'result_loss/scaffold/loss_testtinyimagenet_niid_0.1resnet18.npy'
)

#scaffold_0.5
plot_learning('result_loss/scaffold/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
'result_loss/scaffold/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')
plot_learning('result_loss/scaffold/loss_traintinyimagenet_niid_0.5resnet18.npy',
'result_loss/scaffold/loss_testtinyimagenet_niid_0.5resnet18.npy',
              ax, label='Scaffold', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/scaffold/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
    'result_loss/scaffold/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/scaffold/loss_traintinyimagenet_niid_0.5resnet18.npy',
    'result_loss/scaffold/loss_testtinyimagenet_niid_0.5resnet18.npy'
)

ax.set_xlim([0, 100])
ax.set_ylim([0,4])
ax.set_xticks([0,20,40,60,80,100])
ax.set_yticks([0,1,2,3,4])
ax.set_xlabel('Global Round')
ax.set_ylabel('Generalization Error(test loss-train loss)')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle='-', alpha=0.3)
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()

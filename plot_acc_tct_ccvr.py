import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=14)



def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    data = data[-100:]
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)
fig, ax = plt.subplots(figsize=[5, 4])
#train



# #0.1

plot_learning('result_acc/boontk/acc_testtinyimagenet_niid_0.1resnet18_1_communication.npy',
              ax, label='TCT_1', linestyle='-', color='tab:orange')
plot_learning('result_acc/boontk/acc_testtinyimagenet_niid_0.1resnet18_10_communication.npy',
              ax, label='TCT_10', linestyle='-', color='tab:green')
plot_learning('result_acc/boontk/acc_testtinyimagenet_niid_0.1resnet18_50_communication.npy',
              ax, label='TCT_50', linestyle='-', color='tab:blue')
plot_learning('result_acc/boontk/acc_testtinyimagenet_niid_0.1resnet18_200_communication.npy',
              ax, label='TCT_200', linestyle='-', color='tab:purple')
plot_learning('result_acc/ccvr/acc_testtinyimagenet_niid_0.1resnet18.npy',
              ax, label='CCVR', linestyle='-', color='tab:cyan')
plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.1resnet18_freeze_figure4.npy',
              ax, label='FedForth', linestyle='-', color='tab:red')



# # 0.5
# plot_learning('result_acc/boontk/acc_testtinyimagenet_niid_0.5resnet18_1_communication.npy',
#               ax, label='TCT_1', linestyle='-', color='tab:orange')
# plot_learning('result_acc/boontk/acc_testtinyimagenet_niid_0.5resnet18_10_communication.npy',
#               ax, label='TCT_10', linestyle='-', color='tab:green')
# plot_learning('result_acc/boontk/acc_testtinyimagenet_niid_0.5resnet18_50_communication.npy',
#               ax, label='TCT_50', linestyle='-', color='tab:blue')
# plot_learning('result_acc/boontk/acc_testtinyimagenet_niid_0.5resnet18_200_communication.npy',
#               ax, label='TCT_200', linestyle='-', color='tab:purple')
# plot_learning('result_acc/ccvr/acc_testtinyimagenet_niid_0.5resnet18.npy',
#               ax, label='CCVR', linestyle='-', color='tab:cyan')
# plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.5resnet18_freeze_figure4.npy',
#               ax, label='FedForth', linestyle='-', color='tab:red')

ax.set_xlim([0, 100])
ax.set_ylim([0, 0.6])
ax.set_xticks(np.arange(0,120,20))
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6])
ax.set_xlabel('Global Round')
ax.set_ylabel('Test Accuracy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle='-', alpha=0.3)
legend = ax.legend(
    loc='upper center',          
    bbox_to_anchor=(0.5, -0.17), 
    fontsize=14,                 
    frameon=True,               
    ncol=3,                      
    handlelength=2,            
    columnspacing=0.8,           
    handletextpad=0.5          
)

legend.get_frame().set_alpha(0.8)
legend.get_frame().set_edgecolor('black')

fig.tight_layout()
plt.subplots_adjust(bottom=0.3) 
plt.show()


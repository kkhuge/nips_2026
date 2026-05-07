# FedForth: Frozen Representation with Trainable Head for Linear Convergence and Provable Generalization in Non-IID Federated Learning

This repository contains the codes of the paper FedForth: Frozen Representation with Trainable Head for Linear Convergence and Provable Generalization in Non-IID Federated Learning

## CIFAR-10 Dataset Preparation

```
cd data/cifar10
```
3. Run `generate_cifar_iid.py` to obtain IID CIFAR-10 data.
4. Run the `generate_dirichlet_niid_0.1.py` and `generate_dirichlet_niid_0.5.py` to obtain Dirichlet-based partitions of CIFAR-10 with $\alpha=0.1$ and $\alpha=0.5$.

## CIFAR-100 Dataset Preparation

```
cd data/cifar100
```
Run `generate_dirichlet_niid_0.1.py` and `generate_dirichlet_niid_0.5.py` to obtain Dirichlet-based partitions of CIFAR-100 with $\alpha=0.1$ and $\alpha=0.5$.

```
cd data/tinyimagenet
```
## Tiny-ImageNet Dataset Preparation

Due to repository size limitations, the Tiny-ImageNet dataset is not uploaded directly. You need to download and setup the dataset manually.

### 1. Download and Setup
1. Download **Tiny-ImageNet-200** (e.g., from [Stanford CS231n](http://cs231n.stanford.edu/tiny-imagenet-200.zip)).
2. Navigate to `data/tinyimagenet/` inside this project.
3. **Create a new folder named `data`** inside `tinyimagenet/` if it doesn't exist.
4. Unzip the dataset into that nested `data` folder.

**Correct Directory Structure:**
Ensure your files are organized exactly as shown below. The scripts expect the dataset to be inside a nested `data` folder:

```text
fedavgpy-master/
└── data/
    └── tinyimagenet/
        ├── generate_dirichlet_distribution_0.1_niid.py  <-- Scripts run from here
        ├── generate_dirichlet_distribution_0.5_niid.py
        └── data/                                        <-- Nested 'data' folder
            └── tiny-imagenet-200/                       <-- Dataset goes here
                ├── train/
                ├── val/
                ├── test/
                ├── wnids.txt
                └── words.txt
```

Run `generate_dirichlet_niid_0.1.py` and `generate_dirichlet_niid_0.5.py` to obtain Dirichlet-based partitions of Tiny-Imagenet with $\alpha=0.1$ and $\alpha=0.5$.

## Model： ResNet-18

 The num_class in ```src/models/resnet.py``` is set to 10, 100, 200 for CIFAR-10, CIFAR-100, Tiny-ImageNet dataset.

 ## Improvements on General FL Optimizers

 Run `main.py` using the `fedavg5` trainer for 550 `num_round` to evaluate FedAvg and our FedForth algorithm. You can switch between algorithms by modifying the conditional statement in the LrdWorker class (src/models/worker.py): use ```if round_i < 550:``` for FedAvg and ```if round_i < 450:``` for FedForth.

 Run `main.py` using the `fedavg6` trainer for 550 `num_round` to evaluate FedProx and our FedForth algorithm. You can switch between algorithms by modifying the conditional statement in the ProxWorker class (src/models/worker.py): use ```if round_i < 550:``` for FedProx and ```if round_i < 450:``` for FedForth.

 Run `main.py` using the `scaffold` trainer for 550 `num_round` to evaluate SCAFFOLD and our FedForth algorithm. You can switch between algorithms by modifying the conditional statement in the ScaffoldWorker class (src/models/worker.py): use ```if round_i < 550:``` for SCAFFOLD and ```if round_i < 450:``` for FedForth.

 ## Comparison with Readout-Enhancement Methods

Run `main.py` using the `feduv` trainer for 500 `num_round` with `resnet_uv` model to evaluate FedUV while using the `fedetf` trainer for 500 `num_round` with `resnet_etf` model to evaluate FedETF.

## Comparison with Two-Stage Methods

Run `main.py` using the `boontk` trainer with 1 `num_epoch` to evaluate TCT while using the `ccvr` trainer with 1 `num_epoch` to evaluate CCVR. For the TCT experiments (TCT$_x$), you must update `boontk.py` to align with the multiplier $x$. Specifically, initialize the linear head as `theta_global = torch.zeros(512*x, 200).cuda()` and set the `subsample_size` to $512 \times x$ in compute_eNTK function.

## Sensitivity Analysis

### Figure 4

Run `main.py` using the `fedavg5` trainer for 600 `num_round` to evaluate the impact of Stage 1 length. Adjust the conditional statement ```if round_i < a:``` inside ```src/models/worker.py``` (Class LrdWorker) to set the specific number of rounds for Stage 1 (e.g., a=50)

### Figure 5

Run `main.py` using the `fedavg5` trainer for 550 `num_round` to evaluate the impact of local epochs in Stage 2. Inside ```src/models/worker.py``` (Class LrdWorker), set the conditional statement to ```if round_i < 450:``` and change the Stage 2 loop  ```for i in range(self.num_epoch):``` to ```for i in range(1):``` to obtain the 1 local epoch. Similarly, you can obtain the results for 5 and 50 local epochs by adjusting the range value accordingly.


## Dependency

python = 3.10.19

pytorch = 2.9.0

CUDA = 13.0

Tensordboardx = 2.6.4

Numpy = 2.1.2



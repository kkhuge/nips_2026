from src.trainers.base import BaseTrainer
from src.models.model import choose_model
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class FedAvg4Trainer(BaseTrainer):
    """
    Offline Evaluation for 1D Loss Landscape (Flatness Plot)
    用于评估并绘制 FedAvg 与 FedForth 的损失景观平坦度对比
    """

    def __init__(self, options, dataset, another_dataset):
        # 借用 BaseTrainer 初始化，获取设备和测试集
        # 这里传入一个 dummy optimizer 只是为了满足 BaseTrainer 的初始化要求
        dummy_model = choose_model(options)
        dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.1)
        super().__init__(options, dataset, another_dataset, model=dummy_model, optimizer=dummy_optimizer)

        self.device = options['device']
        self.model = self.worker.model
        # 删除了原来错误的 self.test_dataloader，改为在 test_inference 中直接调用 self.clients 的 dataloader

    def generate_normalized_direction(self, state_dict):
        """生成具备 Filter-wise Normalization 的随机扰动方向"""
        direction = {}
        for key, tensor in state_dict.items():
            if tensor.dtype.is_floating_point:
                d = torch.randn_like(tensor)
                # Filter-wise 归一化: 让扰动方向的范数与原权重的范数对齐
                if tensor.dim() >= 2:  # 卷积层或全连接层
                    for i in range(tensor.size(0)):
                        norm_d = d[i].norm() + 1e-10
                        norm_w = tensor[i].norm()
                        d[i] = d[i] / norm_d * norm_w
                else:  # 偏置项或 1D 张量 (如 BatchNorm)
                    norm_d = d.norm() + 1e-10
                    norm_w = tensor.norm()
                    d = d / norm_d * norm_w
                direction[key] = d
            else:
                direction[key] = torch.zeros_like(tensor)
        return direction

    def apply_perturbation(self, base_state_dict, direction, alpha):
        """将扰动方向乘上标量 alpha 加到基准模型上"""
        perturbed_dict = {}
        for key in base_state_dict.keys():
            if base_state_dict[key].dtype.is_floating_point:
                perturbed_dict[key] = base_state_dict[key] + alpha * direction[key]
            else:
                perturbed_dict[key] = base_state_dict[key]
        return perturbed_dict

    def test_inference(self, model):
        """在所有客户端的测试集上跑前向传播，计算全局 Loss 和 Acc"""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            # 核心修复：直接遍历 BaseTrainer 帮我们建好的所有 clients
            for c in self.clients:
                # 使用每个 client 内部现成的 test_dataloader
                for images, labels in c.test_dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    batch_loss = criterion(outputs, labels)
                    loss += batch_loss.item() * labels.size(0)

                    _, pred_labels = torch.max(outputs, 1)
                    correct += torch.sum(pred_labels == labels).item()
                    total += labels.size(0)

        accuracy = correct / total
        loss = loss / total
        return accuracy, loss

    def train(self):
        """
        重写 train 方法，使其变为测试平坦度并画图的方法。
        通过 main.py 运行此文件时，会直接执行这个画图逻辑。
        """
        # ==================== 模型路径配置 ====================
        # 确保这两个 .pth 模型权重文件存在
        model_fedavg_path = "stage1_resnet18_cifar10_0.1_fedavg.pth"
        model_fedforth_path = "stage1_resnet18_cifar10_0.1_fedforce.pth"
        # ==========================================================

        if not os.path.exists(model_fedavg_path) or not os.path.exists(model_fedforth_path):
            print(f"Error: 请确保 {model_fedavg_path} 和 {model_fedforth_path} 存在！")
            return

        print(f"\n[1/4] Loading FedAvg model from {model_fedavg_path}")
        state_fedavg = torch.load(model_fedavg_path, map_location='cpu')

        print(f"[2/4] Loading FedForth model from {model_fedforth_path}")
        state_fedforth = torch.load(model_fedforth_path, map_location='cpu')

        # 核心逻辑1：生成统一的、对齐范数的随机扰动方向 (以 FedAvg 为基准)
        print("\n[3/4] Generating normalized random direction...")
        direction = self.generate_normalized_direction(state_fedavg)

        # 核心逻辑2：定义扰动范围（横坐标）
        alphas = np.linspace(-0.5, 0.5, 21)
        loss_fedavg, acc_fedavg = [], []
        loss_fedforth, acc_fedforth = [], []

        print("\n[4/4] Starting 1D Loss Landscape evaluation...")
        for alpha in tqdm(alphas, desc="Evaluating alphas"):
            # --- 测试 FedAvg (陡峭测试) ---
            perturbed_fedavg = self.apply_perturbation(state_fedavg, direction, alpha)
            self.model.load_state_dict(perturbed_fedavg)
            self.model.to(self.device)
            acc1, loss1 = self.test_inference(self.model)
            loss_fedavg.append(loss1)
            acc_fedavg.append(acc1)

            # --- 测试 FedForth (平坦测试) ---
            # 必须使用与 FedAvg 绝对相同的 direction
            perturbed_fedforth = self.apply_perturbation(state_fedforth, direction, alpha)
            self.model.load_state_dict(perturbed_fedforth)
            self.model.to(self.device)
            acc2, loss2 = self.test_inference(self.model)
            loss_fedforth.append(loss2)
            acc_fedforth.append(acc2)

        # 绘制并保存结果图表
        self.plot_and_save(alphas, loss_fedavg, loss_fedforth, acc_fedavg, acc_fedforth)

    def plot_and_save(self, alphas, loss_fedavg, loss_fedforth, acc_fedavg, acc_fedforth):
        plt.figure(figsize=(12, 5))

        # 子图 1: Test Loss (平坦度)
        plt.subplot(1, 2, 1)
        plt.plot(alphas, loss_fedavg, label='FedAvg', marker='o')
        plt.plot(alphas, loss_fedforth, label='FedForth', marker='s')
        plt.xlabel('Perturbation ($\\alpha$)')
        plt.ylabel('Test Loss')
        plt.title('1D Loss Landscape (Flatness)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # 子图 2: Test Accuracy (鲁棒性)
        plt.subplot(1, 2, 2)
        plt.plot(alphas, acc_fedavg, label='FedAvg', marker='o')
        plt.plot(alphas, acc_fedforth, label='FedForth', marker='s')
        plt.xlabel('Perturbation ($\\alpha$)')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy Robustness')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plot_name = 'landscape_comparison.png'
        plt.savefig(plot_name, dpi=300)
        print(f"\n✅ Evaluation complete! The landscape plot is saved to {plot_name}")
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
    Evaluate and plot the loss-landscape flatness comparison between FedAvg and FedForth.
    """

    def __init__(self, options, dataset, another_dataset):
        # Reuse BaseTrainer initialization to get the device and test set.
        # Pass a dummy optimizer only to satisfy BaseTrainer initialization requirements.
        dummy_model = choose_model(options)
        dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.1)
        super().__init__(options, dataset, another_dataset, model=dummy_model, optimizer=dummy_optimizer)

        self.device = options['device']
        self.model = self.worker.model
        # Removed the incorrect self.test_dataloader; test_inference directly uses dataloaders from self.clients.

    def generate_normalized_direction(self, state_dict):
        """Generate random perturbation directions with filter-wise normalization."""
        direction = {}
        for key, tensor in state_dict.items():
            if tensor.dtype.is_floating_point:
                d = torch.randn_like(tensor)
                # Filter-wise normalization: align the perturbation norm with the original weight norm.
                if tensor.dim() >= 2:  # Convolutional or fully connected layers.
                    for i in range(tensor.size(0)):
                        norm_d = d[i].norm() + 1e-10
                        norm_w = tensor[i].norm()
                        d[i] = d[i] / norm_d * norm_w
                else:  # Bias terms or 1D tensors such as BatchNorm.
                    norm_d = d.norm() + 1e-10
                    norm_w = tensor.norm()
                    d = d / norm_d * norm_w
                direction[key] = d
            else:
                direction[key] = torch.zeros_like(tensor)
        return direction

    def apply_perturbation(self, base_state_dict, direction, alpha):
        """Add the perturbation direction scaled by alpha to the base model."""
        perturbed_dict = {}
        for key in base_state_dict.keys():
            if base_state_dict[key].dtype.is_floating_point:
                perturbed_dict[key] = base_state_dict[key] + alpha * direction[key]
            else:
                perturbed_dict[key] = base_state_dict[key]
        return perturbed_dict

    def test_inference(self, model):
        """Run forward passes on all client test sets and compute global loss and accuracy."""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            # Core fix: directly iterate over all clients created by BaseTrainer.
            for c in self.clients:
                # Use each client's existing test_dataloader.
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
        Override the train method so it tests flatness and plots the result.
        When this file is run through main.py, it executes this plotting logic directly.
        """
        # ==================== Model path configuration ====================
        # Ensure these two .pth model weight files exist.
        model_fedavg_path = "stage1_resnet18_cifar10_0.1_fedavg.pth"
        model_fedforth_path = "stage1_resnet18_cifar10_0.1_fedforce.pth"
        # ==========================================================

        if not os.path.exists(model_fedavg_path) or not os.path.exists(model_fedforth_path):
            print(f"Error: ensure {model_fedavg_path} and {model_fedforth_path} exist.")
            return

        print(f"\n[1/4] Loading FedAvg model from {model_fedavg_path}")
        state_fedavg = torch.load(model_fedavg_path, map_location='cpu')

        print(f"[2/4] Loading FedForth model from {model_fedforth_path}")
        state_fedforth = torch.load(model_fedforth_path, map_location='cpu')

        # Core logic 1: generate unified random perturbation directions with aligned norms, using FedAvg as the base.
        print("\n[3/4] Generating normalized random direction...")
        direction = self.generate_normalized_direction(state_fedavg)

        # Core logic 2: define the perturbation range on the x-axis.
        alphas = np.linspace(-0.5, 0.5, 21)
        loss_fedavg, acc_fedavg = [], []
        loss_fedforth, acc_fedforth = [], []

        print("\n[4/4] Starting 1D Loss Landscape evaluation...")
        for alpha in tqdm(alphas, desc="Evaluating alphas"):
            # --- Test FedAvg for steepness ---
            perturbed_fedavg = self.apply_perturbation(state_fedavg, direction, alpha)
            self.model.load_state_dict(perturbed_fedavg)
            self.model.to(self.device)
            acc1, loss1 = self.test_inference(self.model)
            loss_fedavg.append(loss1)
            acc_fedavg.append(acc1)

            # --- Test FedForth for flatness ---
            # Must use exactly the same direction as FedAvg.
            perturbed_fedforth = self.apply_perturbation(state_fedforth, direction, alpha)
            self.model.load_state_dict(perturbed_fedforth)
            self.model.to(self.device)
            acc2, loss2 = self.test_inference(self.model)
            loss_fedforth.append(loss2)
            acc_fedforth.append(acc2)

        # Plot and save the result figure.
        self.plot_and_save(alphas, loss_fedavg, loss_fedforth, acc_fedavg, acc_fedforth)

    def plot_and_save(self, alphas, loss_fedavg, loss_fedforth, acc_fedavg, acc_fedforth):
        plt.figure(figsize=(12, 5))

        # Subplot 1: Test Loss, flatness.
        plt.subplot(1, 2, 1)
        plt.plot(alphas, loss_fedavg, label='FedAvg', marker='o')
        plt.plot(alphas, loss_fedforth, label='FedForth', marker='s')
        plt.xlabel('Perturbation ($\\alpha$)')
        plt.ylabel('Test Loss')
        plt.title('1D Loss Landscape (Flatness)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Subplot 2: Test Accuracy, robustness.
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

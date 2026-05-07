from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from torch.optim import SGD
import numpy as np
import torch
import os
from src.models.client import Client

# 初始化结果保存目录
for d in ["result_loss/fedper", "result_acc/fedper"]:
    if not os.path.exists(d):
        os.makedirs(d)

class FedPerWorker(LrdWorker):
    """
    FedPer Worker: 实现基础层 (WB) 与个性化层 (WP) 的逻辑分离 [cite: 9]
    """
    def __init__(self, model, optimizer, optimizer_last_layer, options):
        super(FedPerWorker, self).__init__(model, optimizer, optimizer_last_layer, options)

    def get_base_params(self):
        """提取基础层参数用于全球聚合 [cite: 43]"""
        params = []
        for name, param in self.model.named_parameters():
            if "readout" not in name:
                params.append(param.data.view(-1))
        return torch.cat(params)

    def set_base_params(self, flat_params):
        """同步服务器的基础层参数"""
        offset = 0
        for name, param in self.model.named_parameters():
            if "readout" not in name:
                numel = param.numel()
                param.data.copy_(flat_params[offset:offset + numel].view_as(param.data))
                offset += numel


class FedPerTrainer(BaseTrainer):
    def __init__(self, options, dataset, another_dataset):
        self.options = options
        self.device = torch.device('cuda' if options['gpu'] else 'cpu')

        # 1. 临时模型用于获取维度
        temp_model = choose_model(options)
        if options['gpu']:
            temp_model = temp_model.to(self.device)

        temp_worker = FedPerWorker(temp_model, None, None, options)
        self.latest_model = temp_worker.get_base_params().detach().to(self.device)

        self.clients_per = []
        self.acc_list_test = []
        self.loss_list_test = []

        super(FedPerTrainer, self).__init__(options, dataset, another_dataset, worker=temp_worker)

        # 4. 显式创建独立的个性化客户端 [cite: 42, 120]
        self.clients_per = self.setup_clients_per(dataset, another_dataset)

    def setup_clients_per(self, dataset, another_dataset):
        users, groups, train_data, test_data = dataset
        _, _, another_train_data, another_test_data = another_dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            user_id = int(user[-5:]) if isinstance(user, str) and len(user) >= 5 else int(user)

            local_model = choose_model(self.options).to('cpu')
            optimizer = SGD(local_model.parameters(), lr=self.options['lr'], weight_decay=0.0001)
            optimizer_last_layer = SGD(local_model.readout.parameters(), lr=self.options['lr'], weight_decay=0.001)

            local_worker = FedPerWorker(local_model, optimizer, optimizer_last_layer, self.options)

            c_per = Client(user_id, group, train_data[user], test_data[user],
                       another_train_data[user], another_test_data[user],
                       self.batch_size, local_worker)
            all_clients.append(c_per)

        return all_clients

    def select_clients_per(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients_per))
        np.random.seed(seed)
        return np.random.choice(self.clients_per, num_clients, replace=False).tolist()

    def train(self):
        print(f'>>> FedPer Training: Base + Personalization layers architecture [cite: 9]')

        for round_i in range(self.num_round):
            # 1. 评估
            if len(self.clients_per) > 0:
                self.evaluate_personalized(round_i)

            # 2. 选择客户端
            selected_clients = self.select_clients_per(seed=round_i)
            solns = []
            stats = []

            for c in selected_clients:
                # 显存管理：按需移动到 GPU
                c.worker.model.to(self.device)

                # =================== 核心修复 [FIX] ===================
                # 对应论文 Algorithm 1: "Server sends current representation phi^t to these clients"
                # 这一步必须做：确保客户端在训练 Head 之前，拿到的是最新的全局 Body
                c.worker.set_base_params(self.latest_model)
                # ====================================================

                # B. 本地训练：在最新的 phi^t 基础上，先训练 Head，再更新 Body [cite: 135, 142]
                (num_sample, _), stat = c.local_train(round_i)

                # C. 提取 WB 更新 (Body) 发回给服务器 [cite: 143]
                base_params = c.worker.get_base_params().detach().to(self.device)
                solns.append((num_sample, base_params))
                stats.append(stat)

                # 显存管理：移回 CPU
                c.worker.model.to('cpu')

            self.metrics.extend_commu_stats(round_i, stats)

            # 3. 聚合：服务器更新全局 Representation phi^{t+1} [cite: 143]
            self.latest_model = self.aggregate(solns)

        self.evaluate_personalized(1000)

        self.save_all_results()
        self.metrics.write()

    def aggregate(self, solns):
        if not solns: return self.latest_model
        total_samples = sum([s[0] for s in solns])
        avg_base = torch.zeros_like(solns[0][1])
        for num_sample, local_base in solns:
            avg_base += (num_sample / total_samples) * local_base.to(self.device)
        return avg_base.detach()

    def evaluate_personalized(self, round_i):
        correct_list = []
        num_list = []
        loss_list = []
        criterion = torch.nn.CrossEntropyLoss()

        for c in self.clients_per:
            c.worker.model.to(self.device)

            # 测试时同样需要同步最新的全局 Body，以配合本地的 Head [cite: 95]
            c.worker.set_base_params(self.latest_model)
            c.worker.model.eval()

            test_loss = 0.
            test_acc = 0
            test_total = 0

            with torch.no_grad():
                for data, target in c.test_dataloader:
                    data, target = data.to(self.device), target.to(self.device)
                    pred = c.worker.model(data)
                    loss = criterion(pred, target)
                    _, predicted = torch.max(pred, 1)
                    cor = predicted.eq(target).sum().item()
                    test_acc += cor
                    test_loss += loss.item() * target.size(0)
                    test_total += target.size(0)

            correct_list.append(test_acc)
            num_list.append(test_total)
            loss_list.append(test_loss)
            c.worker.model.to('cpu')

        num_all = np.sum(num_list)
        avg_acc = np.sum(correct_list) / num_all
        avg_loss = np.sum(loss_list) / num_all
        self.acc_list_test.append(avg_acc)
        self.loss_list_test.append(avg_loss)
        print(f'Round {round_i} - Personalized Avg Acc: {avg_acc:.4f}, Loss: {avg_loss:.4f}')

    def save_all_results(self):
        ds = self.options.get("dataset", "data")
        md = self.options.get("model", "model")
        np.save(f'result_acc/fedper/acc_test_{ds}_{md}', self.acc_list_test)
        print(">>> Results saved.")
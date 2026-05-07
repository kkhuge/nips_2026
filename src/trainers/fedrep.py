from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from torch.optim import SGD
import numpy as np
import torch
import os
import copy
from src.models.client import Client

# 初始化结果保存目录
for d in ["result_loss/fedrep", "result_acc/fedrep"]:
    if not os.path.exists(d):
        os.makedirs(d)


class FedRepWorker(LrdWorker):
    """
    FedRep Worker: 实现 Body (Representation) 与 Head 的交替更新
    """

    def __init__(self, model, optimizer, options):
        # 注意：FedRep 需要两个优化器，或者在训练过程中动态切换优化参数
        # 这里我们传入主优化器，但在 train 方法中我们会根据阶段冻结参数
        super(FedRepWorker, self).__init__(model, optimizer, optimizer,options)

        # 定义 Body 和 Head 的参数组
        # 假设 model.readout 是 Head，其余部分是 Body (Representation)
        self.body_params = []
        self.head_params = []
        self.num_head_epochs = options['num_epoch']
        for name, param in self.model.named_parameters():
            if "readout" in name:
                self.head_params.append(param)
            else:
                self.body_params.append(param)

        # 为两个阶段分别创建优化器，避免 momentum 状态混淆
        # Phase 1: 只优化 Head
        self.optimizer_head = SGD(self.head_params, lr=options['lr'], weight_decay=options['wd'])
        # Phase 2: 只优化 Body
        self.optimizer_body = SGD(self.body_params, lr=options['lr'], weight_decay=options['wd'])

    def get_body_params(self):
        """提取 Representation (Body) 参数用于全球聚合"""
        params = []
        for name, param in self.model.named_parameters():
            if "readout" not in name:
                params.append(param.data.view(-1))
        return torch.cat(params)

    def set_body_params(self, flat_params):
        """同步服务器的 Body 参数，保持本地 Head 不变"""
        offset = 0
        for name, param in self.model.named_parameters():
            if "readout" not in name:
                numel = param.numel()
                param.data.copy_(flat_params[offset:offset + numel].view_as(param.data))
                offset += numel

    def train(self, train_dataloader, round_i, client_id=None):
        """
        FedRep 的核心交替训练逻辑
        Phase 1: Freeze Body, Train Head (for multiple epochs)
        Phase 2: Freeze Head, Train Body (for 1 epoch)
        """
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()

        # --- Phase 1: Train Head ---
        # 冻结 Body
        for param in self.body_params:
            param.requires_grad = False
        for param in self.head_params:
            param.requires_grad = True

        # Head 训练通常需要更多轮次来适应新的 Body
        # 使用 options['num_epoch'] 作为 Head 的训练轮数

        for epoch in range(self.num_head_epochs):
            for x, y in train_dataloader:
                x, y = x.cuda(), y.cuda()
                self.optimizer_head.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head_params, max_norm=10.0)
                self.optimizer_head.step()

        # --- Phase 2: Train Body ---
        # 冻结 Head
        for param in self.body_params:
            param.requires_grad = True
        for param in self.head_params:
            param.requires_grad = False

        # Body 训练通常只进行 1 个 epoch (或更少步数)
        # 也可以通过参数控制，这里默认为 1
        num_body_epochs = 1

        train_stats = {}  # 简单记录一下

        for epoch in range(num_body_epochs):
            total_loss = 0
            correct = 0
            total_samples = 0

            for x, y in train_dataloader:
                x, y = x.cuda(), y.cuda()
                self.optimizer_body.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.body_params, max_norm=10.0)
                self.optimizer_body.step()

                total_loss += loss.item() * y.size(0)
                _, predicted = torch.max(pred, 1)
                correct += predicted.eq(y).sum().item()
                total_samples += y.size(0)

            train_stats['loss'] = total_loss / total_samples
            train_stats['acc'] = correct / total_samples

        # 训练结束后，恢复所有参数的梯度需求（虽然下一次 train 会重新设置，但为了安全）
        for param in self.model.parameters():
            param.requires_grad = True

        return train_stats


class FedRepTrainer(BaseTrainer):
    def __init__(self, options, dataset, another_dataset):
        self.options = options
        self.device = torch.device('cuda' if options['gpu'] else 'cpu')

        # 1. 初始化一个临时模型用于获取参数形状
        temp_model = choose_model(options)
        if options['gpu']:
            temp_model = temp_model.to(self.device)

        # 临时 worker
        # 这里的 optimizer 传 None 即可，因为 Worker 内部会自己重新定义
        temp_worker = FedRepWorker(temp_model, None,  options)

        # FedRep 聚合的是 Representation (Body)，所以这里获取 Body 参数
        self.latest_body = temp_worker.get_body_params().detach().to(self.device)

        # 2. 初始化结果列表
        self.clients_rep = []
        self.acc_list_test = []
        self.loss_list_test = []

        # 3. 调用父类初始化 (父类会创建 self.worker = temp_worker)
        super(FedRepTrainer, self).__init__(options, dataset, another_dataset, worker=temp_worker)

        # 4. 核心：为每个客户端创建独立的 Worker 和 Model
        self.clients_rep = self.setup_clients_rep(dataset, another_dataset)

    def setup_clients_rep(self, dataset, another_dataset):
        """
        为每个客户端分配【完全独立】的模型，以保持 Head 的个性化状态
        """
        users, groups, train_data, test_data = dataset
        _, _, another_train_data, another_test_data = another_dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            user_id = int(user[-5:]) if isinstance(user, str) and len(user) >= 5 else int(user)

            # --- 关键点：每个客户端拥有独立的模型对象 ---
            local_model = choose_model(self.options).to('cpu')

            # 创建 FedRep 专用的 Worker
            # 这里的 optimizer 参数其实主要为了占位兼容父类接口，
            # 真正的优化器在 FedRepWorker.__init__ 内部创建了两个 (head & body)
            dummy_optimizer = SGD(local_model.parameters(), lr=self.options['lr'])
            local_worker = FedRepWorker(local_model, dummy_optimizer, self.options)

            # 创建 Client 对象
            c_rep = Client(user_id, group, train_data[user], test_data[user],
                           another_train_data[user], another_test_data[user],
                           self.batch_size, local_worker)
            all_clients.append(c_rep)

        return all_clients

    def select_clients_rep(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients_rep))
        np.random.seed(seed)
        return np.random.choice(self.clients_rep, num_clients, replace=False).tolist()

    def train(self):
        print(f'>>> FedRep Training: Alternating updates (Head -> Body)')

        for round_i in range(self.num_round):
            # 1. 评估：测试个性化性能 (Global Body + Local Head)
            if len(self.clients_rep) > 0:
                self.evaluate_personalized(round_i)

            # 2. 选择客户端
            selected_clients = self.select_clients_rep(seed=round_i)
            solns = []
            stats = []

            for c in selected_clients:
                # 显存管理
                c.worker.model.to(self.device)
                c.worker.device = self.device

                # A. 下载最新的 Global Representation (Body)
                # 注意：Head 保持上一轮本地训练结束时的状态，不动它
                c.worker.set_body_params(self.latest_body)

                # B. 本地训练 (FedRep 逻辑：先修头，再修身)
                # 使用 Client.local_train -> 调用 Worker.train
                # 注意：这里我们修改一下 Client 的 local_train 行为，
                # 或者直接调用 worker.train 比较稳妥，因为 Worker.train 里写了特殊的交替逻辑

                # 获取数据加载器
                train_loader = c.train_dataloader

                # 执行 FedRep 特有的交替训练
                # train 返回的是统计数据
                stat = c.worker.train(train_loader, round_i)
                num_sample = len(c.train_data)

                # C. 提取更新后的 Body 参数用于上传
                # Head 参数留在本地模型中，随对象持久化
                body_params = c.worker.get_body_params().detach().to(self.device)

                solns.append((num_sample, body_params))
                stats.append(stat)

                # 显存管理：移回 CPU
                c.worker.model.to('cpu')

            # 3. 聚合：服务器更新 Representation (Body)
            self.latest_body = self.aggregate(solns)

        # 最终评估
        self.evaluate_personalized(self.num_round)
        self.save_all_results()
        self.metrics.write()

    def aggregate(self, solns):
        """标准的 FedAvg 聚合，但仅针对 Body 参数"""
        if not solns: return self.latest_body
        total_samples = sum([s[0] for s in solns])
        avg_body = torch.zeros_like(solns[0][1])
        for num_sample, local_body in solns:
            avg_body += (num_sample / total_samples) * local_body.to(self.device)
        return avg_body.detach()

    def evaluate_personalized(self, round_i):
        """
        评估每个客户端的个性化准确率 (Global Body + Local Head)
        """
        correct_list = []
        num_list = []
        loss_list = []
        criterion = torch.nn.CrossEntropyLoss()

        for c in self.clients_rep:
            c.worker.model.to(self.device)
            c.worker.device = self.device

            # 确保评估时使用的是最新的 Global Body
            c.worker.set_body_params(self.latest_body)

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
                    test_acc += predicted.eq(target).sum().item()
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

        # 修正：防止 options['noprint'] 不存在时报错
        noprint = self.options.get('noprint', False)
        if not noprint:
            print(f'Round {round_i} - FedRep Personalized Avg Acc: {avg_acc:.4f}, Loss: {avg_loss:.4f}')

    def save_all_results(self):
        ds = self.options.get("dataset", "data")
        md = self.options.get("model", "model")
        np.save(f'result_acc/fedrep/acc_test_{ds}_{md}', self.acc_list_test)
        np.save(f'result_loss/fedrep/loss_test_{ds}_{md}', self.loss_list_test)
        print(">>> FedRep Results saved.")
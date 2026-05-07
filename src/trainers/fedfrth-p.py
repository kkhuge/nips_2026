from sympy.polys.subresultants_qq_zz import correct_sign

from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from torch.optim import SGD
import numpy as np
import torch
import os
import torch.nn as nn
import copy
from tqdm import tqdm
import torch.nn.functional as F


criterion = torch.nn.CrossEntropyLoss()


loss_dir = "result_loss/fedfrth-p"
acc_dir = "result_acc/fedfrth-p"
if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)



class FedFRTHPTrainer(BaseTrainer):
    """
    Original Scheme
    """
    def __init__(self, options, dataset, another_dataset):
        self.theta_0 = 0
        self.error_train = []
        self.loss_list_train = []
        self.acc_list_train = []
        self.loss_list_test = []
        self.acc_list_test = []
        self.theta = []
        self.diff_nonlinear_linear = []
        self.weight_change = []
        self.options = options
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.required_accuracy = options['psi']
        self.tau = options['num_epoch']
        self.learning_rate = options['lr']
        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0001)
        self.optimizer_last_layer = SGD(model.readout.parameters(), lr=options['lr'], weight_decay=0.0001)
        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']
        worker = LrdWorker(model, self.optimizer, self.optimizer_last_layer, options)
        super(FedFRTHPTrainer, self).__init__(options, dataset, another_dataset, worker=worker)

    def train(self):
        # print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        # # Fetch latest flat model parameter
        # self.latest_model = self.worker.get_flat_model_params().detach()
        # for round_i in range(self.num_round):
        #     # Test latest model on train data
        #     # _, accuracy, loss = self.test_latest_model_on_traindata(round_i)
        #     # self.acc_list_train.append(accuracy)
        #     # self.loss_list_train.append(loss)
        #     loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
        #     self.acc_list_test.append(accuracy_test)
        #     self.loss_list_test.append(loss_test)
        #
        #     # Choose K clients prop to data size
        #     selected_clients = self.select_clients(seed=round_i)
        #
        #     # Solve minimization locally
        #     solns, stats = self.local_train(round_i, selected_clients)
        #
        #     # Track communication cost
        #     self.metrics.extend_commu_stats(round_i, stats)
        #
        #     # Update latest model
        #     self.latest_model = self.aggregate(solns)
        #
        #     self.worker.set_flat_model_params(self.latest_model)
        #
        #
        # # Test final model on train data
        # # _, accuracy, loss = self.test_latest_model_on_traindata(self.num_round)
        # # self.acc_list_train.append(accuracy)
        # # self.loss_list_train.append(loss)
        # loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        # self.acc_list_test.append(accuracy_test)
        # self.loss_list_test.append(loss_test)
        # # ===== 保存第一阶段模型 =====
        # torch.save(self.clients[0].worker.model.state_dict(), "FedFRTH-P_stage1_resnet18_cifar100_0.1.pth")

        # ================= 第二阶段：个性化微调 (Personalized Fine-tuning) =================
        print('===================== Start Standard Finetuning with New Models =====================')

        # 1. 加载联邦训练好的全局模型权重
        global_state_dict = torch.load("FedFRTH-P_stage1_resnet18_cifar100_0.1.pth")

        # 论文设定：微调 epoch 数
        finetune_epochs = 200

        # 初始化矩阵来存储数据：[Client数, Epoch数]
        # 修改：不再存 Acc，而是存 Correct 数值和 Total 数值，以便最后计算加权精度
        num_clients = len(self.clients)
        client_correct_matrix = np.zeros((num_clients, finetune_epochs))
        client_loss_sum_matrix = np.zeros((num_clients, finetune_epochs))
        client_sample_matrix = np.zeros((num_clients, finetune_epochs))

        # --- 外层循环遍历 Client ---
        for client_idx, client in enumerate(tqdm(self.clients, desc="Finetuning Clients")):

            # 为当前 client 创建一个完全独立的新模型
            local_model = choose_model(self.options)

            # 将模型移动到 GPU
            if self.options['gpu']:
                local_model = local_model.cuda()

            # 2. 初始化：加载第一阶段训练好的 Body + 随机 Head
            local_model.load_state_dict(global_state_dict)

            # 3. 设置微调优化器 (更新全参数)
            optimizer = torch.optim.SGD(local_model.readout.parameters(),
                                        lr=0.1,
                                        weight_decay=0.0001)

            # --- 内层循环：该客户端连续训练 N 个 Epoch ---
            for epoch in range(finetune_epochs):

                # --- A. 训练 ---
                local_model.train()
                for x, y in client.train_dataloader:
                    if self.options['gpu']:
                        x, y = x.cuda(), y.cuda()

                    optimizer.zero_grad()
                    pred = local_model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 60)
                    optimizer.step()

                # --- B. 测试 (每微调一轮就测一次) ---
                local_model.eval()
                test_loss_sum = 0.
                test_correct = 0.
                test_total = 0.

                with torch.no_grad():
                    for data, target in client.test_dataloader:
                        if self.options['gpu']:
                            data, target = data.cuda(), target.cuda()

                        pred = local_model(data)
                        loss = criterion(pred, target)
                        _, predicted = torch.max(pred, 1)

                        test_correct += predicted.eq(target).sum().item()
                        # loss.item() 是平均值，乘以 batch_size 还原为 sum
                        test_loss_sum += loss.item() * target.size(0)
                        test_total += target.size(0)
                print(test_correct/test_total)

                # 存入矩阵 (存储原始数值，而不是比率)
                client_correct_matrix[client_idx][epoch] = test_correct
                client_loss_sum_matrix[client_idx][epoch] = test_loss_sum
                client_sample_matrix[client_idx][epoch] = test_total

        # ================= 整理并保存数据 =================

        print('\n>>> Finetuning Summary (Weighted Average):')
        for epoch in range(finetune_epochs):
            # 计算这一轮 epoch 所有客户端的总正确数 / 总样本数
            total_correct = np.sum(client_correct_matrix[:, epoch])
            total_loss_sum = np.sum(client_loss_sum_matrix[:, epoch])
            total_samples = np.sum(client_sample_matrix[:, epoch])

            # 加权精度 (Weighted Acc)
            weighted_acc = total_correct / total_samples
            # 加权损失 (Weighted Loss)
            weighted_loss = total_loss_sum / total_samples

            print('Epoch: {}; Acc: {:.4f}; Loss: {:.4f}'.format(epoch, weighted_acc, weighted_loss))

            # 追加到列表
            self.loss_list_test.append(weighted_loss)
            self.acc_list_test.append(weighted_acc)

        # 保存
        np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_finetuning', self.loss_list_test)
        np.save(acc_dir + '/acc_test' + self.dataset + self.model+ '_finetuning', self.acc_list_test)


    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()



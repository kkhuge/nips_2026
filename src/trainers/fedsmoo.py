from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import FedSMOOWorker
from torch.optim import SGD
import numpy as np
import torch
import os

loss_dir = "result_loss/fedsmoo"
acc_dir = "result_acc/fedsmoo"

if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)


class FedSMOOTrainer(BaseTrainer):
    def __init__(self, options, dataset, another_dataset):
        self.error_train = []
        self.loss_list_train = []
        self.acc_list_train = []
        self.loss_list_test = []
        self.acc_list_test = []

        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0001)
        self.optimizer_last_layer = SGD(model.readout.parameters(),lr=options['lr'], weight_decay=0.0001)

        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.model = options['model']

        # 初始化 FedSMOO 的 worker
        worker = FedSMOOWorker(model, self.optimizer, self.optimizer_last_layer, options)
        super(FedSMOOTrainer, self).__init__(options, dataset, another_dataset, worker=worker)

        # --- FedSMOO 的全局变量 ---
        # 全局微扰向量 s
        self.global_s = [torch.zeros_like(p.data).to(self.device) for p in self.worker.model.parameters()]
        # 全局模型对偶变量 lambda
        self.global_lambda = [torch.zeros_like(p.data).to(self.device) for p in self.worker.model.parameters()]
        self.rho = 0.1  # 0.01
        self.beta = 10  # 50

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):
            if round_i >= 450:
                _, accuracy, loss = self.test_latest_model_on_traindata(round_i)
                self.acc_list_train.append(accuracy)
                self.loss_list_train.append(loss)
                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
                self.acc_list_test.append(accuracy_test)
                self.loss_list_test.append(loss_test)
            else:
                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)

            selected_clients = self.select_clients(seed=round_i)

            # 将 self.latest_model (flat shape) 转化为 tensor list 传给客户端
            global_w = []
            idx = 0
            for p in self.worker.model.parameters():
                numel = p.numel()
                global_w.append(self.latest_model[idx:idx + numel].view_as(p).clone().detach())
                idx += numel

            # 执行本地 FedSMOO 训练
            solns, stats, tilde_s_list = self.local_train_smoo(round_i, selected_clients, global_w, self.global_s)

            self.metrics.extend_commu_stats(round_i, stats)

            # ======== FedSMOO 全局聚合阶段 ========

            # 1. 更新全局微扰 s^{t+1} (Eq. 11: s = \frac{1}{n} \sum \tilde{s}_i)
            bar_s = [torch.zeros_like(gs) for gs in self.global_s]
            for tilde_s in tilde_s_list:
                for idx_param, ts in enumerate(tilde_s):
                    bar_s[idx_param] += ts / len(selected_clients)

            # 对 bar_s 投影到半径 r 的球内 (r = self.rho)
            bar_s_norm_sq = sum(torch.sum(bs ** 2) for bs in bar_s)
            bar_s_norm = torch.sqrt(bar_s_norm_sq)
            if bar_s_norm > 0:
                for idx_param in range(len(self.global_s)):
                    self.global_s[idx_param] = self.rho * bar_s[idx_param] / bar_s_norm
            else:
                for idx_param in range(len(self.global_s)):
                    self.global_s[idx_param] = torch.zeros_like(bar_s[idx_param])

            # 2. 计算本轮的中间聚合模型 \frac{1}{n}\sum w_i^t
            old_latest_model_flat = torch.cat([gw.flatten() for gw in global_w])
            self.latest_model = self.aggregate(solns)

            # ================= 关键修改：只在第一阶段应用 SAM 和 ADMM 惩罚 =================
            if round_i < 2000:

                # 3. 更新全局对偶变量 lambda^{t+1} (Algorithm 1 Line 18)
                # lambda^{t+1} = lambda^t - (1/(beta*m)) * \sum_{i \in [n]} (w_i^t - w^t)
                sum_delta_w_flat = len(selected_clients) * (self.latest_model - old_latest_model_flat)
                global_lambda_flat = torch.cat([gl.flatten() for gl in self.global_lambda])
                m = len(self.clients)  # 客户端总数
                global_lambda_flat = global_lambda_flat - (1.0 / (self.beta * m)) * sum_delta_w_flat

                idx = 0
                for gl in self.global_lambda:
                    numel = gl.numel()
                    gl.copy_(global_lambda_flat[idx:idx + numel].view_as(gl))
                    idx += numel

                # 4. 更新全局模型 w^{t+1} (Algorithm 1 Line 19)
                # w^{t+1} = \frac{1}{n}\sum w_i^t - \beta \lambda^{t+1}
                self.latest_model = self.latest_model - self.beta * global_lambda_flat
            self.worker.set_flat_model_params(self.latest_model)

        # 保存结果
        _, accuracy, loss = self.test_latest_model_on_traindata(self.num_round)
        self.acc_list_train.append(accuracy)
        self.loss_list_train.append(loss)
        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        self.acc_list_test.append(accuracy_test)
        self.loss_list_test.append(loss_test)

        # np.save(loss_dir + '/loss_train_' + self.dataset + '_' + self.model, self.loss_list_train)
        # np.save(acc_dir + '/acc_train_' + self.dataset + '_' + self.model, self.acc_list_train)
        np.save(loss_dir + '/loss_test_' + self.dataset + '_' + self.model, self.loss_list_test)
        np.save(acc_dir + '/acc_test_' + self.dataset + '_' + self.model, self.acc_list_test)
        # np.save(loss_dir + '/loss_test_' + self.dataset + '_' + self.model + '_freeze_re', self.loss_list_test)
        # np.save(acc_dir + '/acc_test_' + self.dataset + '_' + self.model + '_freeze_re', self.acc_list_test)

        self.metrics.write()


    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()
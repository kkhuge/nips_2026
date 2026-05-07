from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import FedSAMWorker
from torch.optim import SGD
import numpy as np
import torch
import os

criterion = torch.nn.CrossEntropyLoss()

loss_dir = "result_loss/fedsam"
acc_dir = "result_acc/fedsam"

if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)


class FedSAMTrainer(BaseTrainer):
    """
    FedSAM / FedASAM + SWA Trainer (Single Accuracy Curve)
    """

    def __init__(self, options, dataset, another_dataset):
        self.loss_list_test = []
        self.acc_list_test = []

        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0001)
        self.optimizer_last_layer = SGD(model.readout.parameters(), lr=options['lr'], weight_decay=0.0001)

        self.num_round = options['num_round']
        self.dataset = options["dataset"]
        self.model = options['model']

        worker = FedSAMWorker(model, self.optimizer, self.optimizer_last_layer, options)
        super(FedSAMTrainer, self).__init__(options, dataset, another_dataset, worker=worker)

        # SWA 专属变量
        self.swa_model = None
        self.swa_n = 0
        self.t_start = int(0.75 * self.num_round)  # 论文设定在 75% 轮次开始 SWA [cite: 227]

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):
            # 1. 客户端本地训练 (执行 SAM/ASAM)
            selected_clients = self.select_clients(seed=round_i)
            # 注意：这里调用的是上一步我们在 base.py 里增加的专属函数
            solns, stats = self.local_train_fedsam(round_i, selected_clients)
            self.metrics.extend_commu_stats(round_i, stats)

            # 2. 服务端聚合并更新最新模型 (这代表了不断前行的探索轨迹)
            self.latest_model = self.aggregate(solns)

            # 3. 核心分支：判断测试哪个模型并记录
            if round_i < self.t_start:
                # ====== 阶段一：纯 SAM/ASAM，直接测基础模型 ======
                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
                self.acc_list_test.append(accuracy_test)
                self.loss_list_test.append(loss_test)
            else:
                # ====== 阶段二：启动 SWA ======
                if self.swa_model is None:
                    self.swa_model = self.latest_model.clone()
                    self.swa_n = 1
                else:
                    self.swa_n += 1
                    # SWA 平滑更新
                    self.swa_model = (self.swa_model * (self.swa_n - 1) + self.latest_model) / self.swa_n

                # 临时换上 SWA 模型进行测试
                backup_latest = self.latest_model.clone()
                self.latest_model = self.swa_model

                # 此时测出来的 acc 和 loss 都是 SWA 模型的，直接塞进唯一的列表中
                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
                self.acc_list_test.append(accuracy_test)
                self.loss_list_test.append(loss_test)

                # 测完立刻把基础探索模型换回来，保证下一轮分发给客户端的权重是对的
                self.latest_model = backup_latest

        # 训练结束后，保存唯一的数据集
        np.save(loss_dir + '/loss_test_' + self.dataset + '_' + self.model + '_freeze', self.loss_list_test)
        np.save(acc_dir + '/acc_test_' + self.dataset + '_' + self.model + '_freeze', self.acc_list_test)

        self.metrics.write()

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()
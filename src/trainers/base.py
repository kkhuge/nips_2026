import numpy as np
import os
import torch
import torch.nn as nn
import time
from src.models.client import Client
from src.utils.worker_utils import Metrics
from src.models.worker import Worker
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, ConcatDataset

class BaseTrainer(object):
    def __init__(self, options, dataset, another_dataset, model=None, optimizer=None, name='', worker=None):
        if model is not None and optimizer is not None:
            self.worker = Worker(model, optimizer, options)
        elif worker is not None:
            self.worker = worker
        else:
            raise ValueError("Unable to establish a worker! Check your input parameter!")
        print('>>> Activate a worker for training')
        self.device = options["device"]
        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.all_train_data_num = 0
        _,_,self.all_train_data, self.all_test_data = dataset
        self.clients = self.setup_clients(dataset, another_dataset)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round']
        self.clients_per_round = options['clients_per_round']
        self.eval_every = options['eval_every']
        self.simple_average = not options['noaverage']
        print('>>> Weigh updates by {}'.format(
            'simple average' if self.simple_average else 'sample numbers'))

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']
        self.latest_model = self.worker.get_flat_model_params()

        # combined_features_train = []
        # combined_labels_train = []
        # combined_features_test = []
        # combined_labels_test = []
        #
        # transform = self.all_test_data[0].transform #if cifar, do not clip.
        # # 遍历字典中的每个 MiniDataset
        # for i in self.all_train_data:
        #     dataset = self.all_train_data[i]  # 取出 MiniDataset
        #     features = torch.tensor(dataset.data)  # 提取特征
        #     labels = torch.tensor(dataset.labels)  # 提取标签
        #
        #     combined_features_train.append(features)  # 收集特征
        #     combined_labels_train.append(labels)  # 收集标签
        #
        # # 合并所有特征和标签
        # combined_features_train = torch.cat(combined_features_train, dim=0)  # 按行拼接
        # combined_labels_train = torch.cat(combined_labels_train, dim=0)  # 按行拼接
        # self.all_train_data_client_0 = self.all_train_data[0]
        # self.all_train_data = CustomDataset(data=combined_features_train,labels=combined_labels_train,transform=transform)
        #
        # self.centralized_train_dataloader = DataLoader(self.all_train_data, batch_size=100, shuffle=False)
        #
        #
        # for i in self.all_test_data:
        #     dataset = self.all_test_data[i]  # 取出 MiniDataset
        #     features = torch.tensor(dataset.data)  # 提取特征
        #     labels = torch.tensor(dataset.labels)  # 提取标签
        #
        #     combined_features_test.append(features)  # 收集特征
        #     combined_labels_test.append(labels)  # 收集标签
        #
        # # 合并所有特征和标签
        # combined_features_test = torch.cat(combined_features_test, dim=0)  # 按行拼接
        # combined_labels_test = torch.cat(combined_labels_test, dim=0)  # 按行拼接
        # self.all_test_data = CustomDataset(data=combined_features_test, labels=combined_labels_test,transform=transform)
        # self.centralized_test_dataloader = DataLoader(self.all_test_data, batch_size=64, shuffle=False)

        # ==================== 修改后的代码开始 ====================

        # 1. 处理训练集 (Train Data)
        train_datasets_list = []
        # self.all_train_data 是一个字典: {client_id: dataset}
        for client_id in self.all_train_data:
            dataset = self.all_train_data[client_id]
            train_datasets_list.append(dataset)

        # 使用 ConcatDataset 逻辑合并，不需要提取 data/features
        # 这会自动保留原本 dataset 中的 transform
        self.all_train_data_combined = ConcatDataset(train_datasets_list)

        # 保留对第一个客户端数据的引用（为了兼容原代码某些可能的调用，可选）
        if len(train_datasets_list) > 0:
            self.all_train_data_client_0 = train_datasets_list[0]

        # 兼容原代码逻辑：原代码将 self.all_train_data 覆盖为了合并后的数据集
        self.all_train_data = self.all_train_data_combined

        # 创建集中式训练 Dataloader
        self.centralized_train_dataloader = DataLoader(self.all_train_data,batch_size=64,shuffle=False)

        # 2. 处理测试集 (Test Data)
        test_datasets_list = []
        # self.all_test_data 是一个字典
        for client_id in self.all_test_data:
            dataset = self.all_test_data[client_id]
            test_datasets_list.append(dataset)

        # 逻辑合并测试集
        self.all_test_data_combined = ConcatDataset(test_datasets_list)

        # 覆盖变量以保持兼容
        self.all_test_data = self.all_test_data_combined

        # 创建集中式测试 Dataloader (这是 BooNTK 需要的!)
        self.centralized_test_dataloader = DataLoader(self.all_test_data,batch_size=64,shuffle=False)

        # ==================== 修改结束 ====================




    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def setup_clients(self, dataset,another_dataset):
        """Instantiates clients based on given train and test data directories

        Returns:
            all_clients: List of clients
        """

        users, groups, train_data, test_data = dataset
        another_users, another_groups, another_train_data, another_test_data = another_dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = Client(user_id, group, train_data[user], test_data[user], another_train_data[user], another_test_data[user], self.batch_size, self.worker)
            all_clients.append(c)
        return all_clients

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def select_clients(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()
        # client = []
        # for i in range(10):
        #     client.append(self.clients[i])
        # return client


    def local_train(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)
            # Solve minimization locally
            soln, stat = c.local_train(round_i)
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def local_train_prox(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)
            # Solve minimization locally
            soln, stat = c.local_train_prox(round_i)
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats


    def local_train_scaffold(self, round_i, selected_clients, global_c, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        delta_c_dic = []
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat, delta_c = c.local_train_scaffold(round_i, global_c)
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)
            delta_c_dic.append(delta_c)

        return solns, stats, delta_c_dic

    def local_train_dyn(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train_dyn(round_i)
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def local_train_etf(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train_etf(round_i)
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats



    def local_train_uv(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train_uv(round_i)
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats


    def local_test(self, use_eval_data=True):
        # assert self.latest_model is not None
        self.worker.set_flat_model_params(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for i, c in enumerate(self.clients):
            tot_correct, num_sample, loss = c.local_test(use_eval_data=use_eval_data)

            tot_corrects.append(tot_correct)
            num_samples.append(num_sample)
            losses.append(loss)

        ids = [c.cid for c in self.clients]
        groups = [c.group for c in self.clients]

        stats = {'acc': sum(tot_corrects) / sum(num_samples),
                 'loss': sum(losses) / sum(num_samples),
                 'num_samples': num_samples, 'ids': ids, 'groups': groups}

        return stats


    def test_latest_model_on_traindata(self, round_i):
        # Collect stats from total train data
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        # Record the global gradient
        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        # for c in self.clients:
        #     (num, client_grad), stat = c.solve_grad()
        #     local_grads.append(client_grad)
        #     num_samples.append(num)
        #     global_grads += client_grad * num
        # global_grads /= np.sum(np.asarray(num_samples))
        # stats_from_train_data['gradnorm'] = 0

        # # Measure the gradient difference
        # difference = 0.
        # for idx in range(len(self.clients)):
        #     difference += np.sum(np.square(global_grads - local_grads[idx]))
        # difference /= len(self.clients)
        # stats_from_train_data['graddiff'] = difference
        end_time = time.time()
        #
        # self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('= Train = round: {} / acc: {:.3%} / loss: {:.4f} /'
                  ' time: {:.2f}s'.format(
                   round_i, stats_from_train_data['acc'], stats_from_train_data['loss'],
                    end_time-begin_time))
            print('=' * 102 + "\n")
        return global_grads, stats_from_train_data['acc'], stats_from_train_data['loss']


    def test_latest_model_on_evaldata(self, round_i):
        # Collect stats from total eval data
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result and round_i % self.eval_every == 0:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)
        return stats_from_eval_data['loss'], stats_from_eval_data['acc']





class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(CustomDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, target


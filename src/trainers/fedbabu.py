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


loss_dir = "result_loss/fedbabu"
loss_dir_fedfrth_p ='result_loss/fedavg5'
acc_dir = "result_acc/fedbabu"
acc_dir_fedfrth_p ='result_acc/fedavg5'
if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)



class FedBabuTrainer(BaseTrainer):
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
        backbone_params = []
        for name, p in model.named_parameters():
            if not name.startswith("readout"):
                backbone_params.append(p)
        self.optimizer = SGD(backbone_params, lr=options['lr'], weight_decay=0.0001)
        self.optimizer_last_layer = SGD(model.readout.parameters(), lr=options['lr'], weight_decay=0.0001)
        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']
        worker = LrdWorker(model, self.optimizer, self.optimizer_last_layer, options)
        super(FedBabuTrainer, self).__init__(options, dataset, another_dataset, worker=worker)

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
        # # ===== Save the stage-one model =====
        # torch.save(self.clients[0].worker.model.state_dict(), "FedBabu_stage1_resnet18_cifar100_0.1.pth")

        # ================= Stage 2: Personalized Fine-tuning =================
        print('===================== Start Standard Finetuning with New Models =====================')

        # 1. Load global model weights learned by federated training.
        global_state_dict = torch.load("FedBabu_stage1_resnet18_cifar100_0.1.pth")

        # Paper setting: number of fine-tuning epochs.
        finetune_epochs = 200

        # Initialize matrices for data storage: [client_count, epoch_count].
        # Change: store correct and total counts instead of accuracy so weighted accuracy can be computed later.
        num_clients = len(self.clients)
        client_correct_matrix = np.zeros((num_clients, finetune_epochs))
        client_loss_sum_matrix = np.zeros((num_clients, finetune_epochs))
        client_sample_matrix = np.zeros((num_clients, finetune_epochs))

        # --- Outer loop over clients ---
        for client_idx, client in enumerate(tqdm(self.clients, desc="Finetuning Clients")):

            # Create a fully independent new model for the current client.
            local_model = choose_model(self.options)

            # Move the model to GPU.
            if self.options['gpu']:
                local_model = local_model.cuda()

            # 2. Initialization: load the stage-one trained Body and keep a random Head.
            local_model.load_state_dict(global_state_dict)

            # 3. Set up the fine-tuning optimizer for full-parameter updates.
            optimizer = torch.optim.SGD(local_model.readout.parameters(),
                                        lr=0.1,
                                        weight_decay=0.0001)

            # --- Inner loop: train this client for N consecutive epochs ---
            for epoch in range(finetune_epochs):

                # --- A. Train ---
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

                # --- B. Test after each fine-tuning epoch ---
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
                        # loss.item() is an average; multiply by batch_size to recover the sum.
                        test_loss_sum += loss.item() * target.size(0)
                        test_total += target.size(0)
                print(test_correct/test_total)

                # Store raw values in the matrices instead of ratios.
                client_correct_matrix[client_idx][epoch] = test_correct
                client_loss_sum_matrix[client_idx][epoch] = test_loss_sum
                client_sample_matrix[client_idx][epoch] = test_total

        # ================= Organize and save data =================

        print('\n>>> Finetuning Summary (Weighted Average):')
        for epoch in range(finetune_epochs):
            # Compute total correct count / total sample count across all clients for this epoch.
            total_correct = np.sum(client_correct_matrix[:, epoch])
            total_loss_sum = np.sum(client_loss_sum_matrix[:, epoch])
            total_samples = np.sum(client_sample_matrix[:, epoch])

            # Weighted accuracy.
            weighted_acc = total_correct / total_samples
            # Weighted loss.
            weighted_loss = total_loss_sum / total_samples

            print('Epoch: {}; Acc: {:.4f}; Loss: {:.4f}'.format(epoch, weighted_acc, weighted_loss))

            # Append to lists.
            self.loss_list_test.append(weighted_loss)
            self.acc_list_test.append(weighted_acc)

        # Save.
        np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_finetuning', self.loss_list_test)
        np.save(acc_dir + '/acc_test' + self.dataset + self.model+ '_finetuning', self.acc_list_test)

        # np.save(loss_dir_fedfrth_p + '/loss_test' + self.dataset + self.model, self.loss_list_test)
        # np.save(acc_dir_fedfrth_p + '/acc_test' + self.dataset + self.model, self.acc_list_test)

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()



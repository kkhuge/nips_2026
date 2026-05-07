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

error_dir = 'result_error/boontk'
weight_change_dir = "result_weight_change/boontk"
theta_dir = "result_theta/boontk"
output_dir = "result_output_differ/boontk"
loss_dir = "result_loss/boontk"
acc_dir = "result_acc/boontk"

if not os.path.exists(error_dir):
    os.makedirs(error_dir)
if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
if not os.path.exists(theta_dir):
    os.makedirs(theta_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(weight_change_dir):
    os.makedirs(weight_change_dir)


class BooNTKTrainer(BaseTrainer):
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
        super(BooNTKTrainer, self).__init__(options, dataset, another_dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        for round_i in range(self.num_round):
            if round_i<=450:
                # Test latest model on train data
                # _, accuracy, loss = self.test_latest_model_on_traindata(round_i)
                # self.acc_list_train.append(accuracy)
                # self.loss_list_train.append(loss)
                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
                self.acc_list_test.append(accuracy_test)
                self.loss_list_test.append(loss_test)
        
                # Choose K clients prop to data size
                selected_clients = self.select_clients(seed=round_i)
        
                # Solve minimization locally
                solns, stats = self.local_train(round_i, selected_clients)
        
                # Track communication cost
                self.metrics.extend_commu_stats(round_i, stats)
        
                # Update latest model
                self.latest_model = self.aggregate(solns)
        
                self.worker.set_flat_model_params(self.latest_model)
        
        
        # Test final model on train data
        # _, accuracy, loss = self.test_latest_model_on_traindata(self.num_round)
        # self.acc_list_train.append(accuracy)
        # self.loss_list_train.append(loss)
        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        self.acc_list_test.append(accuracy_test)
        self.loss_list_test.append(loss_test)
        torch.save(self.clients[0].worker.model.state_dict(), "stage1_resnet18_tinyimagenet_0.1.pth")

        print('===================== Start TCT Stage-2 =====================')
        model_path = "stage1_resnet18_tinyimagenet_0.1.pth"
        self.worker.model.load_state_dict(torch.load(model_path))
        stage_1_model = self.worker.model
        stage_1_model.readout =  nn.Linear(512, 200).cuda()
        grad_all_train = []
        traget_all_train = []
        grad_all_test = []
        traget_all_test = []

        for i in range(len(self.clients)):
            # Train NTK
            grad_i_train, target_i_train = self.client_compute_eNTK(stage_1_model, self.clients[i].train_dataloader)


            grad_i_train_cpu = [g.cpu() for g in grad_i_train]
            target_i_train_cpu = [t.cpu() for t in target_i_train]

            grad_all_train.append(grad_i_train_cpu)
            traget_all_train.append(target_i_train_cpu)

            del grad_i_train, target_i_train
            torch.cuda.empty_cache()

            # Test NTK
            grad_i_test, target_i_test = self.client_compute_eNTK(stage_1_model, self.clients[i].test_dataloader)

            grad_i_test_cpu = [g.cpu() for g in grad_i_test]
            target_i_test_cpu = [t.cpu() for t in target_i_test]

            grad_all_test.append(grad_i_test_cpu)
            traget_all_test.append(target_i_test_cpu)

            del grad_i_test, target_i_test
            torch.cuda.empty_cache()

        ## test model
        # grad_train_eval, target_train_eval = self.client_compute_eNTK_all(stage_1_model, self.centralized_train_dataloader)
        grad_test_eval, target_test_eval = self.client_compute_eNTK_all(stage_1_model, self.centralized_test_dataloader)
        # Init linear models
        theta_global = torch.zeros(512, 200).cuda()
        theta_global = torch.tensor(theta_global, requires_grad=False)
        client_thetas = [torch.zeros_like(theta_global).cuda() for _ in range(len(self.clients))]
        client_hi_s = [torch.zeros_like(theta_global).cuda() for _ in range(len(self.clients))]

        # Run TCT-Stage2
        for round_i in range(450,550):
            theta_list = []
            selected_clients = self.select_clients(seed=round_i)
            for i in range(len(selected_clients)):
                grad_i = grad_all_train[selected_clients[i].cid]
                target_i = traget_all_train[selected_clients[i].cid]
                theta_hat_update, h_i_client_update = self.scaffold_update(grad_i,
                                                                      target_i,
                                                                      client_thetas[selected_clients[i].cid],
                                                                      client_hi_s[selected_clients[i].cid],
                                                                      theta_global,
                                                                      M=1,
                                                                      lr_local=0.1)
                client_hi_s[selected_clients[i].cid] = h_i_client_update * 1.0
                client_thetas[selected_clients[i].cid] = theta_hat_update * 1.0
                theta_list.append(theta_hat_update)

            theta_global = torch.zeros_like(theta_list[0]).cuda()
            all_num = 0
            for x in range(len(selected_clients)):
                all_num += len(selected_clients[x].train_data.labels)
            for theta_idx in range(len(selected_clients)):
                theta_global += ( len(selected_clients[theta_idx].train_data.labels) / all_num) * theta_list[theta_idx]

            # # ===== Train Evaluation =====
            # logits_class_train = grad_train @ theta_global
            # targets_train = target_train.cuda().long()
            #
            # # accuracy
            # _, targets_pred_train = logits_class_train.max(1)
            # train_acc = targets_pred_train.eq(targets_train).sum() / logits_class_train.shape[0]
            #
            # # CE loss
            # train_loss = F.cross_entropy(logits_class_train, targets_train)

            # ===== Test Evaluation =====
            logits_class_test = grad_test_eval @ theta_global
            targets_test = target_test_eval.cuda().long()

            _, targets_pred_test = logits_class_test.max(1)
            test_acc = targets_pred_test.eq(targets_test).sum() / logits_class_test.shape[0]

            test_loss = F.cross_entropy(logits_class_test, targets_test)

            self.acc_list_test.append(test_acc.item())
            self.loss_list_test.append(test_loss.item())

            # ===== Print =====
            print('Round %d: train_loss=%.4f train_acc=%.4f | test_loss=%.4f test_acc=%.4f'
                  % (round_i, 0, 0, test_loss.item(), test_acc.item()))

        # np.save(loss_dir + '/loss_train' + self.dataset + self.model + '_freeze', self.loss_list_train)  #
        # np.save(acc_dir + '/acc_train' + self.dataset + self.model+ '_freeze', self.acc_list_train)

        np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_1_communication', self.loss_list_test)  #
        np.save(acc_dir + '/acc_test' + self.dataset + self.model + '_1_communication', self.acc_list_test)


        # Save tracked information
        self.metrics.write()

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()

    def client_compute_eNTK(self,client_model, train_loader):
        """Train a client_model on the train_loder data."""
        grads_data_list = []
        target_list = []
        for data, targets in train_loader:
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data)
            data = data.cuda()

            grads_data = self.compute_eNTK(client_model, data.cuda())
            grads_data = grads_data.float().cuda()
            grads_data_list.append(grads_data)

            if not isinstance(targets, torch.Tensor):
                targets = torch.as_tensor(targets, dtype=torch.long)
            targets = targets.cuda()
            target_list.append(targets)

        return grads_data_list, target_list 

    def client_compute_eNTK_all(self, client_model, train_loader):
        """
        Compute eNTK features for ALL batches in the train_loader.
        Returns:
            grads_all:  [N_total, P_sub]   (stacked eNTK features)
            targets_all: [N_total]         (all labels)
        """

        client_model.eval()
        grads_list = []
        targets_list = []

        for data, targets in train_loader:

            # ensure tensor & cuda
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data)
            data = data.cuda()

            if not isinstance(targets, torch.Tensor):
                targets = torch.as_tensor(targets, dtype=torch.long)
            targets = targets.cuda()

            # compute eNTK for this batch
            grads_batch = self.compute_eNTK(client_model, data)
            grads_batch = grads_batch.float().cuda()

            grads_list.append(grads_batch)
            targets_list.append(targets)

            torch.cuda.empty_cache()

        # concatenate all batches
        grads_all = torch.cat(grads_list, dim=0)
        targets_all = torch.cat(targets_list, dim=0)

        return grads_all, targets_all

    def compute_eNTK(self, model, X, subsample_size=512, seed=123):
        """"compute eNTK"""
        model.eval()
        params = list(model.parameters())
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        param_dim = sum(p.numel() for p in params if p.requires_grad)
        subsample_size = min(subsample_size, param_dim)

        random_index = torch.randperm(param_dim)[:subsample_size]

        grads = None

        for i in tqdm(range(X.size()[0])):

            model.zero_grad()
            inp = X[i: i + 1]

            # ===== Only flatten MLP =====
            first_param = list(model.parameters())[0]
            if first_param.dim() == 2:  # Linear layer
                inp = inp.reshape(inp.size(0), -1)

            # ====== forward ======
            out = model(inp)

            scalar = out.sum()
            scalar.backward()

            # collect grad
            grad = []
            for p in params:
                if p.requires_grad:
                    grad.append(p.grad.flatten())

            grad = torch.cat(grad)
            grad = grad[random_index]

            if grads is None:
                grads = torch.zeros((X.size()[0], grad.size()[0]), dtype=torch.half).cuda()
            grads[i] = grad

        return grads

    def scaffold_update(self, grads_data_list, targets_list, theta_client, h_i_client_pre,
                        theta_global, M=1, lr_local=0.01):


        # ===== SCAFFOLD correction term =====
        h_i_client_update = h_i_client_pre + (1 / (M * lr_local*len(targets_list))) * (theta_global - theta_client)

        # initialize local theta
        theta_hat_local = theta_global.clone()

        for _ in range(M):
            for grads_data,targets in zip(grads_data_list, targets_list):
                grads_data = grads_data.float().cuda()  # G: (N, D)
                targets = targets.cuda().long()  # labels: (N,)
                num_samples = targets.shape[0]
                # ---- 1. logits = G @ θ ----
                logits = grads_data @ theta_hat_local  # (N, C)

                # ---- 2. CE gradient: ∂L/∂logits = softmax(logits) − y_onehot ----
                probs = torch.softmax(logits, dim=1)
                target_onehot = F.one_hot(targets, num_classes=probs.size(1)).float()

                grad_logits = (probs - target_onehot) / num_samples

                # ---- 3. gradient wrt θ: G^T (probs − y) ----
                grad_theta = grads_data.t() @ grad_logits  # (D, C)

                # ---- 4. SCAFFOLD local update ----
                theta_hat_local -= lr_local * (grad_theta )

        return theta_hat_local, h_i_client_update



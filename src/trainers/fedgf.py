from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import FedGFWorker
from torch.optim import SGD
import numpy as np
import torch
import os

criterion = torch.nn.CrossEntropyLoss()

loss_dir = "result_loss/fedgf"
acc_dir = "result_acc/fedgf"

if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)


class FedGFTrainer(BaseTrainer):
    def __init__(self, options, dataset, another_dataset):
        # ======= FedGF-specific hyperparameters are hard-coded here =======
        self.rho = 0.02 #0.02
        self.T_D = 4 #0.3，3
        self.W = 10  #10
        # ==========================================

        self.acc_list_train = []
        self.loss_list_train = []
        self.acc_list_test = []
        self.loss_list_test = []

        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0001)
        self.optimizer_last_layer = SGD(model.readout.parameters(), lr=options['lr'], weight_decay=0.0001)

        self.num_round = options['num_round']
        self.dataset = options["dataset"]
        self.model_name = options['model']

        worker = FedGFWorker(model, self.optimizer, self.optimizer_last_layer, options)
        super(FedGFTrainer, self).__init__(options, dataset, another_dataset, worker=worker)

        self.I_history = []
        self.previous_global_model = None

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        self.latest_model = self.worker.get_flat_model_params().detach()
        self.previous_global_model = self.latest_model.clone()

        for round_i in range(self.num_round):
            if round_i >= 450:
                _, accuracy, loss = self.test_latest_model_on_traindata(round_i)
                self.acc_list_train.append(accuracy)
                self.loss_list_train.append(loss)
                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
                self.acc_list_test.append(accuracy_test)
                self.loss_list_test.append(loss_test)
            # else:
            #     loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)

            selected_clients = self.select_clients(seed=round_i)

            # --- Stage check: in stage two (>=950), aggregate directly without flatness-metric interpolation ---
            if round_i >= 550:
                c = 0.0
                global_perturbed_model = None
                solns, stats, _ = self.local_train_gf(round_i, selected_clients, global_perturbed_model, c)
                self.metrics.extend_commu_stats(round_i, stats)

                # Simple model aggregation; stage two updates only the head.
                self.latest_model = self.aggregate(solns)
                self.worker.set_flat_model_params(self.latest_model)
                self.previous_global_model = self.latest_model.clone()
                continue

            # --- Stage one (< 950): FedGF control flow, compute interpolation coefficient c ---
            if len(self.I_history) == 0:
                c = 0.0
            else:
                window_vals = self.I_history[-self.W:]
                c = sum(window_vals) / len(window_vals)

            # c = 0.0

            # \Delta^r = w^{r-1} - w^r
            delta_r = self.previous_global_model - self.latest_model
            delta_norm = torch.norm(delta_r, p=2)

            # ====== Core fix: compute only the global perturbation vector, not absolute parameter coordinates ======
            if delta_norm > 0:
                global_pert_vec = self.rho * (delta_r / delta_norm)
            else:
                global_pert_vec = torch.zeros_like(delta_r)

            # Pass global_pert_vec to clients; despite the global_perturbed_model name, the actual value is a vector.
            solns, stats, divergences = self.local_train_gf(round_i, selected_clients, global_pert_vec, c)
            self.metrics.extend_commu_stats(round_i, stats)

            # Aggregate the global model and record the previous round.
            self.previous_global_model = self.latest_model.clone()
            self.latest_model = self.aggregate(solns)
            self.worker.set_flat_model_params(self.latest_model)

            # Compute divergence D_r_plus_1 and store it in the sliding window.
            D_r_plus_1 = sum(divergences) / len(divergences)

            # # Debug change 2: print the true divergence to inspect its magnitude.
            # original_c = sum(self.I_history[-self.W:]) / len(self.I_history[-self.W:]) if len(
            #     self.I_history) > 0 else 0.0
            # print(
            #     f">>> [DEBUG] Round {round_i}: D = {D_r_plus_1:.4f} | T_D = {self.T_D} | Original c would be: {original_c:.2f}")

            I_val = 1.0 if D_r_plus_1 > self.T_D else 0.0
            self.I_history.append(I_val)

        # Training finished; test and save.
        _, accuracy, loss = self.test_latest_model_on_traindata(self.num_round)
        self.acc_list_train.append(accuracy)
        self.loss_list_train.append(loss)
        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        self.acc_list_test.append(accuracy_test)
        self.loss_list_test.append(loss_test)

        np.save(os.path.join(loss_dir, 'loss_train' + self.dataset + self.model_name ), self.loss_list_train)
        np.save(os.path.join(acc_dir, 'acc_train' + self.dataset + self.model_name ), self.acc_list_train)
        np.save(os.path.join(loss_dir, 'loss_test' + self.dataset + self.model_name ), self.loss_list_test)
        np.save(os.path.join(acc_dir, 'acc_test' + self.dataset + self.model_name ), self.acc_list_test)


        # np.save(os.path.join(loss_dir, 'loss_train' + self.dataset + self.model_name + '_freeze'), self.loss_list_train)
        # np.save(os.path.join(acc_dir, 'acc_train' + self.dataset + self.model_name + '_freeze'), self.acc_list_train)
        # np.save(os.path.join(loss_dir, 'loss_test' + self.dataset + self.model_name + '_freeze'), self.loss_list_test)
        # np.save(os.path.join(acc_dir, 'acc_test' + self.dataset + self.model_name + '_freeze'), self.acc_list_test)

        self.metrics.write()

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()

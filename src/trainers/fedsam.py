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

        # SWA-specific variables.
        self.swa_model = None
        self.swa_n = 0
        self.t_start = int(0.75 * self.num_round)  # The paper starts SWA at 75% of rounds [cite: 227].

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):
            # 1. Client local training with SAM/ASAM.
            selected_clients = self.select_clients(seed=round_i)
            # Note: this calls the dedicated function added in base.py in the previous step.
            solns, stats = self.local_train_fedsam(round_i, selected_clients)
            self.metrics.extend_commu_stats(round_i, stats)

            # 2. Server aggregation updates the latest model, representing the ongoing exploration trajectory.
            self.latest_model = self.aggregate(solns)

            # 3. Core branch: decide which model to test and record.
            if round_i < self.t_start:
                # ====== Stage one: pure SAM/ASAM, directly test the base model ======
                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
                self.acc_list_test.append(accuracy_test)
                self.loss_list_test.append(loss_test)
            else:
                # ====== Stage two: start SWA ======
                if self.swa_model is None:
                    self.swa_model = self.latest_model.clone()
                    self.swa_n = 1
                else:
                    self.swa_n += 1
                    # SWA smoothing update.
                    self.swa_model = (self.swa_model * (self.swa_n - 1) + self.latest_model) / self.swa_n

                # Temporarily switch to the SWA model for testing.
                backup_latest = self.latest_model.clone()
                self.latest_model = self.swa_model

                # The measured accuracy and loss belong to the SWA model; append them directly to the single list.
                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
                self.acc_list_test.append(accuracy_test)
                self.loss_list_test.append(loss_test)

                # Immediately restore the base exploration model after testing so the next round sends correct weights.
                self.latest_model = backup_latest

        # After training, save the single dataset.
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

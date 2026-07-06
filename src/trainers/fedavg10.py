from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import MSEWorker
from torch.optim import SGD
import numpy as np
import torch
import os



criterion = torch.nn.CrossEntropyLoss()

loss_dir = "result_loss/fedavg10"
acc_dir = "result_acc/fedavg10"


if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)



class FedAvg10Trainer(BaseTrainer):
    """
    Original Scheme
    """
    def __init__(self, options, dataset):
        self.error_train = []
        self.loss_list_train = []
        self.acc_list_train = []
        self.loss_list_test = []
        self.acc_list_test = []
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.required_accuracy = options['psi']
        self.tau = options['num_epoch']
        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0005)
        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']
        worker = MSEWorker(model, self.optimizer,options)
        super(FedAvg10Trainer, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        self.last_centralized_model = self.latest_model.clone()
        # _, loss, accuracy = self.test_latest_model_on_traindata_MSE(0)
        for round_i in range(self.num_round):


            _, accuracy, loss = self.test_latest_model_on_traindata(round_i)
            self.loss_list_train.append(loss)
            self.acc_list_train.append(accuracy)



            loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
            self.loss_list_test.append(loss_test)
            self.acc_list_test.append(accuracy_test)


            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns)







        # Test final model on train data
        _, accuracy, loss = self.test_latest_model_on_traindata(self.num_round)
        self.loss_list_train.append(loss)
        self.acc_list_train.append(accuracy)

        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        self.loss_list_test.append(loss_test)
        self.acc_list_test.append(accuracy_test)



        np.save(loss_dir + '/loss_train' + self.dataset + self.model + '_4', self.loss_list_train)
        np.save(acc_dir + '/acc_train' + self.dataset + self.model + '_4', self.acc_list_train)
        np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_4', self.loss_list_test)
        np.save(acc_dir + '/acc_test' + self.dataset + self.model + '_4', self.acc_list_test)


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




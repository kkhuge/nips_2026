from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker, ScaffoldWorker
from src.optimizers.gd import GD
from torch.optim import SGD
import numpy as np
import torch
import os


criterion = torch.nn.CrossEntropyLoss()

error_dir = 'result_error/scaffold'
weight_change_dir = "result_weight_change/scaffold"
theta_dir = "result_theta/scaffold"
output_dir = "result_output_differ/scaffold"
loss_dir = "result_loss/scaffold"
acc_dir = "result_acc/scaffold"

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


class ScaffoldTrainer(BaseTrainer):
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
        model_path = "scaffold_stage1_resnet18_tinyimagenet_0.5.pth"
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.move_model_to_gpu(model, options)
        self.global_c = [torch.zeros_like(param.data).to(param.device) for param in model.parameters()]
        self.required_accuracy = options['psi']
        self.tau = options['num_epoch']
        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0001)
        self.optimizer_last_layer = SGD(model.readout.parameters(), lr=0.1, weight_decay=0.0001)
        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']
        worker = ScaffoldWorker(model, self.optimizer, self.optimizer_last_layer,options)
        super(ScaffoldTrainer, self).__init__(options, dataset, another_dataset, worker=worker)


    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        for round_i in range(0,100):
            # if round_i > 450:
            # Test latest model on train data
            _, accuracy, loss = self.test_latest_model_on_traindata(round_i)
            self.acc_list_train.append(accuracy)
            self.loss_list_train.append(loss)
            loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
            self.acc_list_test.append(accuracy_test)
            self.loss_list_test.append(loss_test)
            # if round_i == 450:
            # #     # torch.save(self.worker.model.state_dict(), "table1_fedprox_stage1_resnet18_cifar10_2_class.pth")
            #     torch.save(self.worker.model.state_dict(), "scaffold_stage1_resnet18_tinyimagenet_0.5.pth")
            # #     torch.save(self.worker.model.state_dict(), "scaffold_stage1_resnet18_cifar100_0.5.pth")

            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)

            # Solve minimization locally
            solns, stats, delta_c = self.local_train_scaffold(round_i, selected_clients, self.global_c)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model, self.global_c = self.aggregate(solns, delta_c)

            self.worker.set_flat_model_params(self.latest_model)

        # Test final model on train data
        _, accuracy, loss = self.test_latest_model_on_traindata(self.num_round)
        self.acc_list_train.append(accuracy)
        self.loss_list_train.append(loss)
        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        self.acc_list_test.append(accuracy_test)
        self.loss_list_test.append(loss_test)

        np.save(loss_dir + '/loss_train' + self.dataset + self.model, self.loss_list_train)
        np.save(acc_dir + '/acc_train' + self.dataset + self.model , self.acc_list_train)
        np.save(loss_dir + '/loss_test' + self.dataset + self.model , self.loss_list_test)
        np.save(acc_dir + '/acc_test' + self.dataset + self.model , self.acc_list_test)

        # np.save(loss_dir + '/loss_train' + self.dataset + self.model + '_freeze', self.loss_list_train)
        # np.save(acc_dir + '/acc_train' + self.dataset + self.model + '_freeze', self.acc_list_train)
        # np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_freeze', self.loss_list_test)
        # np.save(acc_dir + '/acc_test' + self.dataset + self.model + '_freeze', self.acc_list_test)

        # Save tracked information
        self.metrics.write()

    def aggregate(self, solns, delta_c):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num

        avg_delta_c = []
        for params in zip(*delta_c):  
            stacked = torch.stack(params)
            avg_delta_c.append(stacked.mean(dim=0))

        self.global_c = [gc + self.clients_per_round / len(self.clients) * avg for gc, avg in zip(self.global_c, avg_delta_c)]

        return averaged_solution.detach(), self.global_c



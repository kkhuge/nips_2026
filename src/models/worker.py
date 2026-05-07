import numpy as np
import torch.nn as nn
from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch
import copy
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()


class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """
    def __init__(self, model, optimizer, optimizer_last_layer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.optimizer_last_layer = optimizer_last_layer
        self.batch_size = options['batch_size']
        self.num_epoch = options['num_epoch']
        self.gpu = options['gpu'] if 'gpu' in options else False
        if options["model"] == '2nn' or options["model"] == 'linear' or options["model"] == "linear_regression" or options["model"] == "2nnc":
            self.flat_data = True
        else:
            self.flat_data = False

        # # Setup local model and evaluate its statics
        # self.flops, self.params_num, self.model_bytes = \
        #     get_model_complexity_info(self.model, options['input_shape'], gpu=options['gpu'])
        self.flops = 1
        self.params_num = 1
        self.model_bytes = 1

    @property
    def model_bits(self):
        return self.model_bytes * 8
    
    def flatten_data(self, x):
        if self.flat_data:
            current_batch_size = x.shape[0]
            return x.reshape(current_batch_size, -1)
        else:
            return x

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)



    # def local_train(self, train_dataloader, another_train_dataloader, round_i, global_c, **kwargs):
    #     """Train model locally and return new parameter and computation cost
    #
    #     Args:
    #         train_dataloader: DataLoader class in Pytorch
    #
    #     Returns
    #         1. local_solution: updated new parameter
    #         2. stat: Dict, contain stats
    #             2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
    #             2.2 loss
    #     """
    #     self.model.train()
    #     train_loss = train_acc = train_total = 0
    #     for epoch in range(self.num_epoch):
    #         train_loss = train_acc = train_total = 0
    #         for batch_idx, (x, y) in enumerate(train_dataloader):
    #             # from IPython import embed
    #             # embed()
    #             x = self.flatten_data(x)
    #             if self.gpu:
    #                 x, y = x.cuda(), y.cuda()
    #
    #             self.optimizer.zero_grad()
    #             pred = self.model(x)
    #
    #             # if torch.isnan(pred.max()):
    #             #     from IPython import embed
    #             #     embed()
    #
    #             loss = criterion(pred, y)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
    #             self.optimizer.step()
    #
    #             _, predicted = torch.max(pred, 1)
    #             correct = predicted.eq(y).sum().item()
    #             target_size = y.size(0)
    #
    #             train_loss += loss.item() * y.size(0)
    #             train_acc += correct
    #             train_total += target_size
    #
    #     local_solution = self.get_flat_model_params()
    #     param_dict = {"norm": torch.norm(local_solution).item(),
    #                   "max": local_solution.max().item(),
    #                   "min": local_solution.min().item()}
    #     comp = self.num_epoch * train_total * self.flops
    #     return_dict = {"comp": comp,
    #                    "loss": train_loss/train_total,
    #                    "acc": train_acc/train_total}
    #     return_dict.update(param_dict)
    #     return local_solution, return_dict

    def local_test(self, test_dataloader, another_test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class LrdWorker(Worker):
    def __init__(self, model, optimizer, optimizer_last_layer,options):
        self.num_epoch = options['num_epoch']
        super(LrdWorker, self).__init__(model, optimizer, optimizer_last_layer, options)
    
    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0

        if round_i < 450:  #450, 390
            for i in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    self.optimizer.zero_grad()
                    pred = self.model(x)

                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # lr = 100/(400+current_step+i)
                    self.optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()
                    target_size = y.size(0)

                    train_loss += loss.item() * y.size(0)
                    train_acc += correct
                    train_total += target_size

            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                "max": local_solution.max().item(),
                "min": local_solution.min().item()}
            comp = self.num_epoch * train_total * self.flops
            return_dict = {"comp": comp,
                "loss": train_loss/train_total,
                    "acc": train_acc/train_total}
            return_dict.update(param_dict)
        else:
            for name, param in self.model.named_parameters():
                if "readout" not in name:  # 只让最后一层 readout 训练
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # self.model.eval()

            for i in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    self.optimizer_last_layer.zero_grad()
                    pred = self.model(x)

                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # lr = 100/(400+current_step+i)
                    self.optimizer_last_layer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()
                    target_size = y.size(0)

                    train_loss += loss.item() * y.size(0)
                    train_acc += correct
                    train_total += target_size


            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                          "max": local_solution.max().item(),
                          "min": local_solution.min().item()}
            comp = self.num_epoch * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total}
            return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader,another_test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def get_grad(self, dataloader):
        all_x = []
        all_y = []
        self.optimizer.zero_grad()
        loss = 0
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        pred = self.model(all_x)
        loss = criterion(pred, all_y)
        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def get_jacobian(self, dataloader):
        self.optimizer.zero_grad()
        out_grad = []
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x).squeeze()
            for i in range(len(pred)):
                one_element_grad = []
                for j in range(len(pred[i])):
                    one_out_grad_flat = get_flat_grad(pred[i][j], self.model.parameters(), create_graph=True)
                    one_element_grad.append(one_out_grad_flat)
                one_element_grad = torch.hstack(one_element_grad)
                out_grad.append(one_element_grad)
        out_grad = torch.vstack(out_grad)

        return out_grad

    def get_error(self, test_dataloader):
        error = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                true_value = np.zeros((len(y), 10))
                true_value[np.arange(len(y)), y] = 1

                pred = self.model(x)
                error = error + np.linalg.norm(pred - true_value,ord='fro')
                print(pred)
                print(true_value)
        return error

class ProxWorker(Worker):
    def __init__(self, model, optimizer, optimizer_last_layer,options):
        self.num_epoch = options['num_epoch']
        super(ProxWorker, self).__init__(model, optimizer,optimizer_last_layer, options)

    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):
        mu = 0.01
        self.model.train()
        train_loss = train_acc = train_total = 0
        global_model_params = [p.clone().detach() for p in self.model.parameters()]

        if round_i < 450:
            for i in range(self.num_epoch):
                for x, y in train_dataloader:
                    prox_term = 0.0
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    for param, global_param in zip(self.model.parameters(), global_model_params):
                        prox_term += (mu / 2) * ((param - global_param) ** 2).sum()
                    loss = criterion(pred, y) + prox_term
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # lr = 100/(400+current_step+i)
                    self.optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()
                    target_size = y.size(0)

                    train_loss += loss.item() * y.size(0)
                    train_acc += correct
                    train_total += target_size

            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                          "max": local_solution.max().item(),
                          "min": local_solution.min().item()}
            comp = self.num_epoch * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total}
            return_dict.update(param_dict)
        else:
            for name, param in self.model.named_parameters():
                if "readout" not in name:  # 只让最后一层 readout 训练
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # self.model.eval()
            for i in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    self.optimizer_last_layer.zero_grad()
                    pred = self.model(x)

                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # lr = 100/(400+current_step+i)
                    self.optimizer_last_layer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()
                    target_size = y.size(0)

                    train_loss += loss.item() * y.size(0)
                    train_acc += correct
                    train_total += target_size

            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                          "max": local_solution.max().item(),
                          "min": local_solution.min().item()}
            comp = self.num_epoch * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total}
            return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader, another_test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss




class ScaffoldWorker(Worker):
    def __init__(self, model, optimizer,optimizer_last_layer, options):
        self.num_epoch = options['num_epoch']
        super(ScaffoldWorker, self).__init__(model, optimizer, optimizer_last_layer, options)

    def local_train(self, train_dataloader, another_train_dataloader, round_i, global_c, local_c, **kwargs):
        self.model.train()
        global_model_parameters = self.get_flat_model_params()
        train_loss = train_acc = train_total = 0
        lr = self.optimizer.param_groups[0]['lr']
        if round_i < 450:
            for i in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    self.optimizer.zero_grad()
                    pred = self.model(x)

                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    with torch.no_grad():
                        for p, gc, lc in zip(self.model.parameters(), global_c, local_c):
                            if p.grad is not None:
                                p.data -= lr * (p.grad - lc + gc)

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()
                    target_size = y.size(0)

                    train_loss += loss.item() * y.size(0)
                    train_acc += correct
                    train_total += target_size

            local_solution = self.get_flat_model_params()
            delta_w = global_model_parameters - local_solution
            aaa = [torch.zeros_like(p) for p in self.model.parameters()]
            vector_to_parameters(delta_w, aaa)
            delta_w = aaa
            local_c_new = []
            delta_c = []
            for lc, gc, dw in zip(local_c, global_c, delta_w):
                ci_new = lc - gc + (1.0 / (self.num_epoch * lr * len(train_dataloader))) * dw
                local_c_new.append(ci_new)
                delta_c.append(-gc + (1.0 / (self.num_epoch * lr* len(train_dataloader))) * dw)
            param_dict = {"norm": torch.norm(local_solution).item(),
                          "max": local_solution.max().item(),
                          "min": local_solution.min().item()}
            comp = self.num_epoch * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total}
            return_dict.update(param_dict)
        else:
            delta_c = [torch.zeros_like(gc) for gc in global_c]
            local_c_new = [torch.zeros_like(gc) for gc in global_c]
            for name, param in self.model.named_parameters():
                if "readout" not in name: 
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # self.model.eval()
            for i in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    self.optimizer_last_layer.zero_grad()
                    pred = self.model(x)

                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                    self.optimizer_last_layer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()
                    target_size = y.size(0)

                    train_loss += loss.item() * y.size(0)
                    train_acc += correct
                    train_total += target_size

            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                          "max": local_solution.max().item(),
                          "min": local_solution.min().item()}
            comp = self.num_epoch * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total}
            return_dict.update(param_dict)
        return local_solution, return_dict, delta_c, local_c_new

    def local_test(self, test_dataloader, another_test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class DynWorker(Worker):
    def __init__(self, model, optimizer, optimizer_last_layer,options):
        self.num_epoch = options['num_epoch']
        super(DynWorker, self).__init__(model, optimizer, optimizer_last_layer,options)

    def local_train(self, train_dataloader, another_train_dataloader, round_i, local_alpha, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0
        global_params = self.get_flat_model_params()
        mu = 0.01
        if round_i < 650:
            for i in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    self.optimizer.zero_grad()
                    pred = self.model(x)

                    loss = criterion(pred, y)

    
                    local_params = self.get_flat_model_params()
                    prox_term = mu / 2 * torch.sum((local_params - global_params) ** 2)
                    dyn_term = torch.dot(local_alpha, local_params)
                    loss = loss + prox_term - dyn_term

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    self.optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()
                    target_size = y.size(0)

                    train_loss += loss.item() * y.size(0)
                    train_acc += correct
                    train_total += target_size
            new_local_params = self.get_flat_model_params()
            local_alpha -= mu * (new_local_params - global_params)

            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                          "max": local_solution.max().item(),
                          "min": local_solution.min().item()}
            comp = self.num_epoch * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total}
            return_dict.update(param_dict)
        else:
            for name, param in self.model.named_parameters():
                if "readout" not in name:  
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            for i in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    self.optimizer_last_layer.zero_grad()
                    pred = self.model(x)

                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # lr = 100/(400+current_step+i)
                    self.optimizer_last_layer.step()

                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(y).sum().item()
                    target_size = y.size(0)

                    train_loss += loss.item() * y.size(0)
                    train_acc += correct
                    train_total += target_size

            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                          "max": local_solution.max().item(),
                          "min": local_solution.min().item()}
            comp = self.num_epoch * train_total * self.flops
            return_dict = {"comp": comp,
                           "loss": train_loss / train_total,
                           "acc": train_acc / train_total}
            return_dict.update(param_dict)
        return local_solution, return_dict, local_alpha

    def local_test(self, test_dataloader, another_test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                # embed()
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class ETFWorker(Worker):
    def __init__(self, model, optimizer, optimizer_last_layer, options):
        self.num_epoch = options['num_epoch']
        super(ETFWorker, self).__init__(model, optimizer, optimizer_last_layer, options)

    def get_label_counts(self, dataloader):
        targets = []
        if hasattr(dataloader.dataset, 'labels'): 
            targets = dataloader.dataset.labels
        elif hasattr(dataloader.dataset, 'tensors'):  # TensorDataset
            targets = dataloader.dataset.tensors[1].numpy()
        else:
            for _, y in dataloader:
                targets.append(y.numpy())
            targets = np.concatenate(targets)

        counts = np.bincount(targets, minlength=self.model.proto_classifier.num_classes)
        return torch.tensor(counts).float()

    def balanced_feature_loss(self, logits, targets, label_counts):
        total_count = label_counts.sum()
        frequencies = label_counts / total_count
        frequencies = frequencies + 1e-9
        gamma = 1.0

        # log(pi^gamma) = gamma * log(pi)
        adjustment = gamma * torch.log(frequencies).to(logits.device)
        adjusted_logits = logits + adjustment

        return F.cross_entropy(adjusted_logits, targets)

    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0
        for name, param in self.model.named_parameters():
            if "proto_classifier" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        label_counts = self.get_label_counts(train_dataloader)
        if self.gpu:
            label_counts = label_counts.cuda()

        for epoch in range(self.num_epoch):
            for x, y in train_dataloader:
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()

                logits_etf, _ = self.model(x)
                loss = self.balanced_feature_loss(logits_etf, y, label_counts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(logits_etf, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * target_size
                train_acc += correct
                train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {
            "norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()
        }

        comp = self.num_epoch * train_total * self.flops
        return_dict = {
            "comp": comp,
            "loss": train_loss / train_total,
            "acc": train_acc / train_total
        }
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader, another_test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0

        with torch.no_grad():
            for x, y in test_dataloader:
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                logits, _ = self.model(x)


                loss = criterion(logits, y) 
                _, predicted = torch.max(logits, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class FedUVWorker(Worker):
    def __init__(self, model, optimizer, optimizer_last_layer, options):
        self.num_epoch = options['num_epoch']
        super(FedUVWorker, self).__init__(model, optimizer, optimizer_last_layer, options)
        self.num_classes = self.model.readout.out_features
        self.mu_feduv = 0.1 #0.1
        self.lambda_feduv = self.num_classes / 100 
        self._cached_features = None

        if hasattr(self.model, "readout"):
            self.model.readout.register_forward_hook(self._save_features_hook)
        else:
            raise ValueError("Error")

    def _save_features_hook(self, module, inputs, output):
        self._cached_features = inputs[0]

    def _classifier_variance_loss(self, logits):
        probs = torch.softmax(logits, dim=1)          # [B, D]
        D = probs.size(1)
        var = probs.var(dim=0, unbiased=False)        # [D]
        std = torch.sqrt(var + 1e-8)
        c = math.sqrt(D - 1.0) / D
        margin = torch.clamp(c - std, min=0.0)        # hinge
        return margin.mean()

    def _uniformity_loss(self, reps):
        if reps is None:
            return torch.tensor(0.0, device=self.model.parameters().__next__().device)
        reps = F.normalize(reps, dim=1)               # [B, d]
        B = reps.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=reps.device)
        dist2 = torch.cdist(reps, reps, p=2) ** 2
        mask = ~torch.eye(B, dtype=torch.bool, device=reps.device)
        pairwise = dist2[mask]
        if pairwise.numel() == 0:
            return torch.tensor(0.0, device=reps.device)

        sigma2 = torch.median(pairwise)
        sigma2 = torch.clamp(sigma2, min=1e-12)

        energy = torch.exp(-pairwise / (2.0 * sigma2))
        return energy.mean()
    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0

        for epoch in range(self.num_epoch):
            for x, y in train_dataloader:
                if x.size(0) <= 1:
                    continue
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                self.optimizer.zero_grad()
                pred = self.model(x)
                if isinstance(pred, (tuple, list)):
                    logits_candidates = [p for p in pred if torch.is_tensor(p)]
                    if len(logits_candidates) == 0:
                        raise ValueError("Error")
                    logits = logits_candidates[-1]
                else:
                    logits = pred

                ce_loss = criterion(logits, y)
                reps = self._cached_features
                lu = self._uniformity_loss(reps)
                lv = self._classifier_variance_loss(logits)
                loss = ce_loss + self.mu_feduv * lu + self.lambda_feduv * lv
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(logits, 1)
                correct = predicted.eq(y).sum().item()
                bs = y.size(0)

                train_loss += loss.item() * bs
                train_acc += correct
                train_total += bs

        local_solution = self.get_flat_model_params()
        param_dict = {
            "norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()
        }
        comp = self.num_epoch * train_total * self.flops
        return_dict = {
            "comp": comp,
            "loss": train_loss / train_total,
            "acc": train_acc / train_total
        }
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader, another_test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                if isinstance(pred, (tuple, list)):
                    logits = [p for p in pred if torch.is_tensor(p)][-1]
                else:
                    logits = pred

                loss = criterion(logits, y) 

                _, predicted = torch.max(logits, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss



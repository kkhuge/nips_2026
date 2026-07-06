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
            # for name, param in self.model.named_parameters():
            #     if "readout" not in name:  # Only train the final readout layer.
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False
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
                if "readout" not in name:  # Only train the final readout layer.
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # self.model.eval()

            for i in range(1):
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

    # Add the following method to the LrdWorker class in worker.py.
    def get_separated_grads(self, dataloader):
        """
        Get decoupled backbone and head gradients.
        Use gradient accumulation to avoid memory overflow.
        """
        self.model.train()
        self.optimizer.zero_grad()
        total_samples = 0

        # 1. Iterate over local data and accumulate gradients.
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()

            pred = self.model(x)
            loss = criterion(pred, y)

            # Multiply loss by batch size to compute the exact dataset-average gradient later.
            batch_size = y.size(0)
            (loss * batch_size).backward()
            total_samples += batch_size

        # 2. Separate and flatten Backbone and Head gradients.
        grad_b = []
        grad_h = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Divide by the total sample count to get the average gradient.
                    avg_grad = param.grad.detach().clone() / total_samples
                    if "readout" in name:
                        grad_h.append(avg_grad.view(-1))
                    else:
                        grad_b.append(avg_grad.view(-1))

        # Clear gradients so subsequent training is unaffected.
        self.optimizer.zero_grad()

        # Concatenate into 1D vectors.
        grad_b_flat = torch.cat(grad_b) if len(grad_b) > 0 else torch.tensor([])
        grad_h_flat = torch.cat(grad_h) if len(grad_h) > 0 else torch.tensor([])

        return grad_b_flat, grad_h_flat

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
                if "readout" not in name:  # Only train the final readout layer.
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
            # ---- Update local control variables ----
            delta_w = global_model_parameters - local_solution
            aaa = [torch.zeros_like(p) for p in self.model.parameters()]
            vector_to_parameters(delta_w, aaa)
            delta_w = aaa
            local_c_new = []
            delta_c = []
            for lc, gc, dw in zip(local_c, global_c, delta_w):
                # Compute the new client control variable.
                ci_new = lc - gc + (1.0 / (self.num_epoch * lr * len(train_dataloader))) * dw
                local_c_new.append(ci_new)

                # Compute the delta_c_i value to upload.
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
                if "readout" not in name:  # Only train the final readout layer.
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            self.model.eval()
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

                    # Current model parameters in vectorized form.
                    local_params = self.get_flat_model_params()

                    # FedDyn dynamic regularization target.
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

            # ---- Update alpha_i after local training finishes ----
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
                if "readout" not in name:  # Only train the final readout layer.
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
        # [FIX 1] Remove self.scaling_train = torch.tensor(12.0) here.
        # Use self.model.scaling_train directly.
        super(ETFWorker, self).__init__(model, optimizer, optimizer_last_layer, options)

    def get_label_counts(self, dataloader):
        # Helper: count samples per class for the current client for Balanced Loss.
        # This only needs to run once before training, not on every batch.
        targets = []
        if hasattr(dataloader.dataset, 'labels'):  # Assume the dataset has a labels attribute.
            targets = dataloader.dataset.labels
        elif hasattr(dataloader.dataset, 'tensors'):  # TensorDataset
            targets = dataloader.dataset.tensors[1].numpy()
        else:
            # Fallback: iterate once. This is slow.
            for _, y in dataloader:
                targets.append(y.numpy())
            targets = np.concatenate(targets)

        counts = np.bincount(targets, minlength=self.model.proto_classifier.num_classes)
        return torch.tensor(counts).float()

    def balanced_feature_loss(self, logits, targets, label_counts):
        """
        Implement Eq. (6) from the paper: Balanced Feature Loss.
        Logit adjustment form: logit + gamma * log(frequency).
        """
        # Compute class frequency pi_k_c.
        total_count = label_counts.sum()
        frequencies = label_counts / total_count

        # Avoid log(0).
        frequencies = frequencies + 1e-9

        # The paper does not specify an exact gamma. Balanced Softmax typically uses gamma=1.0.
        # Treat it as a hyperparameter if needed; this implementation defaults to 1.0.
        gamma = 1.0

        # log(pi^gamma) = gamma * log(pi)
        adjustment = gamma * torch.log(frequencies).to(logits.device)

        # Adjust logits.
        # The denominator in Eq. (6) is sum(exp(beta * v^T * mu + log(pi))).
        # This is equivalent to adding adjustment before the softmax input.
        adjusted_logits = logits + adjustment

        return F.cross_entropy(adjusted_logits, targets)

    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0

        # [FedETF] 1. Freeze ETF prototypes.
        for name, param in self.model.named_parameters():
            if "proto_classifier" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # [FIX] Count class samples for Balanced Loss.
        label_counts = self.get_label_counts(train_dataloader)
        if self.gpu:
            label_counts = label_counts.cuda()

        for epoch in range(self.num_epoch):
            for x, y in train_dataloader:
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()

                # ResNet ETF forward
                # Assume resnet_etf.py has been fixed so forward returns logits_etf, logits_ce, feature.
                # Also assume logits_etf has already been multiplied by scaling_train (beta) in the model.
                # If the model is not fixed, multiply manually here:
                # logits, _, _ = self.model(x)
                # logits = logits * self.model.scaling_train

                # This assumes the previously suggested resnet_etf.py fix is in use.
                logits_etf, _ = self.model(x)

                # If ResNet does not multiply by scaling internally, uncomment this:
                # logits_etf = logits_etf * self.model.scaling_train

                # [FIX 2] Use Balanced Feature Loss.
                # Replace the original Centerization and Margin terms with the paper's loss.
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
        # Note: beta is also included in flat_model_params and uploaded.

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

                # Use ETF logits during testing.
                logits, _ = self.model(x)

                # If model.forward does not multiply by scaling, multiply here too.
                # logits = logits * self.model.scaling_train

                loss = criterion(logits, y)  # The test set does not need Balanced Loss.
                _, predicted = torch.max(logits, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class FedUVWorker(Worker):
    """
    FedUV local training:
    loss = CE + mu * L_U + lambda * L_V
    where:
      - L_U: hyperspherical uniformity regularizer on the representation layer.
      - L_V: class-wise variance regularizer on the classifier probability distribution.
    Default hyperparameters in the paper: mu = 0.5, lambda = D/4 (D is the class count) :contentReference[oaicite:0]{index=0}
    """
    def __init__(self, model, optimizer, optimizer_last_layer, options):
        self.num_epoch = options['num_epoch']
        super(FedUVWorker, self).__init__(model, optimizer, optimizer_last_layer, options)

        # Class count and regularization strengths.
        self.num_classes = self.model.readout.out_features
        self.mu_feduv = 0.1 #0.1
        self.lambda_feduv = self.num_classes / 100 #1

        # Cache the penultimate-layer representation g_theta(X).
        self._cached_features = None

        # Register a forward hook on the final readout layer to capture its input as the representation layer.
        if hasattr(self.model, "readout"):
            self.model.readout.register_forward_hook(self._save_features_hook)
        else:
            raise ValueError("FedUVWorker requires the model to have a 'readout' attribute as the final layer.")

    # --------- Hook: save encoder representation ----------
    def _save_features_hook(self, module, inputs, output):
        # inputs is a tuple; inputs[0] is the readout input, i.e. the representation layer.
        self._cached_features = inputs[0]

    # --------- L_V: classifier variance regularizer ----------
    def _classifier_variance_loss(self, logits):
        """
        L_V in Eq. (2)(3):
        - Apply softmax over the batch dimension to obtain probability matrix P_hat.
        - Compute each class variance along the batch dimension -> std.
        - Use the hinge term max(0, c - std_j), then average over all classes.
        Here c = sqrt(D-1)/D, the theoretical standard deviation of a balanced one-hot distribution.
        """
        probs = torch.softmax(logits, dim=1)          # [B, D]
        D = probs.size(1)

        # Variance over the batch dimension without unbiased estimation.
        var = probs.var(dim=0, unbiased=False)        # [D]
        std = torch.sqrt(var + 1e-8)

        # Standard deviation constant c in the theoretical IID case.
        c = math.sqrt(D - 1.0) / D

        margin = torch.clamp(c - std, min=0.0)        # hinge
        return margin.mean()

    # --------- L_U: hyperspherical uniformity regularizer ----------
    def _uniformity_loss(self, reps):
        """
        L_U in Eq. (4):
        LU = E_{x,y} [ exp( - ||x - y||^2 / (2 sigma^2) ) ]
        Approximate the expectation using the mean over all pairs in the batch.
        sigma^2 is set to the median of pairwise squared distances.
        """
        if reps is None:
            # This should not happen in theory; return 0 as a safeguard.
            return torch.tensor(0.0, device=self.model.parameters().__next__().device)

        # Normalize onto the hypersphere.
        reps = F.normalize(reps, dim=1)               # [B, d]
        B = reps.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=reps.device)

        # pairwise squared distance: [B, B]
        dist2 = torch.cdist(reps, reps, p=2) ** 2

        # Remove diagonal self-pairs.
        mask = ~torch.eye(B, dtype=torch.bool, device=reps.device)
        pairwise = dist2[mask]
        if pairwise.numel() == 0:
            return torch.tensor(0.0, device=reps.device)

        sigma2 = torch.median(pairwise)
        sigma2 = torch.clamp(sigma2, min=1e-12)

        energy = torch.exp(-pairwise / (2.0 * sigma2))
        return energy.mean()

    # --------- Local training ----------
    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):
        """
        Mostly identical to standard FedAvg, with L_U and L_V regularizers added to the loss.
        """
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

                # Support models whose forward method returns a tuple, such as feature, logits, extra.
                if isinstance(pred, (tuple, list)):
                    # Prefer the last tensor as logits.
                    logits_candidates = [p for p in pred if torch.is_tensor(p)]
                    if len(logits_candidates) == 0:
                        raise ValueError("The tuple returned by model.forward contains no tensor that can be used as logits.")
                    logits = logits_candidates[-1]
                else:
                    logits = pred

                ce_loss = criterion(logits, y)

                # Get the encoder representation from the hook.
                reps = self._cached_features

                lu = self._uniformity_loss(reps)
                lv = self._classifier_variance_loss(logits)

                loss = ce_loss + self.mu_feduv * lu + self.lambda_feduv * lv

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                # Track training metrics.
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
        """
        Override the test function:
        1. Handle cases where model(x) returns a tuple.
        2. Compute classification loss only, without regularization terms.
        """
        self.model.eval()
        test_loss = test_acc = test_total = 0.

        with torch.no_grad():
            for x, y in test_dataloader:
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)

                # Handle multi-output cases: use the last tensor as logits.
                if isinstance(pred, (tuple, list)):
                    logits = [p for p in pred if torch.is_tensor(p)][-1]
                else:
                    logits = pred

                loss = criterion(logits, y)  # Regularization is usually omitted during testing.

                _, predicted = torch.max(logits, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class FedSMOOWorker(Worker):
    def __init__(self, model, optimizer, optimizer_last_layer, options):
        self.num_epoch = options.get('num_epoch', 5)
        # According to the paper, r is usually 0.01 or 0.1; this implementation defaults to 0.1.
        self.rho = 0.1
        self.beta = 10
        super(FedSMOOWorker, self).__init__(model, optimizer, optimizer_last_layer, options)

    def local_train(self, train_dataloader, another_train_dataloader, round_i, global_w, global_s, local_lambda,
                    local_mu, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0
        lr = self.optimizer.param_groups[0]['lr']

        # =================================================================================
        # Stage 1: before round 450, run the full FedSMOO algorithm (full-parameter update + SAM + dynamic regularization).
        # =================================================================================
        if round_i < 2000:

            # Initialize variables used to compute final tilde_s_i.
            hat_s_ik = [torch.zeros_like(p) for p in self.model.parameters()]

            for epoch in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    # ---- Step 1: compute the standard gradient at w_{i,k}^t ----
                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # Compute perturbations s_{i,k} and hat{s}_{i,k}.
                    s_ik_list = []
                    s_ik_norm_sq = 0.0
                    for p, l_mu, g_s in zip(self.model.parameters(), local_mu, global_s):
                        if p.grad is None:
                            s_ik_list.append(torch.zeros_like(p))
                        else:
                            s_ik = (p.grad - l_mu) - g_s
                            s_ik_list.append(s_ik)
                            s_ik_norm_sq += torch.sum(s_ik ** 2)

                    s_ik_norm = torch.sqrt(s_ik_norm_sq)

                    hat_s_ik = []
                    with torch.no_grad():
                        for idx, (p, l_mu, g_s) in enumerate(zip(self.model.parameters(), local_mu, global_s)):
                            if p.grad is None:
                                hat_s_ik.append(torch.zeros_like(p))
                                continue

                            # Normalize to radius r (rho).
                            if s_ik_norm > 0:
                                h_s = self.rho * s_ik_list[idx] / s_ik_norm
                            else:
                                h_s = torch.zeros_like(p)
                            hat_s_ik.append(h_s)

                            # Update dual variable mu_i = mu_i + (hat{s}_{i,k} - s).
                            l_mu.add_(h_s - g_s)

                            # Move parameters to the maximum-loss point w + hat{s}.
                            p.add_(h_s)

                    # ---- Step 2: compute flattened gradient hat{g}_{i,k}^t at the perturbed parameters ----
                    self.optimizer.zero_grad()
                    pred_adv = self.model(x)
                    loss_adv = criterion(pred_adv, y)
                    loss_adv.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # Move back to the original point and modify the gradient there.
                    with torch.no_grad():
                        for p, h_s, l_lambda, g_w in zip(self.model.parameters(), hat_s_ik, local_lambda, global_w):
                            if p.grad is None: continue
                            p.sub_(h_s)
                            # Inject the correction direction into .grad.
                            p.grad.add_(-l_lambda + (1.0 / self.beta) * (p - g_w))

                    # Update.
                    self.optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    train_loss += loss.item() * y.size(0)
                    train_acc += predicted.eq(y).sum().item()
                    train_total += y.size(0)

            # Collect variables to upload.
            tilde_s = []
            with torch.no_grad():
                for p, l_mu, h_s, l_lambda, g_w in zip(self.model.parameters(), local_mu, hat_s_ik, local_lambda,
                                                       global_w):
                    t_s = l_mu - h_s
                    tilde_s.append(t_s)
                    l_lambda.sub_((1.0 / self.beta) * (p - g_w))

            local_solution = self.get_flat_model_params()
            comp = self.num_epoch * train_total * self.flops * 2  # SAM uses two passes.
            return_dict = {
                "comp": comp,
                "loss": train_loss / train_total,
                "acc": train_acc / train_total,
                "norm": torch.norm(local_solution).item(),
                "max": local_solution.max().item(),
                "min": local_solution.min().item()
            }
            return local_solution, return_dict, tilde_s, local_lambda, local_mu

        # =================================================================================
        # Stage 2: from round 450 onward, switch to FedAvg logic (freeze the backbone and fine-tune only readout).
        # =================================================================================
        else:

            # Freeze feature extraction layers and enable only readout.
            for name, param in self.model.named_parameters():
                if "readout" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            self.model.eval()

            # Follow the provided logic: Stage 2 runs only 1 epoch.
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
                    train_loss += loss.item() * y.size(0)
                    train_acc += predicted.eq(y).sum().item()
                    train_total += y.size(0)

            local_solution = self.get_flat_model_params()
            # Standard SGD single forward/backward pass.
            comp = 1 * train_total * self.flops

            # In basic FedAvg mode, SAM variables are no longer used or updated.
            # Return all-zero tilde_s so the server-side global perturbation s does not drift; return dual variables unchanged.
            tilde_s = [torch.zeros_like(p) for p in self.model.parameters()]

            return_dict = {
                "comp": comp,
                "loss": train_loss / train_total,
                "acc": train_acc / train_total,
                "norm": torch.norm(local_solution).item(),
                "max": local_solution.max().item(),
                "min": local_solution.min().item()
            }

            return local_solution, return_dict, tilde_s, local_lambda, local_mu


# Add at the end of worker.py:
class FedGFWorker(Worker):
    def __init__(self, model, optimizer, optimizer_last_layer, options):
        super(FedGFWorker, self).__init__(model, optimizer, optimizer_last_layer, options)
        # ======= FedGF-specific parameters are hard-coded here =======
        self.rho = 0.02
        # ==============================================

    def local_train(self, train_dataloader, another_train_dataloader, round_i, global_perturbed_model=None,
                    c_interp=0.0, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0

        # =====================================================================
        # Stage 1: before round 950, run FedGF bidirectional perturbation interpolation SAM.
        # =====================================================================
        if round_i < 550:

            # global_perturbed_model is now passed in as a pure perturbation vector.
            g_pert_list = []
            if global_perturbed_model is not None:
                offset = 0
                for p in self.model.parameters():
                    numel = p.numel()
                    g_pert_list.append(global_perturbed_model[offset:offset + numel].view_as(p).to(p.device))
                    offset += numel

            for epoch in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    # ---- First forward pass: compute local gradients ----
                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # Compute the L2 norm of the full-network gradient.
                    grad_norm = torch.norm(
                        torch.stack([p.grad.norm(p=2) for p in self.model.parameters() if p.grad is not None]),
                        p=2
                    )

                    epsilons = []
                    with torch.no_grad():
                        for idx, p in enumerate(self.model.parameters()):
                            if p.grad is None:
                                epsilons.append(torch.zeros_like(p))
                                continue

                            # Local perturbation vector.
                            if grad_norm > 0:
                                loc_pert = self.rho * p.grad / grad_norm
                            else:
                                loc_pert = torch.zeros_like(p)

                            # ====== Core fix 1: do not subtract p again; interpolate directly ======
                            g_pert_vec = g_pert_list[idx] if g_pert_list else torch.zeros_like(p)
                            eps_interp = c_interp * g_pert_vec + (1.0 - c_interp) * loc_pert
                            # ===================================================

                            epsilons.append(eps_interp)
                            p.add_(eps_interp)  # Move to the trial point.

                    # ---- Second forward pass: compute the flattened gradient ----
                    self.optimizer.zero_grad()
                    pred_adv = self.model(x)
                    loss_adv = criterion(pred_adv, y)
                    loss_adv.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # Move back to the original point and use the interpolated-point gradient for the actual weight update.
                    with torch.no_grad():
                        for p, eps in zip(self.model.parameters(), epsilons):
                            if p.grad is None: continue
                            p.sub_(eps)

                    self.optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    train_loss += loss.item() * y.size(0)
                    train_acc += predicted.eq(y).sum().item()
                    train_total += y.size(0)

            local_solution = self.get_flat_model_params()
            comp = self.num_epoch * train_total * self.flops * 2  # SAM requires twice the compute.
            return_dict = {
                "comp": comp, "loss": train_loss / train_total, "acc": train_acc / train_total,
                "norm": torch.norm(local_solution).item(), "max": local_solution.max().item(),
                "min": local_solution.min().item()
            }
            return local_solution, return_dict

        # =====================================================================
        # Stage 2: from round 950 onward, unfreeze layer4 for GroupNorm and reuse the main optimizer.
        # =====================================================================
        else:
            for name, param in self.model.named_parameters():
                if "readout" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            self.model.eval()


            # ====== Core fix 3: use self.optimizer consistently and reduce the learning rate ======

            for i in range(self.num_epoch):  # Stage 2 defaults to 1 epoch.
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu: x, y = x.cuda(), y.cuda()

                    self.optimizer_last_layer.zero_grad()  # Must switch to the main optimizer.
                    pred = self.model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                    self.optimizer_last_layer.step()       # Must switch to the main optimizer.

                    _, predicted = torch.max(pred, 1)
                    train_loss += loss.item() * y.size(0)
                    train_acc += predicted.eq(y).sum().item()
                    train_total += y.size(0)

            local_solution = self.get_flat_model_params()
            comp = 1 * train_total * self.flops
            return_dict = {
                "comp": comp, "loss": train_loss / train_total, "acc": train_acc / train_total,
                "norm": torch.norm(local_solution).item(), "max": local_solution.max().item(),
                "min": local_solution.min().item()
            }
            return local_solution, return_dict


class FedSAMWorker(Worker):
    def __init__(self, model, optimizer, optimizer_last_layer, options):
        self.num_epoch = options.get('num_epoch', 5)
        # ASAM recommends a larger rho such as 0.5; standard SAM recommends 0.05.
        self.adaptive = options.get('adaptive', True)  # Enable ASAM by default.
        self.rho = 0.01 if self.adaptive else 0.05
        super(FedSAMWorker, self).__init__(model, optimizer, optimizer_last_layer, options)

    def _grad_norm(self):
        # Compute gradient norm. ASAM accounts for the weight scale.
        norm = 0.0
        if self.adaptive:
            for p in self.model.parameters():
                if p.grad is not None:
                    scale = torch.abs(p) + 1e-4
                    norm += torch.sum((scale * p.grad) ** 2)
        else:
            for p in self.model.parameters():
                if p.grad is not None:
                    norm += torch.sum(p.grad ** 2)
        return torch.sqrt(norm)

    def local_train(self, train_dataloader, another_train_dataloader, round_i, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0

        # ==========================================================
        # Stages 1 and 2 (rounds 0-474): full-model SAM / ASAM training.
        # ==========================================================
        if round_i < 550:
            # Unfreeze all layers for full-parameter fine-tuning.
            for name, param in self.model.named_parameters():
                param.requires_grad = True

            for epoch in range(self.num_epoch):
                for x, y in train_dataloader:
                    x = self.flatten_data(x)
                    if self.gpu:
                        x, y = x.cuda(), y.cuda()

                    # Step 1: first forward pass to compute loss and initial gradients.
                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # Step 2: compute perturbation epsilon and move model parameters to the trial point w + eps.
                    grad_norm = self._grad_norm()
                    epsilons = []
                    with torch.no_grad():
                        for p in self.model.parameters():
                            if p.grad is None:
                                epsilons.append(torch.zeros_like(p))
                                continue
                            if self.adaptive:
                                # ASAM logic.
                                scale = torch.abs(p) + 1e-4
                                eps = self.rho * scale * p.grad / (grad_norm + 1e-12)
                            else:
                                # Standard SAM logic.
                                eps = self.rho * p.grad / (grad_norm + 1e-12)

                            epsilons.append(eps)
                            p.add_(eps)

                    # Step 3: run the second forward pass at the trial point (w + eps) and compute the flattened gradient.
                    self.optimizer.zero_grad()
                    pred_adv = self.model(x)
                    loss_adv = criterion(pred_adv, y)
                    loss_adv.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

                    # Step 4: move parameters back to the original point (w), then update with the new gradients.
                    with torch.no_grad():
                        for p, eps in zip(self.model.parameters(), epsilons):
                            if p.grad is None: continue
                            p.sub_(eps)

                    self.optimizer.step()

                    _, predicted = torch.max(pred, 1)
                    train_loss += loss.item() * y.size(0)
                    train_acc += predicted.eq(y).sum().item()
                    train_total += y.size(0)

            local_solution = self.get_flat_model_params()
            comp = self.num_epoch * train_total * self.flops * 2  # SAM needs two backward passes, doubling compute.
            return_dict = {
                "comp": comp,
                "loss": train_loss / train_total,
                "acc": train_acc / train_total,
                "norm": torch.norm(local_solution).item(),
                "max": local_solution.max().item(),
                "min": local_solution.min().item()
            }
            return local_solution, return_dict

        # ==========================================================
        # Stage 3 (rounds 475-499): freeze the backbone and fine-tune only the readout layer.
        # ==========================================================
        else:
            for name, param in self.model.named_parameters():
                if "readout" not in name:  # Only train the final readout layer.
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            self.model.eval()

            for i in range(self.num_epoch):  # Head fine-tuning runs for 1 epoch.
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
            param_dict = {
                "norm": torch.norm(local_solution).item(),
                "max": local_solution.max().item(),
                "min": local_solution.min().item()
            }
            comp = 1 * train_total * self.flops
            return_dict = {
                "comp": comp,
                "loss": train_loss / train_total,
                "acc": train_acc / train_total
            }
            return_dict.update(param_dict)
            return local_solution, return_dict

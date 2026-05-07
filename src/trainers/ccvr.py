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
from src.utils.torch_utils import get_flat_params_from


criterion = torch.nn.CrossEntropyLoss()

error_dir = 'result_error/ccvr'
weight_change_dir = "result_weight_change/ccvr"
theta_dir = "result_theta/ccvr"
output_dir = "result_output_differ/ccvr"
loss_dir = "result_loss/ccvr"
acc_dir = "result_acc/ccvr"

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


class CCVRTrainer(BaseTrainer):
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
        super(CCVRTrainer, self).__init__(options, dataset, another_dataset, worker=worker)

        # CCVR hyper-params
        self.ccvr_base_M = 10000 # base number of virtual features
        self.ccvr_epochs = 100  # retrain epochs for classifier
        self.ccvr_lr = 0.1  # lr for classifier retrain
        self.ccvr_proportional = True
        self.ccvr_apply_relu =  True
        self.ccvr_apply_tukey = True
        self.ccvr_tukey_beta = 0.5

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

        # ---------------- CCVR server-side calibration ----------------
        print('===================== Start CCVR Calibration (Server) =====================')
        model_path = "stage1_resnet18_tinyimagenet_0.1.pth"
        self.worker.model.load_state_dict(torch.load(model_path))
        global_model = self.worker.model
        global_model.eval()

        # # load final global weights into a model object for calibration
        # global_model = copy.deepcopy(self.clients[0].worker.model)
        # self.worker.set_flat_model_params(self.latest_model)
        # global_model.load_state_dict(self.clients[0].worker.model.state_dict())
        # global_model.eval()


        self.loss_list_test, self.acc_list_test = self.ccvr_calibrate(global_model)




        np.save(loss_dir + '/loss_test' + self.dataset + self.model , self.loss_list_test)  #
        np.save(acc_dir + '/acc_test' + self.dataset + self.model , self.acc_list_test)


        # Save tracked information
        self.metrics.write()

    # ------------------------- CCVR core -------------------------
    def ccvr_calibrate(self, global_model):
        device = next(global_model.parameters()).device
        C = global_model.readout.out_features
        d = global_model.readout.in_features
        global_max_R = 0.0
        client_stats = []
        for client in self.clients:
            stats_k, client_R = self.compute_client_class_stats(global_model, client.train_dataloader, C, d, device)
            client_stats.append(stats_k)


            if client_R > global_max_R:
                global_max_R = client_R

        sensitivity_score = global_max_R + (global_max_R ** 2)

        print(f"\n[Privacy Analysis] Feature Dimension d={d}")
        print(f"[Privacy Analysis] Max Feature Norm (R) = {global_max_R:.4f}")
        print(f"[Privacy Analysis] Privacy Sensitivity (R + R^2) = {sensitivity_score:.4f}")
        print("---------------------------------------------------------------")

        # ---- 2) server aggregate stats ----
        Nc_list = torch.zeros(C, device=device)
        for stats_k in client_stats:
            for c in range(C):
                Nc_list[c] += stats_k[c]["Nc_k"].to(device) 

        totalN = Nc_list.sum().item()
        mu_global = [torch.zeros(d, device=device) for _ in range(C)]
        Sigma_global = [torch.zeros(d, d, device=device) for _ in range(C)]

        # global mean
        for c in range(C):
            if Nc_list[c] < 1: continue
            for stats_k in client_stats:
                Nc_k = stats_k[c]["Nc_k"].to(device)
                if Nc_k > 0:
                    mu_global[c] += (Nc_k / Nc_list[c]) * stats_k[c]["mu_c_k"].to(device)

        # global covariance
        for c in range(C):
            Nc = Nc_list[c]
            if Nc <= 1:
                Sigma_global[c] = torch.eye(d, device=device)
                continue

            part1 = torch.zeros(d, d, device=device)
            part2 = torch.zeros(d, d, device=device)
            for stats_k in client_stats:
                Nc_k = stats_k[c]["Nc_k"].to(device)
                if Nc_k <= 0: continue

                sigma_k = stats_k[c]["Sigma_c_k"].to(device)
                mu_k = stats_k[c]["mu_c_k"].to(device)

                if Nc_k > 1:
                    part1 += ((Nc_k - 1) / (Nc - 1)) * sigma_k
                part2 += (Nc_k / (Nc - 1)) * torch.ger(mu_k, mu_k)

            Sigma_global[c] = part1 + part2 - (Nc / (Nc - 1)) * torch.ger(mu_global[c], mu_global[c])
            Sigma_global[c] += 1e-5 * torch.eye(d, device=device)

        del client_stats
        import gc
        gc.collect()

        virtual_Z = []
        virtual_Y = []
        for c in range(C):
            Nc = Nc_list[c].item()
            if Nc < 1: continue

            if self.ccvr_proportional:
                Mc = max(1, int(self.ccvr_base_M * (Nc / totalN)))
            else:
                Mc = self.ccvr_base_M

            dist = torch.distributions.MultivariateNormal(mu_global[c], covariance_matrix=Sigma_global[c])
            zc = dist.sample((Mc,))

            if self.ccvr_apply_relu: zc = F.relu(zc)
            if self.ccvr_apply_tukey:
                beta = self.ccvr_tukey_beta
                zc = torch.sign(zc) * (torch.abs(zc) ** beta)

            virtual_Z.append(zc.cpu())
            virtual_Y.append(torch.full((Mc,), c, device='cpu', dtype=torch.long))

        virtual_Z = torch.cat(virtual_Z, dim=0)
        virtual_Y = torch.cat(virtual_Y, dim=0)

        # ---- 4) retrain classifier on server ----
        with torch.enable_grad():
            for name, p in global_model.named_parameters():
                if "readout" not in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

            opt = SGD(global_model.readout.parameters(), lr=self.ccvr_lr, weight_decay=0.0001)

            batch_size = 64
            num_batches = int(np.ceil(len(virtual_Y) / batch_size))

            # self.latest_model = get_flat_params_from(global_model)
            # loss_test, accuracy_test = self.test_latest_model_on_evaldata(0)
            # print(f"[CCVR Final], test_loss={loss_test:.4f}, test_acc={accuracy_test:.4f}")
            for ep in range(self.ccvr_epochs):
                perm = torch.randperm(len(virtual_Y))
                z_shuf = virtual_Z[perm]  # CPU tensor
                y_shuf = virtual_Y[perm]  # CPU tensor

                epoch_loss = 0.0
                for b in range(num_batches):
                    zb = z_shuf[b * batch_size:(b + 1) * batch_size]
                    yb = y_shuf[b * batch_size:(b + 1) * batch_size]

                    zb = zb.to(device)
                    yb = yb.to(device)

                    logits = global_model.readout(zb)
                    loss = F.cross_entropy(logits, yb)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()


                # self.latest_model = get_flat_params_from(global_model)
                # self.worker.set_flat_model_params(get_flat_params_from(global_model))
                accuracy_test, loss_test  = self.test_model(global_model, self.centralized_test_dataloader)

                self.loss_list_test.append(loss_test)
                self.acc_list_test.append(accuracy_test)
                print(f"[CCVR Final] ep={ep}, test_loss={loss_test:.4f}, test_acc={accuracy_test:.4f}")

        return self.loss_list_test, self.acc_list_test


    # ------------------------- helpers -------------------------
    def get_feature_extractor(self, model: nn.Module):
        """
        get feature extractor part.
        We assume model has `readout` as final linear head.
        We'll wrap a forward hook to obtain features right before readout.
        """
        return model  # placeholder: we only need its params to freeze

    @torch.no_grad()
    def extract_features(self, model, x):
        """
        Robustly extract features before readout.
        Tries common APIs, else uses hook on readout input.
        """
        if hasattr(model, "extract_features"):
            return model.extract_features(x)
        if hasattr(model, "forward_features"):
            return model.forward_features(x)
        if hasattr(model, "features"):
            z = model.features(x)
            if z.dim() > 2:
                z = torch.flatten(z, 1)
            return z

        feats = {}

        def hook_fn(m, inp, out):
            feats["z"] = inp[0].detach()

        h = model.readout.register_forward_hook(hook_fn)
        _ = model(x)
        h.remove()
        z = feats["z"]
        if z.dim() > 2:
            z = torch.flatten(z, 1)
        return z

    @torch.no_grad()
    def compute_client_class_stats(self, global_model, loader, C, d, device):
        storage_device = 'cpu'
        sum_x = torch.zeros(C, d, device=storage_device)
        sum_xxT = torch.zeros(C, d, d, device=storage_device)
        N_c = torch.zeros(C, device=storage_device)

        client_max_R = 0.0

        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            z = self.extract_features(global_model, data)

            batch_norms = torch.norm(z, p=2, dim=1)
            current_batch_max = batch_norms.max().item()

            if current_batch_max > client_max_R:
                client_max_R = current_batch_max
            # ==========================================

            unique_classes = torch.unique(targets)
            for c in unique_classes:
                mask = (targets == c)
                z_c = z[mask]
                n_c_batch = z_c.size(0)

                batch_sum_x = z_c.sum(dim=0)
                batch_xxT = torch.mm(z_c.T, z_c)

                c_idx = int(c.item())
                N_c[c_idx] += n_c_batch
                sum_x[c_idx] += batch_sum_x.to(storage_device)
                sum_xxT[c_idx] += batch_xxT.to(storage_device)

            del z
            # torch.cuda.empty_cache()

        stats_k = {}
        for c in range(C):
            n = N_c[c].item()
            mu_cpu = sum_x[c] / n if n > 0 else torch.zeros(d)

            if n > 1:
                term1 = sum_xxT[c]
                term2 = n * torch.ger(mu_cpu, mu_cpu)
                Sigma_cpu = (term1 - term2) / (n - 1)
            else:
                Sigma_cpu = torch.eye(d)

            stats_k[c] = {
                "Nc_k": N_c[c],  # CPU
                "mu_c_k": mu_cpu,  # CPU
                "Sigma_c_k": Sigma_cpu  # CPU
            }

        return stats_k, client_max_R

    def test_model(self, model, test_dataloader):
        # model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # from IPython import embed
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                features = self.extract_features(model, x)

                if self.ccvr_apply_relu:
                    features = F.relu(features)
                if self.ccvr_apply_tukey:
                    features = torch.sign(features) * (torch.abs(features) ** self.ccvr_tukey_beta)

                pred = model.readout(features)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)
        test_acc = test_acc / test_total
        test_loss = test_loss / test_total
        return test_acc, test_loss

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num

        return averaged_solution.detach()


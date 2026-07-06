from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from torch.optim import SGD
import numpy as np
import torch
import os
import copy
from src.models.client import Client

# Initialize the result save directory.
for d in ["result_loss/fedrep", "result_acc/fedrep"]:
    if not os.path.exists(d):
        os.makedirs(d)


class FedRepWorker(LrdWorker):
    """
    FedRep Worker: implements alternating updates between Body (Representation) and Head.
    """

    def __init__(self, model, optimizer, options):
        # Note: FedRep needs two optimizers, or dynamic optimizer parameter switching during training.
        # We pass the main optimizer here, but freeze parameters by stage in the train method.
        super(FedRepWorker, self).__init__(model, optimizer, optimizer,options)

        # Define Body and Head parameter groups.
        # Assume model.readout is the Head and the rest is the Body (Representation).
        self.body_params = []
        self.head_params = []
        self.num_head_epochs = options['num_epoch']
        for name, param in self.model.named_parameters():
            if "readout" in name:
                self.head_params.append(param)
            else:
                self.body_params.append(param)

        # Create separate optimizers for the two stages to avoid mixing momentum states.
        # Phase 1: optimize only the Head.
        self.optimizer_head = SGD(self.head_params, lr=options['lr'], weight_decay=options['wd'])
        # Phase 2: optimize only the Body.
        self.optimizer_body = SGD(self.body_params, lr=options['lr'], weight_decay=options['wd'])

    def get_body_params(self):
        """Extract Representation (Body) parameters for global aggregation."""
        params = []
        for name, param in self.model.named_parameters():
            if "readout" not in name:
                params.append(param.data.view(-1))
        return torch.cat(params)

    def set_body_params(self, flat_params):
        """Synchronize Body parameters from the server while keeping the local Head unchanged."""
        offset = 0
        for name, param in self.model.named_parameters():
            if "readout" not in name:
                numel = param.numel()
                param.data.copy_(flat_params[offset:offset + numel].view_as(param.data))
                offset += numel

    def train(self, train_dataloader, round_i, client_id=None):
        """
        Core alternating training logic for FedRep.
        Phase 1: Freeze Body, Train Head (for multiple epochs)
        Phase 2: Freeze Head, Train Body (for 1 epoch)
        """
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()

        # --- Phase 1: Train Head ---
        # Freeze Body.
        for param in self.body_params:
            param.requires_grad = False
        for param in self.head_params:
            param.requires_grad = True

        # Head training usually needs more epochs to adapt to the new Body.
        # Use options['num_epoch'] as the Head training epoch count.

        for epoch in range(self.num_head_epochs):
            for x, y in train_dataloader:
                x, y = x.cuda(), y.cuda()
                self.optimizer_head.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head_params, max_norm=10.0)
                self.optimizer_head.step()

        # --- Phase 2: Train Body ---
        # Freeze Head.
        for param in self.body_params:
            param.requires_grad = True
        for param in self.head_params:
            param.requires_grad = False

        # Body training usually runs for only 1 epoch or fewer steps.
        # This can also be controlled by parameters; the default here is 1.
        num_body_epochs = 1

        train_stats = {}  # Simple records.

        for epoch in range(num_body_epochs):
            total_loss = 0
            correct = 0
            total_samples = 0

            for x, y in train_dataloader:
                x, y = x.cuda(), y.cuda()
                self.optimizer_body.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.body_params, max_norm=10.0)
                self.optimizer_body.step()

                total_loss += loss.item() * y.size(0)
                _, predicted = torch.max(pred, 1)
                correct += predicted.eq(y).sum().item()
                total_samples += y.size(0)

            train_stats['loss'] = total_loss / total_samples
            train_stats['acc'] = correct / total_samples

        # After training, restore gradient requirements for all parameters for safety, although the next train call resets them.
        for param in self.model.parameters():
            param.requires_grad = True

        return train_stats


class FedRepTrainer(BaseTrainer):
    def __init__(self, options, dataset, another_dataset):
        self.options = options
        self.device = torch.device('cuda' if options['gpu'] else 'cpu')

        # 1. Initialize a temporary model to get parameter shapes.
        temp_model = choose_model(options)
        if options['gpu']:
            temp_model = temp_model.to(self.device)

        # Temporary worker.
        # Passing None for optimizer is enough because Worker redefines optimizers internally.
        temp_worker = FedRepWorker(temp_model, None,  options)

        # FedRep aggregates the Representation (Body), so get Body parameters here.
        self.latest_body = temp_worker.get_body_params().detach().to(self.device)

        # 2. Initialize result lists.
        self.clients_rep = []
        self.acc_list_test = []
        self.loss_list_test = []

        # 3. Call parent initialization, which creates self.worker = temp_worker.
        super(FedRepTrainer, self).__init__(options, dataset, another_dataset, worker=temp_worker)

        # 4. Core step: create an independent Worker and Model for each client.
        self.clients_rep = self.setup_clients_rep(dataset, another_dataset)

    def setup_clients_rep(self, dataset, another_dataset):
        """
        Assign a fully independent model to each client to preserve personalized Head state.
        """
        users, groups, train_data, test_data = dataset
        _, _, another_train_data, another_test_data = another_dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            user_id = int(user[-5:]) if isinstance(user, str) and len(user) >= 5 else int(user)

            # --- Key point: each client owns an independent model object ---
            local_model = choose_model(self.options).to('cpu')

            # Create a FedRep-specific Worker.
            # The optimizer argument is mainly a placeholder for parent interface compatibility.
            # The actual head and body optimizers are created in FedRepWorker.__init__.
            dummy_optimizer = SGD(local_model.parameters(), lr=self.options['lr'])
            local_worker = FedRepWorker(local_model, dummy_optimizer, self.options)

            # Create the Client object.
            c_rep = Client(user_id, group, train_data[user], test_data[user],
                           another_train_data[user], another_test_data[user],
                           self.batch_size, local_worker)
            all_clients.append(c_rep)

        return all_clients

    def select_clients_rep(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients_rep))
        np.random.seed(seed)
        return np.random.choice(self.clients_rep, num_clients, replace=False).tolist()

    def train(self):
        print(f'>>> FedRep Training: Alternating updates (Head -> Body)')

        for round_i in range(self.num_round):
            # 1. Evaluate personalized performance with Global Body + Local Head.
            if len(self.clients_rep) > 0:
                self.evaluate_personalized(round_i)

            # 2. Select clients.
            selected_clients = self.select_clients_rep(seed=round_i)
            solns = []
            stats = []

            for c in selected_clients:
                # GPU memory management.
                c.worker.model.to(self.device)
                c.worker.device = self.device

                # A. Download the latest Global Representation (Body).
                # Note: keep the Head state from the end of the previous local training round unchanged.
                c.worker.set_body_params(self.latest_body)

                # B. Local training with FedRep logic: train the Head first, then the Body.
                # Use Client.local_train -> Worker.train.
                # Modify Client.local_train behavior here, or call worker.train directly for stability,
                # because Worker.train implements the special alternating logic.

                # Get data loaders.
                train_loader = c.train_dataloader

                # Run FedRep-specific alternating training.
                # train returns statistics.
                stat = c.worker.train(train_loader, round_i)
                num_sample = len(c.train_data)

                # C. Extract updated Body parameters for upload.
                # Head parameters remain in the local model and persist with the object.
                body_params = c.worker.get_body_params().detach().to(self.device)

                solns.append((num_sample, body_params))
                stats.append(stat)

                # GPU memory management: move back to CPU.
                c.worker.model.to('cpu')

            # 3. Aggregation: the server updates the Representation (Body).
            self.latest_body = self.aggregate(solns)

        # Final evaluation.
        self.evaluate_personalized(self.num_round)
        self.save_all_results()
        self.metrics.write()

    def aggregate(self, solns):
        """Standard FedAvg aggregation, but only for Body parameters."""
        if not solns: return self.latest_body
        total_samples = sum([s[0] for s in solns])
        avg_body = torch.zeros_like(solns[0][1])
        for num_sample, local_body in solns:
            avg_body += (num_sample / total_samples) * local_body.to(self.device)
        return avg_body.detach()

    def evaluate_personalized(self, round_i):
        """
        Evaluate personalized accuracy for each client with Global Body + Local Head.
        """
        correct_list = []
        num_list = []
        loss_list = []
        criterion = torch.nn.CrossEntropyLoss()

        for c in self.clients_rep:
            c.worker.model.to(self.device)
            c.worker.device = self.device

            # Ensure evaluation uses the latest Global Body.
            c.worker.set_body_params(self.latest_body)

            c.worker.model.eval()

            test_loss = 0.
            test_acc = 0
            test_total = 0

            with torch.no_grad():
                for data, target in c.test_dataloader:
                    data, target = data.to(self.device), target.to(self.device)
                    pred = c.worker.model(data)
                    loss = criterion(pred, target)

                    _, predicted = torch.max(pred, 1)
                    test_acc += predicted.eq(target).sum().item()
                    test_loss += loss.item() * target.size(0)
                    test_total += target.size(0)

            correct_list.append(test_acc)
            num_list.append(test_total)
            loss_list.append(test_loss)

            c.worker.model.to('cpu')

        num_all = np.sum(num_list)
        avg_acc = np.sum(correct_list) / num_all
        avg_loss = np.sum(loss_list) / num_all

        self.acc_list_test.append(avg_acc)
        self.loss_list_test.append(avg_loss)

        # Fix: avoid errors when options['noprint'] does not exist.
        noprint = self.options.get('noprint', False)
        if not noprint:
            print(f'Round {round_i} - FedRep Personalized Avg Acc: {avg_acc:.4f}, Loss: {avg_loss:.4f}')

    def save_all_results(self):
        ds = self.options.get("dataset", "data")
        md = self.options.get("model", "model")
        np.save(f'result_acc/fedrep/acc_test_{ds}_{md}', self.acc_list_test)
        np.save(f'result_loss/fedrep/loss_test_{ds}_{md}', self.loss_list_test)
        print(">>> FedRep Results saved.")

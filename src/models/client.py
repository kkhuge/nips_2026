import time
from torch.utils.data import DataLoader
import torch
import copy


class Client(object):
    """Base class for all local clients

    Outputs of gradients or local_solutions will be converted to np.array
    in order to save CUDA memory.
    """
    def __init__(self, cid, group, train_data, test_data, another_train_data, another_test_data, batch_size, worker):
        self.cid = cid
        self.group = group
        self.worker = worker
        self.train_data = train_data
        self.test_data = test_data
        self.another_train_data = another_train_data
        self.another_test_data = another_test_data
        self.local_c = [torch.zeros_like(param.data).to(param.device) for param in self.worker.model.parameters()]
        self.local_alpha = torch.zeros_like(self.worker.get_flat_model_params())
        # Do not call .to(param.device); use .cpu() or leave the device unspecified.
        self.local_lambda = [torch.zeros_like(param.data).cpu() for param in self.worker.model.parameters()]
        self.local_mu = [torch.zeros_like(param.data).cpu() for param in self.worker.model.parameters()]

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # Keep shuffle=False when measuring the kernel matrix.
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.another_train_dataloader = DataLoader(another_train_data, batch_size=batch_size, shuffle=True)  # Keep shuffle=False when measuring the kernel matrix.
        self.another_test_dataloader = DataLoader(another_test_data, batch_size=batch_size, shuffle=True)

    def get_model_params(self):
        """Get model parameters"""
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        """Set model parameters"""
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def get_flat_grads(self):
        """Get model gradient"""
        grad_in_tenser = self.worker.get_flat_grads(self.train_dataloader)
        return grad_in_tenser.cpu().detach().numpy()

    def get_grad(self):
        stats = {'id': self.cid}
        grad_in_tenser = self.worker.get_grad(self.train_dataloader)
        return (len(self.train_data), grad_in_tenser.cpu().detach().numpy()), stats

    def get_prediction(self):
        "Get prediction value"
        prediction = self.worker.get_prediction(self.train_dataloader)
        return prediction.cpu().detach().numpy()

    def get_jacobian(self):
        """Get model gradient"""
        jacobian_in_tenser = self.worker.get_jacobian(self.train_dataloader)
        return jacobian_in_tenser

    def solve_grad(self):
        """Get model gradient with cost"""
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * len(self.train_data)
        bytes_r = self.worker.model_bytes
        stats = {'id': self.cid, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}
        grads = self.get_flat_grads()  # Return grad in numpy array

        return (len(self.train_data), grads), stats

    def solve_jacobian(self):
        """Get model gradient with cost"""
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * len(self.train_data)
        bytes_r = self.worker.model_bytes
        stats = {'id': self.cid, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}

        jacobian = self.get_jacobian()  # Return grad in numpy array

        return (len(self.train_data), jacobian), stats

    # Add the following method to the Client class in client.py.
    def get_separated_grads(self):
        """Get the decoupled gradient for the current client over the full local training set."""
        grad_b_tensor, grad_h_tensor = self.worker.get_separated_grads(self.train_dataloader)
        num_samples = len(self.train_data)

        return num_samples, grad_b_tensor.cpu().numpy(), grad_h_tensor.cpu().numpy()

    def local_train(self, round_i, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats = self.worker.local_train(self.train_dataloader, self.another_train_dataloader,round_i,**kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats

    def local_train_prox(self, round_i, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats = self.worker.local_train(self.train_dataloader, self.another_train_dataloader,round_i,**kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats

    def local_train_scaffold(self, round_i, global_c, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats, delta_c, local_c_new = self.worker.local_train(self.train_dataloader, self.another_train_dataloader,round_i,global_c,self.local_c, **kwargs)
        self.local_c = local_c_new
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats, delta_c


    def local_train_dyn(self, round_i, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats, local_alpha_new = self.worker.local_train(self.train_dataloader, self.another_train_dataloader,round_i,self.local_alpha, **kwargs)
        self.local_alpha = local_alpha_new
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats


    def local_train_etf(self, round_i, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats = self.worker.local_train(self.train_dataloader, self.another_train_dataloader,round_i, **kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats


    def local_train_uv(self, round_i, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats= self.worker.local_train(self.train_dataloader, self.another_train_dataloader,round_i, **kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats

    def local_train_smoo(self, round_i, global_w, global_s, **kwargs):
        bytes_w = self.worker.model_bytes
        begin_time = time.time()

        # 1. Before training: move the current client's dual variable to the GPU.
        device = next(self.worker.model.parameters()).device
        local_lambda_gpu = [l.to(device) for l in self.local_lambda]
        local_mu_gpu = [m.to(device) for m in self.local_mu]

        # 2. Train with the GPU copy of the variable.
        local_solution, worker_stats, tilde_s, local_lambda_new, local_mu_new = self.worker.local_train(
            self.train_dataloader, self.another_train_dataloader, round_i,
            global_w, global_s, local_lambda_gpu, local_mu_gpu, **kwargs
        )

        # 3. After training: immediately move the updated variable back to CPU storage to free GPU memory.
        self.local_lambda = [l.cpu().detach() for l in local_lambda_new]
        self.local_mu = [m.cpu().detach() for m in local_mu_new]

        # 4. Optional: force cleanup of fragmented GPU memory.
        torch.cuda.empty_cache()

        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time - begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats, tilde_s

    # Add this at the end of the Client class in client.py:
    def local_train_gf(self, round_i, global_perturbed_model, c_interp, **kwargs):
        """Solves local optimization problem for FedGF"""
        bytes_w = self.worker.model_bytes
        begin_time = time.time()

        # Pass global_perturbed_model and c_interp to the worker.
        local_solution, worker_stats = self.worker.local_train(
            self.train_dataloader, self.another_train_dataloader, round_i,
            global_perturbed_model=global_perturbed_model, c_interp=c_interp, **kwargs
        )

        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time - begin_time, 2)}
        stats.update(worker_stats)

        # local_solution is a 1D Tensor; the return format matches BaseTrainer.aggregate.
        return (len(self.train_data), local_solution), stats

    def local_train_fedsam(self, round_i, **kwargs):
        """Solves local optimization problem using FedSAM

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """
        bytes_w = self.worker.model_bytes
        begin_time = time.time()

        # Call worker.local_train, which contains the SAM logic implemented in worker.py.
        local_solution, worker_stats = self.worker.local_train(
            self.train_dataloader, self.another_train_dataloader, round_i, **kwargs)

        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time - begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats


    def local_test(self, use_eval_data=True):
        """Test current model on local eval data

        Returns:
            1. tot_correct: total # correct predictions
            2. test_samples: int
        """
        if use_eval_data:
            dataloader, dataset = self.test_dataloader, self.test_data
            another_dataloader, another_dataset = self.another_test_dataloader, self.another_test_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data
            another_dataloader, another_dataset = self.another_train_dataloader, self.another_train_data

        tot_correct, loss = self.worker.local_test(dataloader,another_dataloader)

        return tot_correct, len(dataset), loss



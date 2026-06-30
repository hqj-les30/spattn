import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import copy
from dataclasses import dataclass
# from models import set_metrics
import logging
# from evaluation import generate_and_save_image_grid
import os
import abc
from typing import Type
from models import set_model_fn
from utils import diff_state_dict

def set_client(
    method: str      
):
    return Client


@dataclass
class ClientSolver:
    batch_size: int
    local_epochs: int
    lossfunc_tar: nn.Module
    modelfunc_tar: Type[nn.Module]
    save_path: str
    normalization: str = None

    @classmethod
    def from_args(cls, args):
        params = {
            'batch_size': getattr(args, 'batch_size', 256),
            'local_epochs': getattr(args, 'local_epochs', 2),
            'lossfunc_tar': nn.CrossEntropyLoss(),
            'modelfunc_tar': set_model_fn(args.dataset),
            'save_path': getattr(args, 'path', './result/')
        }
        return cls(**params)

class BaseClient(abc.ABC):
    def __init__(self, ID, ds, solver: ClientSolver):
        self.ID = ID

        self.ds = ds
        self.num_samples = len(ds)
        self.solver = solver
        self.loader = DataLoader(self.ds, batch_size=self.solver.batch_size, shuffle=True)

        # self.server = None
        # self.model = self.solver.modelfunc_tar()

        self.logger = logging.getLogger('Client')

    def attach_to_server(self, server):
        # self.server = server
        self.data_shape = server.data_shape
        self.num_classes = server.num_classes
        self.tar_model_fn = server.solver.tar_model_fn

class Client(BaseClient):
    def __init__(self, ID, ds, solver: ClientSolver):
        super().__init__(ID, ds, solver)
    
    def local_update(self, device, global_parameter, lr=1e-3, return_grad=True):
        local_parameter = copy.deepcopy(global_parameter)
        model = self.solver.modelfunc_tar(n_class=self.num_classes)
        model.load_state_dict(local_parameter)
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        # self.logger.info(f"{self.solver.normalization} normalization used in local update.")

        train_loss_sum = 0.0
        total_seen = 0
        for e in range(self.solver.local_epochs):
            for x, y in self.loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = self.solver.lossfunc_tar(pred, y)
                if self.solver.normalization == 'fedprox':
                    loss_norm = 0.
                    for key, w in model.named_parameters():
                        w_t = global_parameter[key].to(device)
                        loss_norm += torch.norm(w - w_t, 2)
                    loss += 5e-4 * loss_norm
                # combo-attention (WT2 scalar enc-input): accumulate training loss for
                # free (no extra forward) so the Q-net input has a per-client loss signal.
                train_loss_sum += loss.detach().item() * x.shape[0]
                total_seen += x.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.to('cpu')
        model.eval()
        updated_parameter = model.state_dict()
        if return_grad:
            param = diff_state_dict(local_parameter, updated_parameter)
        else:
            param = updated_parameter

        return {
            'model_params': param,
            'num_samples': self.num_samples,
            'id': self.ID,
            'train_loss': train_loss_sum / total_seen if total_seen > 0 else 0.0
        }
    
    @torch.no_grad()
    def local_stats(self, device, global_parameter, local_parameter):
        model = self.solver.modelfunc_tar(n_class=self.num_classes)
        model.load_state_dict(local_parameter)
        model.to(device)
        model.eval()
        local_loss = 0.
        for x, y in self.loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = self.solver.lossfunc_tar(out, y)
            local_loss += loss.item() * x.shape[0]

        model.to('cpu')

        delta_parameter = diff_state_dict(global_parameter, local_parameter)
        gradient_norm = 0.
        for key, w in delta_parameter.items():
            gradient_norm += torch.norm(w.float(), 2)**2
        gradient_norm += torch.sqrt(gradient_norm).item()

        return {
            'local_loss': local_loss / self.num_samples,
            'gradient_norm': gradient_norm
        }
            # compute other metrics if needed


    def gradient_estimate(self, device, global_parameter):
        local_parameter = copy.deepcopy(global_parameter)
        model = self.solver.modelfunc_tar(n_class=self.num_classes)
        model.load_state_dict(local_parameter)
        model.to(device)
        model.train()

        avg_loss = 0.
        total_samples = 0   
        grad_list = []
        for x, y in self.loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = self.solver.lossfunc_tar(out, y)
            grad = torch.autograd.grad(loss, model.parameters())
            grad_list.append(grad)
            avg_loss += loss.item()
            total_samples += x.shape[0]
        
        avg_grad = []
        for i in range(len(grad_list[0])):
            layer_grads = torch.stack([g[i] for g in grad_list])
            avg_grad.append(torch.mean(layer_grads, dim=0).cpu())

        model.to('cpu')

        return {
            'gradients': avg_grad,
            'avg_loss': avg_loss / total_samples,
            'id': self.ID
        }

        
            


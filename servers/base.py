import copy
import os
import math
import torch
import numpy as np
import logging
import json
import abc
from dataclasses import dataclass

import torch.utils
import torch.utils.data
from utils import setup_logger, parse_devices, client_task_wrapper, task_initializer, generate_zero_weights
from models import set_model_fn
from flops import full_param_numel, forward_macs_for_model, client_training_flops, uplink_full_bytes
from data_wrapper import get_dataset_preparation_fn
from clients import set_client, ClientSolver, BaseClient
from typing import Type, Callable, List, Dict, Iterable, Any
import torch.multiprocessing as mp
import random
from client_sampler import set_client_sampler
import evaluation as eva

mp.set_sharing_strategy('file_system')

_SERVER_REGISTRY: Dict[str, Type] = {}

def register_server(name: str):
    def decorator(cls):
        _SERVER_REGISTRY[name] = cls
        return cls
    return decorator

def set_server(method: str):
    # FLOW is the only method shipped in this repo (registered as 'Ours'); accept either name.
    if method.lower() == 'flow':
        method = 'Ours'
    cls = _SERVER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(f"Unknown method: {method}. Available: {list(_SERVER_REGISTRY.keys())}")
    return cls

@dataclass
class ServerSolver:
    gpus: List = None
    global_epoch: int = 200
    n_clients: int = 50
    evaluation_interval: int = 1
    cluster_size: int = 20
    selection_size: int = 5
    insert: int = 2
    tar_model_fn: Type[torch.nn.Module] = None
    dataset_prepare_fn: Callable = None
    cluster_sampler: str = None
    lr: float = 1e-3
    metric: str = 'f1'
    recall: int = 3
    workers_per_gpu: int = 2
    log_path = None
    args = None

    @classmethod
    def from_args(cls, args):
        default_cls = cls()
        default_cls.gpus = parse_devices(getattr(args, 'gpu_ids', None))
        default_cls.n_clients = getattr(args, 'n_clients', 50)
        default_cls.global_epoch = getattr(args, 'total_epochs', 200)
        default_cls.selection_size = getattr(args, 'selection_size', 10)
        default_cls.tar_model_fn = set_model_fn(args.dataset)
        default_cls.dataset_prepare_fn = get_dataset_preparation_fn(
            setting=args.setting,
            alpha=getattr(args, 'alpha', 0.1),
            num_shards=getattr(args, 'num_shards', 2)
        )
        default_cls.cluster_sampler = set_client_sampler(getattr(args, 'cluster_sampler', 'randcls'))
        default_cls.cluster_size = getattr(args, 'cluster_size', 20)
        default_cls.log_path = getattr(args, 'path')/ 'log.log'
        default_cls.args = args
        default_cls.insert = getattr(args, 'insert', 0)
        default_cls.metric = getattr(args, 'metric', 'f1')
        default_cls.recall = getattr(args, 'recall', 3)
        default_cls.workers_per_gpu = getattr(args, 'workers_per_gpu', 2)
        return default_cls

class BaseServer(abc.ABC):
    def __init__(self, solver: ServerSolver):
        self.solver = solver
        setup_logger(self.solver.log_path)
        self.logger = logging.getLogger(solver.args.method)
        self.client_vec: List[BaseClient] = []
        local_ds_vec, mode_vec, test_dataset, data_preparation_info = self.solver.dataset_prepare_fn(
            n_clients=self.solver.n_clients,
            dataset=self.solver.args.dataset,
        )
        self.logger.info(f"{len(local_ds_vec)} clients' datasets prepared.")
        self.logger.info(f"{[len(ds) for ds in local_ds_vec]} samples per client.")
        self.num_classes = data_preparation_info['num_classes']
        self.data_shape = data_preparation_info['data_shape']
        base_path = getattr(self.solver.args, 'path')
        file_path = os.path.join(base_path, 'client_datasets_indexes.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_preparation_info['client_datasets_indexes'], f, ensure_ascii=False, indent=4)
        self.client_solver = ClientSolver.from_args(self.solver.args)
        client_fn = set_client(self.solver.args.method)
        for i, ds in enumerate(local_ds_vec):
            self.client_vec.append(
                client_fn(ID=i, ds=ds, solver=self.client_solver)
            )
        self.num_clients = self.solver.n_clients
        self.test_ds = test_dataset
        self.client_sampler = self.solver.cluster_sampler(self.solver.n_clients, mode_vec, self.solver.cluster_size, self.solver.global_epoch)
        self.num_gpus = len(self.solver.gpus)
        self.manager = None
        self.pool = None
        self.glob_model = self.solver.tar_model_fn(n_class=self.num_classes)
        # total float params of the full model (used for communication cost)
        self.full_param_size = full_param_numel(self.glob_model.state_dict())

    def _aggregate_parameters(self, model_dicts: Iterable[Dict], weights: Iterable, model_fn: Callable = None, w0: Dict = None):
        total_weights = sum(weights)
        weights = [w / total_weights for w in weights]
        with torch.no_grad():
            if w0 is not None:
                new_weights = copy.deepcopy(w0)
            else:
                new_weights = copy.deepcopy(generate_zero_weights(model_fn))
            for k, model_dict in enumerate(model_dicts):
                for key in new_weights.keys():
                    if key.endswith('num_batches_tracked'):
                        if k == 0:
                            new_weights[key] = model_dict[key].clone()
                    else:
                        new_weights[key].add_(model_dict[key], alpha=weights[k])
            return new_weights

    def save_details_to_jsonl(self, data_dict: Dict[str, Any], mode: str = 'a'):
        try:
            base_path = getattr(self.solver.args, 'path')
            file_path = os.path.join(base_path, 'details.jsonl')
            json_string = json.dumps(data_dict, ensure_ascii=False)
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(json_string + '\n')
        except AttributeError:
            self.logger.error("错误: 无法找到 'self.solver.args.path' 属性。")
        except TypeError as e:
            self.logger.error(f"错误: 字典无法序列化为 JSON。错误信息: {e}")
        except IOError as e:
            self.logger.error(f"错误: 写入文件时发生 I/O 错误。错误信息: {e}")
        except Exception as e:
            self.logger.error(f"发生未知错误: {e}")

    def run(self):
        if self.num_gpus >= 1:
            mp.set_start_method('spawn', force=True)
            self._init_pool()
        self.logger.info("Start Running, results in " + str(self.solver.log_path))

    def _init_pool(self):
        self.manager = mp.Manager()
        gpu_queue = self.manager.Queue()
        wpg = self.solver.workers_per_gpu
        for gpu in self.solver.gpus:
            for _ in range(wpg):
                gpu_queue.put(gpu)
        self.pool = mp.Pool(
            processes=self.num_gpus * wpg,
            initializer=task_initializer,
            initargs=(gpu_queue, self.solver.log_path)
        )

    def _run_local_update(self, selected_clients: List[int]):
        task_args = [
            (
                self.client_vec[i],
                'update_tar',
                {
                    'global_parameter': self.glob_model.state_dict(),
                    'lr': self.solver.lr,
                    'return_grad': False
                }
            )
            for i in selected_clients
        ]
        if self.num_gpus >= 1:
            results = self.pool.starmap(client_task_wrapper, task_args)
        else:
            results = [client_task_wrapper(*args) for args in task_args]
        return results

    def _close_pool(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
        if self.manager:
            self.manager = None

    def _checkpoint(self, models: List, **kwargs) -> Dict[Any, Any]:
        checkpoint = {**kwargs}
        for i, model in enumerate(models):
            if not hasattr(model, 'state_dict'):
                self.logger.warning(f"列表中的第 {i} 个对象没有 'state_dict' 方法，将被跳过。")
                continue
            checkpoint[i] = model.state_dict()
        return checkpoint

    def _evaluate_global_model(self):
        test_loss, test_acc, per_class_acc = eva.evaluate_classfication(
            model=self.glob_model,
            dataset=self.test_ds,
            batch_size=512,
            loss_fn=torch.nn.CrossEntropyLoss(),
            gpu=self.solver.gpus[random.randint(0, self.num_gpus - 1)] if self.num_gpus > 0 else 'cpu'
        )
        self.logger.info(f"Evaluation on Test Dataset -- Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'per_class_accuracy': per_class_acc
        }

    def _update_global_model(self, model_dicts: Iterable[Dict], weights: Iterable):
        total_weights = sum(weights)
        weights = [w / total_weights for w in weights]
        new_weights = self._aggregate_parameters(model_dicts, weights, self.solver.tar_model_fn)
        self.glob_model.load_state_dict(new_weights)

    def compute_efficiency_stats(self) -> Dict[str, Any]:
        """Shared (method-agnostic) efficiency stats.

        Client FLOPs are computed analytically from the model architecture and each
        client's dataset size (averaged over the client population). Communication
        counts only model-parameter bytes, uplink only: here the aggregation uplink
        (selected clients uploading full models). FLOW adds the projected-vector
        selection uplink + server RL FLOPs in its override.
        """
        batch_size = self.client_solver.batch_size
        local_epochs = self.client_solver.local_epochs
        cnn_macs = forward_macs_for_model(
            self.solver.tar_model_fn, self.data_shape, n_classes=self.num_classes
        )
        per_client_flops = []
        for c in self.client_vec:
            nb = math.ceil(c.num_samples / batch_size) if c.num_samples > 0 else 0
            per_client_flops.append(client_training_flops(cnn_macs, nb, local_epochs))
        client_avg_flops = int(sum(per_client_flops) / len(per_client_flops)) if per_client_flops else 0

        selection_size = self.solver.selection_size
        cluster_size = self.solver.cluster_size
        return {
            'client_avg_flops': client_avg_flops,
            'client_cnn_forward_macs': int(cnn_macs),
            'client_batch_size': batch_size,
            'client_local_epochs': local_epochs,
            'full_param_size': int(self.full_param_size),
            'n_clients': self.num_clients,
            'cluster_size': cluster_size,
            'selection_size': selection_size,
            'uplink_full_bytes': uplink_full_bytes(selection_size, self.full_param_size),
        }

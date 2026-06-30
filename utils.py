import numpy as np
import torch
from datetime import datetime
from typing import List, Optional, Type, Dict
from functools import lru_cache
import logging
import sys
import random
import os
from math import comb
os.environ["OMP_NUM_THREADS"] = "4"
def parse_devices(device_str: Optional[str]) -> List[torch.device]:
    """
    从命令行字符串解析设备列表
    - 解析逗号分隔的 GPU ID (e.g., '0,1,2')
    - 处理特殊值 'cpu'
    - 移除了所有对 cuda.is_available() 和 cuda.device_count() 的检查。

    Args:
        device_str: 从命令行传入的设备字符串。

    Returns:
        一个包含 torch.device 对象的列表。

    Raises:
        ValueError: 如果 GPU ID 无效 (例如非整数或负数)。
    """
    # 1. 处理默认情况 -> 直接默认为 cuda:0
    if not device_str:
        return [torch.device("cuda:0")]

    # 2. 解析 GPU IDs (不再检查 CUDA 是否可用或设备数量)
    gpu_ids_str = device_str.split(',')
    devices = []

    for id_str in gpu_ids_str:
        id_str = id_str.strip()
        if not id_str:
            continue

        try:
            gpu_id = int(id_str)
        except ValueError:
            raise ValueError(f"无效的 GPU ID '{id_str}'。ID 必须是整数。")

        if gpu_id < 0:
            raise ValueError(f"GPU ID {gpu_id} 不能为负数。")
        
        devices.append(torch.device(f'cuda:{gpu_id}'))

    if not devices:
        raise ValueError("设备字符串中未包含任何有效的设备 ID。")

    return devices

def setup_logger(log_path, level=logging.INFO):
    """
    配置 root logger。

    这个函数是幂等的，即多次调用不会重复添加 handler。

    Args:
        log_path (str): 日志文件的完整路径。
        level (int): 设置的日志级别，例如 logging.INFO, logging.DEBUG。
    """
    # 1. 获取 root logger
    # root logger 是所有 logger 的祖先，配置它会影响到整个应用
    logger = logging.getLogger()
    
    # 2. 设置日志级别
    # 这是总开关，只有高于这个级别的日志才会被处理
    logger.setLevel(level)

    # 3. 清理已有的 handlers，防止重复记录
    # 如果在其他地方已经配置过 logger，这一步可以确保我们的配置生效
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. 创建一个统一的 Formatter
    # 定义日志的输出格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 5. 创建 FileHandler，用于写入日志文件
    # mode='a' 表示追加模式，encoding='utf-8' 防止中文乱码
    try:
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"错误：无法设置文件日志处理器 at {log_path}: {e}")


    # 6. 创建 StreamHandler，用于输出到控制台
    # sys.stdout 是标准输出流
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def set_seed(seed: int = 42):
    """
    设置Python、NumPy、PyTorch等库的随机种子，以确保实验结果可复现。
    
    参数:
        seed (int): 随机数种子（默认值为42）
    """
    
    # Python内置随机数模块
    random.seed(seed)
    
    # NumPy随机数模块
    np.random.seed(seed)
    
    # 环境变量（影响某些hash随机性）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 如果使用了 PyTorch，则一并设置
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时设置所有卡的seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # 没装PyTorch也不报错

worker_gpu = None
def task_initializer(gpu_queue, log_path):
    global worker_gpu
    try:
        # 从队列中获取一个唯一的GPU ID
        worker_gpu = gpu_queue.get()
        # 关键：为当前进程设置CUDA设备
        torch.cuda.set_device(worker_gpu)
        # print(f"[Worker Initializer PID: {os.getpid()}] successfully set device to GPU {worker_gpu_id}.")
        setup_logger(log_path)
    except Exception as e:
        print(f"Error in worker initializer: {e}")

def client_task_wrapper(client, task: str, param_dict: Dict = {}):
    global worker_gpu
    if worker_gpu is None:
        raise RuntimeError("Worker process has not been initialized with a GPU ID.")
    
    if task == 'local':
        updated_model = client.run_local_learning(worker_gpu, **param_dict)
        updated_model.update(client.local_stats(worker_gpu, param_dict['global_parameter'], updated_model['model_params']))
        return updated_model
    if task == 'gradient_estimate':
        client_result = client.gradient_estimate(worker_gpu, **param_dict)
        return client_result
    if task == 'update_gen':
        updated_model_params = []
        for i in range(client.num_clusters):
            updated_model = client.update_gen_model(i, worker_gpu, **param_dict)
            updated_model_params.append(updated_model)
        return updated_model_params
    if task == 'update_tar':
        client_result = client.local_update(worker_gpu, **param_dict)
        client_result.update(client.local_stats(worker_gpu, param_dict['global_parameter'], client_result['model_params']))
        return client_result
            
@lru_cache(maxsize=2)
def generate_zero_weights(model_fn: Type[torch.nn.Module]):
    weights = {}
    for key, val in model_fn().state_dict().items():
        weights[key] = torch.zeros(size=val.shape, dtype=torch.float32)

    return weights

def diff_state_dict(state_dict_a: Dict, state_dict_b: Dict):
    """
    计算两个state_dict之间的差值(b - a)返回一个新的state_dict。
    
    参数:
        state_dict_a: 第一个state_dict (通常是旧参数)
        state_dict_b: 第二个state_dict (通常是新参数)
    返回:
        diff_dict: 每个参数键对应 (state_dict_b[key] - state_dict_a[key]) 的张量
    """
    diff_dict = {}
    for key in state_dict_a.keys():
        if key not in state_dict_b:
            raise KeyError(f"Key '{key}' not found in second state_dict.")
        diff_dict[key] = state_dict_b[key] - state_dict_a[key]
    return diff_dict

def parse_modelfunc(model_fn: Type[torch.nn.Module], *args, **kwargs):
    """
    Factory function that returns a size and two functions:
      - size: total number of parameters
      - flatten(model) -> 1D tensor
      - unflatten(flat_tensor) -> state_dict
    The structure (shapes, slices) is precomputed once based on a template model.

    Args:
        model_fn: a callable that returns an nn.Module instance
        *args, **kwargs: arguments passed to model_fn when creating the template model

    Returns:
        flatten_fn, unflatten_fn
    """
    # Create a template model and record its parameter info
    model = model_fn(*args, **kwargs)
    sd = model.state_dict()
    if hasattr(model, "reserved"):
        reserved_modules = model.reserved
        reserved_keys = [
            k for k in sd.keys()
            if any(k.startswith(m + ".") for m in reserved_modules)
        ]
    else:
        reserved_keys = list(sd.keys())

    keys, shapes, slices = [], [], []
    pointer = 0

    for k in reserved_keys:
        v = sd[k]
        numel = v.numel()
        keys.append(k)
        shapes.append(v.shape)
        slices.append(slice(pointer, pointer + numel))
        pointer += numel

    total_size = pointer

    # Define flatten function
    def flatten_fn(sd):
        """Flatten model parameters into a 1D tensor."""
        return torch.cat([sd[k].reshape(-1) for k in keys])

    # Define unflatten function
    def unflatten_fn(flat_tensor):
        """Reconstruct a state_dict from a flat tensor."""
        new_sd = {}
        for k, shape, sl in zip(keys, shapes, slices):
            new_sd[k] = flat_tensor[sl].view(shape)
        return new_sd

    return total_size, flatten_fn, unflatten_fn

def linear_decay(max_value, min_value, k, t):
    """
    Compute a linearly decayed value with slope k, clipped at min_value.

    Args:
        max_value (float): starting value
        min_value (float): minimum value
        k (float): decay slope per step
        t (int): current step

    Returns:
        float: decayed value at step t
    """
    value = max_value - k * t
    return max(value, min_value)

def exponential_decay(max_value, min_value, k, t):
    """
    Compute an exponentially decayed value, clipped at min_value.

    Args:
        max_value (float): starting value
        min_value (float): minimum value
        decay_rate (float): decay rate per step
        t (int): current step

    Returns:
        float: decayed value at step t
    """
    value = min_value + (max_value - min_value) * np.exp(-k * t)
    return value

from sklearn.metrics import f1_score

def compute_output_dist(model, dataset, device, num_classes: int = 10):
    """
    Compute the output distribution of a PyTorch model on a dataset.

    Args:
        model (torch.nn.Module): the trained model
        dataloader (torch.utils.data.DataLoader): data loader for evaluation
        device (str): 'cpu' or 'cuda'
        num_classes (int): number of classes

    Returns:
        torch.Tensor: output distribution over classes
    """
    model.eval()
    model.to(device)
    all_preds = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            # Predicted class (argmax over logits)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds)

    counts = torch.bincount(all_preds, minlength=num_classes).float()
    pred_dist = counts / counts.sum().clamp(min=1.0)
    model.to('cpu')
    return pred_dist

def compute_macro_f1(model, dataset, device='cpu', num_classes: int = 10):
    """
    Compute the macro F1 score of a PyTorch model on a dataset.

    Args:
        model (torch.nn.Module): the trained model
        dataloader (torch.utils.data.DataLoader): data loader for evaluation
        device (str): 'cpu' or 'cuda'

    Returns:
        float: macro F1 score
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            # Predicted class (argmax over logits)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute macro F1 score
    n_class = num_classes
    macro_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average=None, labels=list(range(n_class)), zero_division=0)
    model.to('cpu')

    counts = torch.bincount(all_preds, minlength=n_class).float()
    pred_dist = counts / counts.sum().clamp(min=1.0)
    return torch.from_numpy(macro_f1), pred_dist

def compute_accuracy(model, dataset, device='cpu', num_classes: int = 10):
    """
    Compute per-class accuracy of a PyTorch model on a dataset.

    Args:
        model (torch.nn.Module): trained model
        dataset (torch.utils.data.Dataset): evaluation dataset
        num_classes (int): number of classes
        device (str): 'cpu' or 'cuda'

    Returns:
        acc (torch.Tensor): shape (num_classes,), per-class accuracy on CPU
        counts (torch.Tensor): shape (num_classes,), number of samples per class on CPU
    """
    model.eval()
    model.to(device)

    correct = torch.zeros(num_classes, device=device)
    total = torch.zeros(num_classes, device=device)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            for c in range(num_classes):
                mask = (labels == c)
                total[c] += mask.sum()
                correct[c] += (preds[mask] == c).sum()

    acc = torch.zeros(num_classes, device=device)
    nonzero_mask = total > 0
    acc[nonzero_mask] = correct[nonzero_mask] / total[nonzero_mask]

    model.to('cpu')
    return acc.cpu()

@lru_cache(maxsize=512)
def action2selection(action: int, N: int, K: int) -> List[int]:
        """
        Converts an integer action index (rank) to an unordered selection (Combination) 
        using the Combinatorial Number System (Combinadics).

        Args:
            action: Integer rank [0, C(N, K) - 1].

        Returns:
            A list of selected client indices (Combination, in descending order).

        Raises:
            ValueError: If action is out of bounds.
        """
            
        # available_clients in the original code is not directly used for Combinadics, 
        # but we must track the current action value and the current search bound (N).
        # We use a temporary variable 'current_N' to track the decreasing upper bound c_i.
        current_N = N 
        result = [0] * K

        # i corresponds to the index position in the Combinadics formula: C(c_i, i).
        # i ranges from K down to 1 (number of elements still to select).
        for i in range(K, 0, -1):
            
            # f (Fictitious/Combinadics Base): We search for c_i such that C(c_i, i) <= action.
            # We use linear search to find the largest c_i.
            
            # The smallest possible value for c_i (pos) is i - 1.
            # The largest possible value for c_i (pos) is current_N - (i - 1)
            
            low = i - 1  
            high = current_N - 1
            pos = low # pos is the index c_i we are looking for

            # Linear search for the largest pos (c_i) such that math.comb(pos, i) <= action
            for c_i_candidate in range(low, high + 1):
                comb_val = comb(c_i_candidate, i)
                if comb_val <= action:
                    pos = c_i_candidate
                else:
                    break
            
            # result index K-i corresponds to the c_i value.
            result[K - i] = pos
            
            # Update action: action = action - C(c_i, i)
            action -= comb(pos, i)
            
            # Update current_N for the next iteration: c_{i-1} must be less than c_i.
            current_N = pos

        # Note: The original logic for 'available_clients.pop(pos)' is replaced by direct calculation.
        # This implementation returns the selected indices in DESCENDING order.
        return result

@torch.no_grad()
def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute cosine similarity between two 1-d tensors.

    Args:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Cosine similarity values.
    """
    a_norm = a / (a.norm() + eps)
    b_norm = b / (b.norm() + eps)
    return torch.dot(a_norm, b_norm).item()

@torch.no_grad()
def project_X_onto_A_column_space(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # Solve min ||A C - X^T||
    C = torch.linalg.lstsq(A, X.T).solution  # (k, n)
    return (A @ C).T

@torch.no_grad()
def entropy(p: torch.Tensor, dim=-1, eps=1e-12):
    """
    Compute entropy along given dimension.
    p: probability tensor, should sum to 1 along dim
    """
    p = p / p.sum(dim=dim, keepdim=True)
    p = p.clamp(min=eps)   # 避免 log(0)
    return -torch.sum(p * torch.log(p), dim=dim)/torch.log(torch.tensor(p.shape[dim], dtype=p.dtype))
@torch.no_grad()
def emd_1d(p: torch.Tensor, q: torch.Tensor, dim=-1):
    """
    Compute 1D Earth Mover's Distance (Wasserstein-1) between p and q.

    Args:
        p, q: normalized probability tensors with same shape
    """
    # p = p / (p.sum() + 1e-8)
    # q = q / (q.sum() + 1e-8)
    return 0.5 * torch.abs(p - q).sum()

class MetaInfoQueue:
    def __init__(self, n=None, queue_length=3, dtype=torch.float32):
        """
        Fixed-length FIFO queue implemented using a single (queue_length x n) tensor.

        Args:
            n (int): Dimension of one-hot vectors.
            queue_length (int): Queue length.
            device (str): Device to store the queue tensor.
            dtype: Tensor dtype.
        """
        self.n = n
        self.queue_length = queue_length
        self.dtype = dtype

        # Initialize a fixed-size queue tensor
        if self.n is None:
            self.buffer = None
        else:
            self.buffer = torch.zeros(queue_length, n, dtype=dtype)

    def push(self, x: torch.Tensor):
        """
        Insert a one-hot vector into the queue and remove the oldest one.

        Args:
            indices (list[int]): Indices to set to 1 in the new one-hot vector.
        """
        if self.n is None and self.buffer is None:
            self.n = x.shape[0]
            self.buffer = torch.zeros(self.queue_length, self.n, dtype=self.dtype)
            for i in range(self.queue_length):
                self.buffer[i, :].copy_(x)


        if self.queue_length > 1:
            self.buffer = torch.cat([self.buffer[1:], torch.zeros(1, self.n, dtype=self.dtype)], dim=0)
        elif self.queue_length == 1:
            self.buffer.zero_()
        self.buffer[-1, :].copy_(x)

        return self.buffer.cpu()


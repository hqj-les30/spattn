import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict
from torchvision.utils import save_image, make_grid

def evaluate_classfication(
    model: nn.Module,
    dataset: Dataset,
    gpu: torch.device,
    loss_fn: nn.Module,
    batch_size: int = 512
) -> Tuple[float, float, List]:
    """
    计算模型在给定数据集上的平均损失、总体准确率和每一类的准确率。

    此函数假设模型输出的是原始 logits，标签是类别索引。

    Args:
        model (nn.Module): 需要评估的 PyTorch 模型。
        dataset (Dataset): 包含评估数据的 PyTorch 数据集。
        device (torch.device): 运行计算的设备 (例如 'cuda' 或 'cpu')。
        loss_fn (nn.Module): 用于计算损失的损失函数 (例如 nn.CrossEntropyLoss)。
        batch_size (int, optional): DataLoader 使用的批次大小。默认为 64。

    Returns:
        Tuple[float, float, np.ndarray]: 一个包含 (平均损失, 总体准确率, 每类准确率) 的元组。
                                         - 平均损失 (float): 整个数据集上的平均损失。
                                         - 总体准确率 (float): 0.0 到 1.0 之间的浮点数。
                                         - 每类准确率 (np.ndarray): 一个 NumPy 数组，其索引对应类别索引，
                                           值为该类的准确率。如果某个类在数据集中不存在，
                                           其准确率将为 0。
    """
    # 1. 创建 DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if gpu:
        device = gpu
    else:
        from utils import worker_gpu
        device = worker_gpu

    # 2. 移动模型到指定设备并设置为评估模式
    model.to(device)
    # original_mode = model.training
    model.eval()

    # 3. 初始化用于追踪的变量
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 使用 defaultdict 来方便地统计每个类别
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    try:
        # 4. 在不计算梯度的上下文中进行评估
        with torch.no_grad():
            for inputs, labels in loader:
                # a. 将数据移动到指定设备
                inputs = inputs.to(device)
                labels = labels.to(device)

                # b. 前向传播
                outputs = model(inputs)

                # c. 计算并累加损失
                loss = loss_fn(outputs, labels)
                num_samples_in_batch = inputs.size(0)
                total_loss += loss.item() * num_samples_in_batch

                # d. 计算总体准确率
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += num_samples_in_batch
                
                # e. 计算每一类的准确率
                # 将预测结果和真实标签都移动到CPU上进行统计
                correct_preds = (predicted == labels).cpu()
                labels_cpu = labels.cpu()
                
                for i in range(len(labels_cpu)):
                    label = labels_cpu[i].item()
                    class_total[label] += 1
                    if correct_preds[i]:
                        class_correct[label] += 1
    finally:
        # 5. 恢复模型到原始的训练/评估模式
        # model.train(original_mode)
        pass

    # 6. 计算最终的各项指标
    average_loss = total_loss / total_samples
    overall_accuracy = total_correct / total_samples

    # 确定类别总数，以创建完整的每类准确率数组
    if not class_total:
        # 如果数据集为空，则返回空结果
        return average_loss, overall_accuracy, np.array([])
        
    num_classes = max(class_total.keys()) + 1
    per_class_accuracy = np.zeros(num_classes)
    
    for c, total in class_total.items():
        if total > 0:
            per_class_accuracy[c] = class_correct[c] / total

    per_class_accuracy = np.round(1000 * per_class_accuracy) / 10

    model.to('cpu')

    return round(average_loss,4), round(100*overall_accuracy, 3), per_class_accuracy.tolist()

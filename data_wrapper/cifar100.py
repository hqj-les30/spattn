import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from pathlib import Path
from .base import IndexedDataset, FullDataset, DatasetConfig, register_dataset, path_to_data

class Cifar100Local(IndexedDataset):
    source_data = datasets.CIFAR100(root=path_to_data, train=True, download=False,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))

class Cifar100Dev(FullDataset):
    source_data = datasets.CIFAR100(root=path_to_data, train=False, download=False,
        transform=transforms.Compose([transforms.ToTensor()]))

def create_cifar100_indices():
    if (path_to_data / 'cifar100_indices.npy').exists():
        return
    train_data = datasets.CIFAR100(root=path_to_data, train=True, download=False,
        transform=transforms.ToTensor())
    indices = [[] for _ in range(100)]
    for i, (x, y) in enumerate(train_data):
        indices[y].append(i)
    np.save(path_to_data / 'cifar100_indices.npy', indices)

register_dataset(DatasetConfig(
    name='cifar100', n_class=100, data_shape=(3, 32, 32),
    local_cls=Cifar100Local, dev_cls=Cifar100Dev,
    create_fn=create_cifar100_indices
))

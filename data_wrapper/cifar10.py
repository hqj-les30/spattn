import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from pathlib import Path
from .base import IndexedDataset, FullDataset, DatasetConfig, register_dataset, path_to_data

class Cifar10Local(IndexedDataset):
    source_data = datasets.CIFAR10(root=path_to_data, train=True, download=False,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))

class Cifar10Dev(FullDataset):
    source_data = datasets.CIFAR10(root=path_to_data, train=False, download=False,
        transform=transforms.Compose([transforms.ToTensor()]))

def create_cifar10_indices():
    abspath = Path(path_to_data).expanduser()
    if abspath.joinpath('cifar10_indices.npy').exists():
        return
    train_data = datasets.CIFAR10(root=path_to_data, train=True, download=False,
        transform=transforms.ToTensor())
    indices = [[] for _ in range(10)]
    for i, (x, y) in enumerate(train_data):
        indices[y].append(i)
    np.save(abspath.joinpath('cifar10_indices.npy'), indices)

register_dataset(DatasetConfig(
    name='cifar10', n_class=10, data_shape=(3, 32, 32),
    local_cls=Cifar10Local, dev_cls=Cifar10Dev,
    create_fn=create_cifar10_indices
))

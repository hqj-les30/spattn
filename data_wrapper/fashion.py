import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from .base import IndexedDataset, FullDataset, DatasetConfig, register_dataset, path_to_data

class FashionLocal(IndexedDataset):
    source_data = datasets.FashionMNIST(root=path_to_data, train=True, download=False,
        transform=transforms.ToTensor())

class FashionDev(FullDataset):
    source_data = datasets.FashionMNIST(root=path_to_data, train=False, download=False,
        transform=transforms.Compose([transforms.ToTensor()]))

def create_fashion_indices():
    if (path_to_data / 'fashion_indices.npy').exists():
        return
    train_data = datasets.FashionMNIST(root=path_to_data, train=True, download=False,
        transform=transforms.ToTensor())
    indices = [[] for _ in range(10)]
    for i, (x, y) in enumerate(train_data):
        indices[y].append(i)
    np.save(path_to_data / 'fashion_indices.npy', indices)

register_dataset(DatasetConfig(
    name='fashion', n_class=10, data_shape=(1, 28, 28),
    local_cls=FashionLocal, dev_cls=FashionDev,
    create_fn=create_fashion_indices
))

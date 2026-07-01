from .base import DatasetConfig, IndexedDataset, FullDataset, get_dataset_config, path_to_data
from .preparation import get_dataset_preparation_fn, proxy_data_preparation
from .preparation import local_data_preparation, dirichlet_data_preparation

# Import dataset modules to trigger registration
from . import cifar10, fashion, har

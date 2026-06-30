from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Tuple, Type, Optional, Callable, Dict
from pathlib import Path

path_to_data = Path('~/data').expanduser()

@dataclass
class DatasetConfig:
    name: str
    n_class: int
    data_shape: Tuple[int, ...]
    local_cls: Type[Dataset]
    dev_cls: Type[Dataset]
    create_fn: Optional[Callable] = None

_REGISTRY: Dict[str, DatasetConfig] = {}

def register_dataset(config: DatasetConfig):
    _REGISTRY[config.name] = config

def get_dataset_config(name: str) -> DatasetConfig:
    config = _REGISTRY.get(name)
    if config is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_REGISTRY.keys())}")
    return config

class IndexedDataset(Dataset):
    source_data = None

    def __init__(self, indices, n_per_class=None):
        if indices is None:
            indices = list(range(len(self.__class__.source_data)))
        self.indices = indices
        self.size = len(self.indices)
        self.targets = [self.__class__.source_data.targets[i] for i in self.indices]

    def __getitem__(self, item):
        return self.__class__.source_data[self.indices[item]][0], \
               self.__class__.source_data[self.indices[item]][1]

    def __len__(self):
        return self.size

class FullDataset(Dataset):
    source_data = None

    def __init__(self):
        self.size = len(self.__class__.source_data)

    def __getitem__(self, item):
        return self.__class__.source_data[item][0], self.__class__.source_data[item][1]

    def __len__(self):
        return self.size

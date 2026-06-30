import numpy as np
from torch.utils.data import Dataset
from .base import DatasetConfig, register_dataset, path_to_data

class UCIHARDataset(Dataset):
    def __init__(self, split="train"):
        assert split in ["train", "test"]
        base = path_to_data / 'UCI HAR Dataset' / split
        self.X = np.loadtxt(base / f"X_{split}.txt").astype(np.float32)
        self.y = np.loadtxt(base / f"y_{split}.txt").astype(np.int64) - 1
        self.subject = np.loadtxt(base / f"subject_{split}.txt").astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx], "subject": self.subject[idx]}

def split_by_subject(dataset):
    from collections import defaultdict
    user_data = defaultdict(list)
    for i in range(len(dataset)):
        user_data[int(dataset.subject[i])].append(i)
    return dict(user_data)

class UCIHARLocal(Dataset):
    data = UCIHARDataset(split="train")

    def __init__(self, indices):
        if indices is None:
            indices = [i for i in range(len(UCIHARLocal.data))]
        self.indices = indices
        self.size = len(self.indices)
        self.targets = [UCIHARLocal.data.y[i] for i in self.indices]

    def __getitem__(self, item):
        sample = UCIHARLocal.data[self.indices[item]]
        return sample["x"], sample["y"]

    def __len__(self):
        return self.size

class UCIHARDev(Dataset):
    data = UCIHARDataset(split="test")

    def __init__(self):
        self.size = len(UCIHARDev.data)

    def __getitem__(self, item):
        sample = UCIHARDev.data[item]
        return sample["x"], sample["y"]

    def __len__(self):
        return self.size

register_dataset(DatasetConfig(
    name='har', n_class=6, data_shape=(561,),
    local_cls=UCIHARLocal, dev_cls=UCIHARDev,
    create_fn=None
))

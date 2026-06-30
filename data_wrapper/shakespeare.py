import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .base import DatasetConfig, register_dataset, path_to_data

SHAKESPEARE_SEQ_LEN = 80

class ShakespeareLocal(Dataset):
    inputs = None
    targets = None
    n_class = None

    @classmethod
    def _ensure_loaded(cls):
        if cls.inputs is None:
            create_shakespeare_data()

    def __init__(self, indices, n_per_class=None):
        self._ensure_loaded()
        if indices is None:
            indices = list(range(len(ShakespeareLocal.targets)))
        self.indices = indices
        self.size = len(self.indices)
        self.targets = ShakespeareLocal.targets[indices].tolist()

    def __getitem__(self, item):
        self._ensure_loaded()
        idx = self.indices[item]
        return ShakespeareLocal.inputs[idx], int(ShakespeareLocal.targets[idx])

    def __len__(self):
        return self.size

class ShakespeareDev(Dataset):
    inputs = None
    targets = None

    def __init__(self):
        ShakespeareLocal._ensure_loaded()
        self.size = len(ShakespeareDev.targets)

    def __getitem__(self, item):
        ShakespeareLocal._ensure_loaded()
        return ShakespeareDev.inputs[item], int(ShakespeareDev.targets[item])

    def __len__(self):
        return self.size

def create_shakespeare_data(seq_len=SHAKESPEARE_SEQ_LEN):
    abspath = Path(path_to_data).expanduser()
    train_path = abspath / 'shakespeare_train.pt'
    train_tgt_path = abspath / 'shakespeare_train_targets.pt'
    test_path = abspath / 'shakespeare_test.pt'
    test_tgt_path = abspath / 'shakespeare_test_targets.pt'
    vocab_path = abspath / 'shakespeare_vocab.npy'

    if train_path.exists() and ShakespeareLocal.inputs is None:
        ShakespeareLocal.inputs = torch.load(train_path, weights_only=False)
        ShakespeareLocal.targets = torch.load(train_tgt_path, weights_only=False)
        ShakespeareDev.inputs = torch.load(test_path, weights_only=False)
        ShakespeareDev.targets = torch.load(test_tgt_path, weights_only=False)
        vocab_info = np.load(vocab_path, allow_pickle=True).item()
        ShakespeareLocal.n_class = vocab_info['n_class']
        return

    if ShakespeareLocal.inputs is not None:
        return

    raw_path = abspath / 'shakespeare_raw.txt'
    if not raw_path.exists():
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading Shakespeare data from {url}...")
        urllib.request.urlretrieve(url, raw_path)

    with open(raw_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    n_class = len(chars)
    np.save(vocab_path, {'char2idx': char2idx, 'n_class': n_class})

    encoded = torch.LongTensor([char2idx[c] for c in text])

    split = int(len(encoded) * 0.9)
    train_text = encoded[:split]
    test_text = encoded[split:]

    train_windows = train_text.unfold(0, seq_len + 1, 1)
    train_inputs = train_windows[:, :-1].clone()
    train_targets = train_windows[:, -1].clone()

    test_windows = test_text.unfold(0, seq_len + 1, 1)
    test_inputs = test_windows[:, :-1].clone()
    test_targets = test_windows[:, -1].clone()

    ShakespeareLocal.inputs = train_inputs
    ShakespeareLocal.targets = train_targets
    ShakespeareLocal.n_class = n_class
    ShakespeareDev.inputs = test_inputs
    ShakespeareDev.targets = test_targets

    torch.save(train_inputs, train_path)
    torch.save(train_targets, train_tgt_path)
    torch.save(test_inputs, test_path)
    torch.save(test_targets, test_tgt_path)

register_dataset(DatasetConfig(
    name='shakespeare', n_class=None, data_shape=(SHAKESPEARE_SEQ_LEN,),
    local_cls=ShakespeareLocal, dev_cls=ShakespeareDev,
    create_fn=create_shakespeare_data
))

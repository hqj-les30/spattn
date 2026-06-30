import numpy as np
import sampling as sp
from .base import get_dataset_config, path_to_data
from .har import split_by_subject

def local_data_preparation(
        n_clients = 50,
        dataset = 'cifar10',
        p = [0.4, 0.3, 0.3],
        size_range = (500, 600),
        alpha = 0.8,
        r0 = 0.95,
        r1 = 0.98,
        num_shards = 2
):
    n_iid = round(n_clients*p[0])
    n_mix0 = round(n_clients*p[1])
    n_mix1 = n_clients - n_iid - n_mix0

    config = get_dataset_config(dataset)
    if config.create_fn:
        config.create_fn()

    if dataset == 'har':
        n_class = config.n_class
        data_func = config.local_cls
        dev_dataset = config.dev_cls()
        split = split_by_subject(data_func.data)
        client_datasets = [data_func(v) for k, v in split.items()]
        return client_datasets, None, dev_dataset, {
            'num_classes': n_class,
            'data_shape': config.data_shape,
            'client_datasets_indexes': split
        }

    n_class = config.n_class
    data_func = config.local_cls
    dev_dataset = config.dev_cls()
    datashape = config.data_shape

    mode = [0] * n_iid + [1] * n_mix0 + [2] * n_mix1

    indices_per_class = np.load(path_to_data / (dataset + '_indices.npy'))
    size_per_class = indices_per_class.shape[1]
    len_iid = int(size_per_class * p[0])
    len_mix0 = int(size_per_class * p[1])
    len_mix1 = int(size_per_class * p[2])

    iid_clients = {}
    if len_iid > 0:
        index_iid = indices_per_class[:, :len_iid].reshape((n_class * len_iid,))
        ds_iid = data_func(index_iid)
        iid_clients = sp.get_iid(ds_iid, n_iid)
        for id, index in iid_clients.items():
            iid_clients[id] = index_iid[index]

    mix0_clients = {}
    if len_mix0 > 0:
        index_mix0 = indices_per_class[:, len_iid: len_iid+len_mix0].reshape((n_class * len_mix0,))
        np.random.shuffle(index_mix0)
        ds_mix0 = data_func(index_mix0)
        mix0_clients = sp.get_mixed_noniid(ds_mix0, n_mix0, r0, num_items=num_shards)
        for id, index in mix0_clients.items():
            mix0_clients[id] = index_mix0[index]

    mix1_clients = {}
    if len_mix1 > 0:
        index_mix1 = indices_per_class[:, len_iid+len_mix0: len_iid+len_mix0+len_mix1].reshape((n_class * len_mix1,))
        np.random.shuffle(index_mix1)
        ds_mix1 = data_func(index_mix1)
        mix1_clients = sp.get_mixed_noniid(ds_mix1, n_mix1, r1, num_items=num_shards)
        for id, index in mix1_clients.items():
            mix1_clients[id] = index_mix1[index]

    client_datasets = []
    client_datasets_indexes = {}
    offset = 0
    for d in [iid_clients, mix0_clients, mix1_clients]:
        for k, v in d.items():
            client_datasets.append(data_func(v))
            client_datasets_indexes[k+offset] = v.tolist()
        offset += len(d)

    return client_datasets, mode, dev_dataset, {
        'num_classes': n_class,
        'data_shape': datashape,
        'client_datasets_indexes': client_datasets_indexes
    }

def dirichlet_data_preparation(
        n_clients = 50,
        dataset = 'cifar10',
        alpha = 0.1,
):
    config = get_dataset_config(dataset)
    if config.create_fn:
        config.create_fn()

    data_func = config.local_cls
    dev_dataset = config.dev_cls()
    datashape = config.data_shape

    if dataset == 'shakespeare':
        n_class = data_func.n_class
        all_indices = np.arange(len(data_func.targets))
        ds_all = data_func(all_indices)
        dirichlet_clients = sp.get_dirichlet_noniid(ds_all, n_clients, alpha)

        client_datasets = []
        client_datasets_indexes = {}
        for k, v in dirichlet_clients.items():
            client_datasets.append(data_func(v))
            client_datasets_indexes[k] = v.tolist()

        return client_datasets, None, dev_dataset, {
            'num_classes': n_class,
            'data_shape': datashape,
            'client_datasets_indexes': client_datasets_indexes
        }

    n_class = config.n_class
    indices_per_class = np.load(path_to_data / (dataset + '_indices.npy'))
    size_per_class = indices_per_class.shape[1]
    index_all = indices_per_class.reshape((n_class * size_per_class,))
    np.random.shuffle(index_all)
    ds_all = data_func(index_all)
    dirichlet_clients = sp.get_dirichlet_noniid(ds_all, n_clients, alpha)
    for id, index in dirichlet_clients.items():
        dirichlet_clients[id] = index_all[index]

    client_datasets = []
    client_datasets_indexes = {}
    for k, v in dirichlet_clients.items():
        client_datasets.append(data_func(v))
        client_datasets_indexes[k] = v.tolist()

    return client_datasets, None, dev_dataset, {
        'num_classes': n_class,
        'data_shape': datashape,
        'client_datasets_indexes': client_datasets_indexes
    }

def proxy_data_preparation(
        size = 500,
        dataset = 'cifar10'
):
    config = get_dataset_config(dataset)
    data_func = config.local_cls

    if dataset in ('har', 'shakespeare'):
        total_data = len(data_func.targets) if dataset == 'shakespeare' else len(data_func.data)
        idx = np.random.choice(total_data, size)
    else:
        indices = np.load(path_to_data / (dataset + '_indices.npy'))
        total_data = sum(len(row) for row in indices)
        idx = np.random.choice(total_data, size)
    return data_func(idx)

def get_dataset_preparation_fn(setting='niid', alpha=0.1, num_shards=2):
    if setting == 'niid':
        from functools import partial
        return partial(local_data_preparation, p=[0.0,0.0,1.0], r1=1.0, num_shards=num_shards)
    if setting == 'diri':
        from functools import partial
        return partial(dirichlet_data_preparation, alpha=alpha)

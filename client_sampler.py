import numpy as np
import random
from typing import Iterator, List

def random_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    """Randomly samples a batch of clients."""
    for _ in range(total_round):
        yield random.sample(range(n_clients), bs)

def sequence_cluster_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    """Clusters all clients into random groups of batch_size."""
    client_ids = list(range(n_clients))
    random.shuffle(client_ids)
    groups = [client_ids[i:i + bs] for i in range(0, n_clients, bs)]
    # random.shuffle(groups)
    i = -1
    for _ in range(total_round):
        i = (i + 1) % len(groups)
        yield groups[i]

def random_cluster_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    """Clusters all clients into random groups of batch_size."""
    client_ids = list(range(n_clients))
    random.shuffle(client_ids)
    groups = [client_ids[i:i + bs] for i in range(0, n_clients, bs)]
    # random.shuffle(groups)
    i = -1
    for _ in range(total_round):
        # i = (i + 1) % len(groups)
        i = random.randint(0, len(groups)-1)
        yield groups[i]

def mode_cluster_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    """Clusters clients of the same mode into groups of batch_size."""
    mode_to_clients = {m: np.where(mode == m)[0].tolist() for m in np.unique(mode)}
    groups = []
    for mode, clients in mode_to_clients.items():
        random.shuffle(clients)
        mode_groups = [clients[i:i + bs] for i in range(0, len(clients), bs)]
        groups.extend(mode_groups)
    # random.shuffle(groups)
    i = -1
    for _ in range(total_round):
        i = (i + 1) % len(groups)
        yield groups[i]

def on_off_cluster_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    client_ids = list(range(n_clients))
    p = bs / n_clients
    for _ in range(total_round):
        # sample = np.random.choice(n_clients, bs, replace=False).tolist()
        sample = np.random.rand(n_clients)
        selected = [client_ids[i] for i in range(n_clients) if sample[i] < p]
        yield selected

def overlap_cluster_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    client_ids = list(range(n_clients))
    random.shuffle(client_ids)
    groups = [client_ids[i:i + bs] for i in range(0, n_clients, bs)]
    alpha = 0.2
    multi = random.sample(client_ids, int(n_clients * alpha))
    n_multi_group = len(multi) // len(groups)
    multis = [multi[i:i+n_multi_group] for i in range(0, len(multi), n_multi_group)]
    for i in range(len(groups)):
        groups[i].extend(multis[i])

    for _ in range(total_round):
        r = random.randint(0, len(groups)-1)
        yield groups[r]


def mode_cluster_sequence_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    """Clusters clients of the same mode into groups of batch_size."""
    mode_to_clients = {m: np.where(mode == m)[0].tolist() for m in np.unique(mode)}
    groups = []
    steps = []
    for mode, clients in mode_to_clients.items():
        random.shuffle(clients)
        mode_groups = [clients[i:i + bs] for i in range(0, len(clients), bs)]
        groups.append(mode_groups)
        step = len(mode_groups)
        steps.append(step)

    total_steps = sum(steps)
    steps = np.cumsum(steps) / total_steps * total_round
    steps = np.round(steps)
    steps[-1] = total_round

    # random.shuffle(groups)
    stage = 0
    for i in range(total_round): 
        if i >= steps[stage]:
            stage += 1
        r = i % len(groups[stage])
        yield groups[stage][r]

def mode_cluster_reverse_sampler(n_clients, mode, bs, total_round = 1000) -> Iterator[List[int]]:
    """Clusters clients of the same mode into groups of batch_size."""
    mode_to_clients = {m: np.where(mode == m)[0].tolist() for m in np.unique(mode)}
    groups = []
    steps = []
    for mode, clients in mode_to_clients.items():
        random.shuffle(clients)
        mode_groups = [clients[i:i + bs] for i in range(0, len(clients), bs)]
        groups.append(mode_groups)
        step = len(mode_groups)
        steps.append(step)
    groups.reverse()
    steps.reverse()

    total_steps = sum(steps)
    steps = np.cumsum(steps) / total_steps * total_round
    steps = np.round(steps)
    steps[-1] = total_round

    # random.shuffle(groups)
    stage = 0
    for i in range(total_round): 
        if i >= steps[stage]:
            stage += 1
        r = i % len(groups[stage])
        yield groups[stage][r]

CS_MAP = {
    'rand': random_sampler,
    'randcls': sequence_cluster_sampler,
    'onoff': on_off_cluster_sampler,
    'clsrand': random_cluster_sampler
}

def set_client_sampler(name='rand'):
    return CS_MAP[name]
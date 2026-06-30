#!/usr/bin/env python3
"""
C1 efficiency statistics — per-group raw comm/compute/accuracy CSV.

Reads `blcompare_results.json` (list of groups, each with seed dirs under
``result/``) and emits ONE long-format CSV per group containing, per
method × seed × round, the raw efficiency signals required by
``efficiency_c1_requirements.md`` (§2.1 accuracy trajectory, §2.2 comm,
§2.3 server FLOPs, §2.4 client FLOPs, §2.5 static memory, §2.6 meta).

BLCompare runs (result/) were produced by code whose details.jsonl has
NO ``efficiency`` field, so every comm/compute quantity here is derived
ANALYTICALLY from the model architecture + each method's per-round mechanics.
Formulas reuse ``flops.py`` helpers and replicate
``servers/flow.py:compute_efficiency_stats`` for the Ours (FLOW) method.

Conventions (match flops.py / the recorded efficiency field exactly):
- cnn_macs  = forward MACs of one single-sample client-CNN forward pass.
- FLOPs     = 2 * MACs.
- client_training_flops(cnn, nb, le) = 2 * (3*cnn) * nb * le   (1 fwd + 1 bwd per batch).
- proxy / gradient operations counted per-sample-summed (cnn * num_samples).

Run from the repo root (data lives under result/):
    python vis/efficiency_stats.py \
        --input-json /path/to/blcompare_results.json \
        --result-root result \
        --output-dir result/efficiency_csv

Self-check (validates the Ours formula against a recorded efficiency field):
    python vis/efficiency_stats.py --self-check \
        result/CompAblation_cifar10_niid_50_10_5_clsrand/Ours_*_niid_full/details.jsonl
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import torch

# allow `import flops`, `import models` when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flops import (  # noqa: E402
    full_param_numel, forward_macs_for_model, client_training_flops,
    macs_to_flops, uplink_full_bytes, uplink_projected_bytes, count_forward_macs,
)
import models  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed experimental config (verified from exp/baseline_compare.sh + AgentSolver)
# ---------------------------------------------------------------------------
DS_META = {
    'cifar10': dict(shape=(3, 32, 32), n_class=10, n_clients=100, avg_samples=500),
    'fashion': dict(shape=(1, 28, 28), n_class=10, n_clients=100, avg_samples=500),
    'har':     dict(shape=(561,),      n_class=6,  n_clients=21,  avg_samples=None),  # per-subject
}
CLIENT_BATCH = 256          # client_solver.batch_size  (efficiency client_batch_size)
LOCAL_EPOCHS = 3            # k[3] in dir names / client_local_epochs
PROXY_SIZE = 1000           # proxy_data_preparation(size=1000), hardcoded in flow/fedawac/edgeflow
# Ours (FLOW) agent config — baseline_compare.sh:47 `-F 128 --recall 5`
FEATURE_DIM = 128
RECALL = 5
AGENT_B = 5                 # AgentSolver.b (5 train batches/round)
AGENT_BATCH = 32            # AgentSolver.batch_size
EMB_DIM = 32
ATTN_DIM = 64
BUFFER_SIZE = 640           # AgentSolver.buffer_size
# FedAWAC slide-window = recall window (baseline_compare.sh:66 `--recall 2`)
FEDAWAC_WINDOW = 2

BYTES_F32 = 4
METHODS = ['Ours', 'FedProx', 'HA', 'FedAWAC', 'F3AST']
CSV_COLUMNS = [
    'group', 'dataset', 'setting', 'sampler', 'seed_dir', 'method', 'method_subdir', 'round',
    'test_accuracy', 'test_loss',
    'uplink_projected_bytes', 'uplink_full_bytes', 'uplink_extra_bytes',
    'uplink_total_bytes', 'downlink_bytes',
    'server_qnet_forward_flops', 'server_qnet_train_flops', 'server_selection_flops',
    'server_aggregation_flops', 'server_proxy_eval_flops', 'server_grad_dist_flops',
    'server_total_flops', 'server_qnet_forward_macs_b1',
    'client_per_client_flops', 'client_total_flops', 'client_cnn_forward_macs',
    'client_num_batches', 'client_size_source',
    'mem_replay_buffer_bytes', 'mem_id_embedding_bytes', 'mem_history_bytes',
    'mem_qnet_params_bytes',
    'n_clients', 'cluster_size', 'selection_size', 'full_param_size',
    'local_epochs', 'batch_size', 'feature_dim', 'recall', 'agent_b',
    'agent_batch', 'total_epochs', 'proxy_size', 'qnet_type',
]


# ---------------------------------------------------------------------------
# dataset + qnet constants
# ---------------------------------------------------------------------------
def ds_constants(dataset):
    """cnn_macs (per-sample fwd MACs), full param count d, n_class, n_clients."""
    meta = DS_META[dataset]
    modelfn = models.set_model_fn(dataset)
    cnn_macs = forward_macs_for_model(modelfn, meta['shape'], meta['n_class'])
    m = modelfn(n_class=meta['n_class'])
    d = full_param_numel(m.state_dict())
    return dict(cnn_macs=int(cnn_macs), d=int(d), n_class=meta['n_class'],
                n_clients=meta['n_clients'], avg_samples=meta['avg_samples'])


def build_qnet(dataset):
    """Build the Ours Q-net (multistep) with the BLCompare config."""
    meta = DS_META[dataset]
    qnetfn = models.set_qnet_fn('multistep')
    return qnetfn(d_raw_feature=FEATURE_DIM, d_embedding=EMB_DIM, d_attention=ATTN_DIM,
                  nclasses=meta['n_class'], use_temporal_attn=True, use_spatial_attn=True,
                  encoder='attention')


def qnet_forward_macs(qnet, cluster_size, selection_size):
    """Replicate servers/flow.py:_qnet_forward_macs (batch=1)."""
    N, K, F, H = cluster_size, selection_size, FEATURE_DIM, RECALL
    nclasses = qnet.nclasses
    B = 1
    x_u = torch.zeros(B, N, F)
    x_s = torch.zeros(B, F)
    indicators = torch.zeros(B, N, dtype=torch.long)
    distribution = torch.zeros(B, nclasses)
    history = torch.zeros(B, H, F + K)
    return count_forward_macs(qnet, (x_u, x_s, indicators, distribution, history, None))


# ---------------------------------------------------------------------------
# per-method per-round constants
# ---------------------------------------------------------------------------
def client_per_client_flops(cnn_macs, dataset):
    """Per-client per-round training FLOPs (client_training_flops convention)."""
    avg = DS_META[dataset]['avg_samples']
    if avg is None:                       # HAR: per-subject sizes vary -> use population mean
        # UCI-HAR: 21 subjects over 7352 train samples
        avg = 7352 // DS_META[dataset]['n_clients']
        src = 'approx(har_mean)'
    else:
        src = 'exact(balanced)'
    nb = math.ceil(avg / CLIENT_BATCH) if avg > 0 else 0
    return client_training_flops(cnn_macs, nb, LOCAL_EPOCHS), nb, src


def server_flops(method, d, cnn_macs, qm, cluster_size, selection_size, avg_samples):
    """Return dict of per-round server FLOPs breakdown for one method.

    Notation: d=param count, cnn=per-sample fwd MACs, qm=Q-net fwd MACs (b=1),
    C=cluster_size, K=selection_size, H=RECALL.
    """
    C, K, H = cluster_size, selection_size, RECALL
    zero = dict(qnet_forward=0, qnet_train=0, selection=0, aggregation=0,
                proxy_eval=0, grad_dist=0, qnet_forward_macs_b1=0)
    if method == 'Ours':
        act = macs_to_flops(qm)
        train = macs_to_flops(5 * AGENT_B * AGENT_BATCH * qm)
        agg = macs_to_flops(d * (K + H + 1))          # temporal aggregation window on
        proxy = macs_to_flops(cnn_macs * PROXY_SIZE * 2)   # output_dist + reward (fwd-only)
        total = act + train + agg + proxy
        return dict(qnet_forward=act, qnet_train=train, selection=0, aggregation=agg,
                    proxy_eval=proxy, grad_dist=0, total=total, qnet_forward_macs_b1=qm)
    if method in ('FedAvg', 'FedProx'):
        agg = macs_to_flops(d * K)
        return dict(zero, aggregation=agg, total=agg)
    if method == 'F3AST':
        sel = macs_to_flops(C + C * max(1, math.log2(C)))   # p^2/r^2 (C) + argsort C*logC
        agg = macs_to_flops(d * K)                           # importance-weighted avg
        return dict(zero, selection=sel, aggregation=agg, total=sel + agg)
    if method == 'FedAWAC':
        # _estimate_logits_variance: K selected clients each forwarded (fwd-only) over proxy
        sel = macs_to_flops(cnn_macs * PROXY_SIZE * K)
        agg = macs_to_flops(d * (K + FEDAWAC_WINDOW))       # weighted agg + slide-window avg
        return dict(zero, selection=sel, aggregation=agg, total=sel + agg)
    if method == 'HA':
        # _run_local_gradient_estimate: ALL C cluster clients do fwd+bwd over their local data
        grad_est = macs_to_flops(cnn_macs * avg_samples * 3 * C)  # 3 = 1 fwd + 1 bwd(≈2)
        # _run_proxy_gradient: fwd+bwd over proxy on global model
        proxy_g = macs_to_flops(cnn_macs * PROXY_SIZE * 3)
        grad_dist = macs_to_flops(d * C)                     # C L2-norms over params (~2*d MACs each)
        agg = macs_to_flops(d * K)
        total = grad_est + proxy_g + grad_dist + agg
        return dict(zero, selection=grad_est + proxy_g, aggregation=agg,
                    proxy_eval=0, grad_dist=grad_dist, total=total)
    raise ValueError(f'unknown method {method}')


def comm_bytes(method, d, cluster_size, selection_size):
    """Per-round uplink/downlink bytes."""
    C, K = cluster_size, selection_size
    up_full = uplink_full_bytes(K, d)                  # K × d × 4
    up_proj = uplink_projected_bytes(C, FEATURE_DIM) if method == 'Ours' else 0   # C × F × 4
    # HA-Edgeflow: each of the C clients uploads only the scalar L2 distance
    # between its local gradient and the proxy gradient (computed locally); the
    # selected K then upload full params (like Ours). The C scalars are negligible,
    # so HA uplink ≈ K × d (same as FedAvg/Ours), NOT the (C+K)×d of uploading
    # full gradients.
    up_extra = (C * BYTES_F32) if method == 'HA' else 0
    up_total = up_full + up_proj + up_extra
    down = K * d * BYTES_F32                            # broadcast global model to K selected clients
    return dict(uplink_projected=up_proj, uplink_full=up_full, uplink_extra=up_extra,
                uplink_total=up_total, downlink=down)


def static_memory(method, d, qnet, n_clients):
    """Server static memory (bytes). Only Ours carries DQN machinery."""
    if method != 'Ours':
        return dict(replay=0, id_emb=0, history=0, qnet_params=0)
    # replay buffer: capacity × per-transition bytes (state tensors; documented estimate).
    # a transition holds: client features (N×F), global feat (F), output dist (n_class),
    # history (recall × (F+K)).
    trans_elements = (n_clients * FEATURE_DIM + FEATURE_DIM + qnet.nclasses
                      + RECALL * (FEATURE_DIM + 5))
    replay = BUFFER_SIZE * trans_elements * BYTES_F32
    # id embedding table: detect nn.Embedding in qnet
    id_emb = sum(p.numel() for mod in qnet.modules() if isinstance(mod, torch.nn.Embedding)
                 for p in mod.parameters()) * BYTES_F32
    history = (RECALL + 1) * d * BYTES_F32             # slide_window deque(maxlen=recall+1)
    qnet_params = full_param_numel(qnet.state_dict()) * 2 * BYTES_F32   # local + target
    return dict(replay=replay, id_emb=id_emb, history=history, qnet_params=qnet_params)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_rounds(details_path):
    """Yield per-round dicts (skip the first efficiency/meta line)."""
    with open(details_path) as f:
        first = f.readline()
        # first line may or may not be the meta header; details rows have 'epoch'
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'epoch' in rec:
                yield rec


def find_method_dirs(group_dir):
    """Map method -> list of (seed_dirname, details_path) under a BLCompare group dir."""
    out = {m: [] for m in METHODS}
    if not Path(group_dir).is_dir():
        return out
    for sub in sorted(Path(group_dir).iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if name.endswith('_combo'):
            # skip the new-method (combo) Ours runs; this script tracks the
            # original-method results. combo is handled separately.
            continue
        det = sub / 'details.jsonl'
        if not det.exists():
            continue
        for m in METHODS:
            if name.startswith(m + '_'):
                out[m].append((name, str(det)))
                break
    return out


def group_csv_name(group):
    return (f"efficiency_{group['dataset']}_{group['setting']}_{group['sampler']}"
            f"_C{group['cluster_size']}_K{group['selection_size']}.csv")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def write_group_csv(group, result_root, output_dir, ds_cache, qnet_cache):
    dataset = group['dataset']
    C, K = group['cluster_size'], group['selection_size']
    dsc = ds_cache[dataset]
    d, cnn_macs = dsc['d'], dsc['cnn_macs']
    avg_samples = dsc['avg_samples'] or (7352 // dsc['n_clients'])
    qnet = qnet_cache[dataset]
    qm = qnet_forward_macs(qnet, C, K)

    per_client_flops, nb, size_src = client_per_client_flops(cnn_macs, dataset)
    # the per-method subdirs live under EACH seed dir; iterate all seed dirs.
    # tag each (method, details) with its BLCompare seed-dir name so seeds stay
    # distinguishable in the CSV.
    method_seed_details = {m: [] for m in METHODS}
    for seed_dir in group['dirs']:
        gd = Path(result_root) / seed_dir
        for m, lst in find_method_dirs(gd).items():
            for (subname, det_path) in lst:
                method_seed_details[m].append((seed_dir, subname, det_path))

    out_path = Path(output_dir) / group_csv_name(group)
    n_rows = 0
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for method in METHODS:
            sf = server_flops(method, d, cnn_macs, qm, C, K, avg_samples)
            cb = comm_bytes(method, d, C, K)
            mem = static_memory(method, d, qnet, dsc['n_clients'])
            for seed_dirname, subname, det_path in method_seed_details[method]:
                for rec in read_rounds(det_path):
                    row = {
                        'group': group_csv_name(group).replace('.csv', ''),
                        'dataset': dataset, 'setting': group['setting'],
                        'sampler': group['sampler'], 'seed_dir': seed_dirname,
                        'method': method, 'method_subdir': subname,
                        'round': rec.get('epoch'),
                        'test_accuracy': rec.get('test_accuracy'),
                        'test_loss': rec.get('test_loss'),
                        'uplink_projected_bytes': cb['uplink_projected'],
                        'uplink_full_bytes': cb['uplink_full'],
                        'uplink_extra_bytes': cb['uplink_extra'],
                        'uplink_total_bytes': cb['uplink_total'],
                        'downlink_bytes': cb['downlink'],
                        'server_qnet_forward_flops': sf['qnet_forward'],
                        'server_qnet_train_flops': sf['qnet_train'],
                        'server_selection_flops': sf['selection'],
                        'server_aggregation_flops': sf['aggregation'],
                        'server_proxy_eval_flops': sf['proxy_eval'],
                        'server_grad_dist_flops': sf['grad_dist'],
                        'server_total_flops': sf['total'],
                        'server_qnet_forward_macs_b1': sf['qnet_forward_macs_b1'],
                        'client_per_client_flops': per_client_flops,
                        'client_total_flops': per_client_flops * K,
                        'client_cnn_forward_macs': cnn_macs,
                        'client_num_batches': nb, 'client_size_source': size_src,
                        'mem_replay_buffer_bytes': mem['replay'],
                        'mem_id_embedding_bytes': mem['id_emb'],
                        'mem_history_bytes': mem['history'],
                        'mem_qnet_params_bytes': mem['qnet_params'],
                        'n_clients': dsc['n_clients'], 'cluster_size': C,
                        'selection_size': K, 'full_param_size': d,
                        'local_epochs': LOCAL_EPOCHS, 'batch_size': CLIENT_BATCH,
                        'feature_dim': FEATURE_DIM, 'recall': RECALL, 'agent_b': AGENT_B,
                        'agent_batch': AGENT_BATCH, 'total_epochs': group['epochs'],
                        'proxy_size': PROXY_SIZE, 'qnet_type': 'multistep',
                    }
                    w.writerow(row)
                    n_rows += 1
    return out_path, n_rows


def print_summary(group, result_root, ds_cache, qnet_cache):
    """One-line per-method constant summary for a group (sanity check)."""
    dataset = group['dataset']
    C, K = group['cluster_size'], group['selection_size']
    d, cnn_macs = ds_cache[dataset]['d'], ds_cache[dataset]['cnn_macs']
    avg_samples = ds_cache[dataset]['avg_samples'] or (7352 // ds_cache[dataset]['n_clients'])
    qm = qnet_forward_macs(qnet_cache[dataset], C, K)
    print(f"\n=== {group_csv_name(group)} (d={d}, cnn_macs={cnn_macs}, qnet_macs_b1={qm}) ===")
    print(f"{'method':<10}{'uplink_tot(B)':>16}{'downlink(B)':>14}{'server_flops':>16}"
          f"{'client_tot':>16}")
    for m in METHODS:
        sf = server_flops(m, d, cnn_macs, qm, C, K, avg_samples)
        cb = comm_bytes(m, d, C, K)
        pc, _, _ = client_per_client_flops(cnn_macs, dataset)
        print(f"{m:<10}{cb['uplink_total']:>16,}{cb['downlink']:>14,}{sf['total']:>16,}"
              f"{pc * K:>16,}")


def self_check(details_path):
    """Compare analytically-derived Ours constants to a recorded efficiency field."""
    with open(details_path) as f:
        meta = json.loads(f.readline())
    if 'efficiency' not in meta:
        print('FAIL: this details.jsonl has no efficiency field (old run).')
        return
    eff = meta['efficiency']
    dataset = None
    for k in ('cifar10', 'fashion', 'har'):
        if k in str(details_path):
            dataset = k
            break
    dataset = dataset or 'cifar10'
    C = eff.get('cluster_size', 10)
    K = eff.get('selection_size', 5)
    dsc = ds_constants(dataset)
    d, cnn_macs = dsc['d'], dsc['cnn_macs']
    avg_samples = dsc['avg_samples'] or (7352 // dsc['n_clients'])
    qnet = build_qnet(dataset)
    qm = qnet_forward_macs(qnet, C, K)
    # recompute ours server flops exactly as flow.py does
    sf = server_flops('Ours', d, cnn_macs, qm, C, K, avg_samples)

    up_proj_want = eff.get('uplink_projected_bytes', uplink_projected_bytes(C, FEATURE_DIM))
    checks = [
        ('full_param_size', d, eff['full_param_size']),
        ('client_cnn_forward_macs', cnn_macs, eff['client_cnn_forward_macs']),
        ('uplink_projected_bytes', uplink_projected_bytes(C, FEATURE_DIM), up_proj_want),
        ('server.rl_flops', sf['qnet_forward'] + sf['qnet_train'], eff['server_flops']['rl_flops']),
        ('server.aggregation_flops', sf['aggregation'], eff['server_flops']['aggregation_flops']),
        ('server.proxy_eval_flops', sf['proxy_eval'], eff['server_flops']['proxy_eval_flops']),
        ('server.total', sf['total'], eff['server_flops']['total']),
        ('server.qnet_forward_macs_b1', qm, eff['server_flops']['qnet_forward_macs_b1']),
    ]
    print(f"self-check vs {details_path} (dataset={dataset}, C={C}, K={K})")
    ok = True
    for name, got, want in checks:
        match = (got == want)
        ok = ok and match
        print(f"  [{'OK ' if match else 'XX '}] {name:<28} got={got:>14,}  want={want:>14,}")
    # client avg (may differ slightly: real per-client sizes vs avg)
    avg = dsc['avg_samples'] or (7352 // dsc['n_clients'])
    nb = math.ceil(avg / CLIENT_BATCH)
    pc = client_training_flops(cnn_macs, nb, LOCAL_EPOCHS)
    cmatch = (pc == eff['client_avg_flops'])
    ok = ok and cmatch
    print(f"  [{'OK ' if cmatch else '~~ '}] client_avg_flops            got={pc:>14,}  "
          f"want={eff['client_avg_flops']:>14,}  (avg-sample convention)")
    print('RESULT:', 'ALL MATCH ✓' if ok else 'MISMATCH (see XX above)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-json', default='blcompare_results.json')
    ap.add_argument('--result-root', default='result')
    ap.add_argument('--output-dir', default='result/efficiency_csv')
    ap.add_argument('--self-check', metavar='DETAILS_JSONL',
                    help='validate Ours formula against a recorded efficiency field')
    args = ap.parse_args()

    if args.self_check:
        self_check(args.self_check)
        return

    with open(args.input_json) as f:
        groups = json.load(f)
    datasets = sorted({g['dataset'] for g in groups})
    ds_cache = {ds: ds_constants(ds) for ds in datasets}
    qnet_cache = {ds: build_qnet(ds) for ds in datasets}
    print('dataset constants:')
    for ds, c in ds_cache.items():
        print(f"  {ds}: full_param_size={c['d']:,}  cnn_forward_macs={c['cnn_macs']:,}  "
              f"n_clients={c['n_clients']}  n_class={c['n_class']}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for g in groups:
        out, n = write_group_csv(g, args.result_root, args.output_dir, ds_cache, qnet_cache)
        print(f"wrote {out} ({n:,} rows)")
        print_summary(g, args.result_root, ds_cache, qnet_cache)


if __name__ == '__main__':
    main()

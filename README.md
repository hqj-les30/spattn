# STCS

**Client Selection for Federated Learning via Deep Reinforcement Learning**

[English](README.md) | [中文](README_zh.md)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg)

> Official supplementary code release. This repository ships **only our method
> (STCS)** together with the infrastructure required to run and visualize it.
> The baseline methods compared against in the paper (FedAvg/FedProx, Edgeflow,
> FedAWAC, F3AST) are **not** included.

---

## Overview

In federated learning, the server must decide *which* clients train each round.
Choosing well under heterogeneous (non-IID) data is hard: a bad selection wastes
communication and steers the global model toward dominant clients. **STCS** treats
client selection as a sequential decision problem and learns a **policy (a DQN
Q-network)** that selects clients from the current cluster to maximize a proxy-data
reward.

Each communication round, STCS:

1. samples a cluster of clients and runs their local updates,
2. encodes every client from a **compact, low-bandwidth signal** —
   `[gradient-norm, training-loss, |D|]` — rather than raw model parameters,
3. scores clients with a **Q-network that uses temporal self-attention** (over a
   recall history) and **spatial self-attention** (across the cluster), and outputs
   **per-class Q-values** that are scalarized for selection,
4. aggregates the selected updates, then **temporally smooths the global parameters**
   over a sliding window,
5. computes an EMA-smoothed proxy-data reward and updates the Q-network.

The default configuration (the method used in the paper) is:
`-q multistep_vec --enc-input scalar --balance-select mean`.

## Key features

- **RL-based selection** — a DQN learns whom to train each round, instead of random or
  heuristic selection.
- **Per-class vectorized Q-values** (`-q multistep_vec`) scalarized by `--balance-select`
  (`mean` for balanced accuracy, `maxmin` to favor the worst class).
- **Low-bandwidth encoder input** (`--enc-input scalar`) — clients upload a few scalars
  for selection, full parameters only when selected.
- **Spatio-temporal attention** in the Q-network (`--recall` controls the history length).
- **Temporal parameter aggregation** across a sliding window (disable with
  `--no-temporal-agg`).
- Built-in **communication / compute (FLOPs) accounting** (`flops.py`), written to every
  run's `details.jsonl`.

## Installation

```bash
conda create -n stcs python=3.10 -y && conda activate stcs
pip install -r requirements.txt
```

A CUDA build of PyTorch is recommended for training speed.

## Datasets

All datasets are read from `~/data/` (the `path_to_data` constant in
`data_wrapper/base.py` — edit it to relocate).

| Dataset       | `-d`       | How to provide                                          |
|---------------|------------|---------------------------------------------------------|
| CIFAR-10      | `cifar10`  | torchvision format under `~/data/` (provide yourself)   |
| CIFAR-100     | `cifar100` | torchvision format under `~/data/`                      |
| Fashion-MNIST | `fashion`  | torchvision format under `~/data/`                      |
| UCI HAR       | `har`      | place the `UCI HAR Dataset/` folder under `~/data/`     |
| Shakespeare   | `shake`    | auto-downloaded on first use                            |

For CIFAR / Fashion-MNIST the loaders use `download=False`; either place
torchvision-format data under `~/data/` once, or flip `download=False` → `True` in the
corresponding `data_wrapper/<dataset>.py`.

## Quick start

Run from the repository root:

```bash
python main.py -a STCS -d cifar10 --setting niid \
    -g 0,1,2,3 -c 10 -s 5 -t 1500 -k 3 \
    -F 128 --recall 5 --enc-input scalar -q multistep_vec \
    -z clsrand --workers-per-gpu 1
```

Key arguments (see `python main.py -h` for the full list):

| Flag                | Meaning                                                              | Default          |
|---------------------|----------------------------------------------------------------------|------------------|
| `-a`                | method (default `STCS`)                                              | `STCS`           |
| `-d`                | dataset                                                              | `cifar10`        |
| `--setting`         | heterogeneity: `niid` (label skew) / `diri` (Dirichlet)              | `niid`           |
| `-g`                | GPU ids, comma-separated                                             | `0,1,2,3`        |
| `-c` / `-s`         | cluster size / selection size                                        | `10` / `5`       |
| `-t` / `-k`         | communication rounds / local epochs                                  | `50` / `3`       |
| `-F`                | feature dimension                                                    | `64`             |
| `--recall`          | history length                                                       | `1`              |
| `-q`                | Q-net: `multistep_vec` (per-class) / `multistep` / `noemb` / `singleT` | `multistep_vec` |
| `--enc-input`       | `scalar` (default) / `abs` / `grad`                                  | `scalar`         |
| `--balance-select`  | `mean` (default) / `maxmin` (favor worst class)                      | `mean`           |
| `--alpha`           | Dirichlet α (smaller = more heterogeneous)                           | `0.1`            |
| `--num-shards`      | label-skew shards per client (fewer = more heterogeneous)            | `2`              |
| `--workers-per-gpu` | process-pool workers per GPU                                         | `2`              |

Per-round metrics (and an `efficiency` header record) are written to
`./result/<run-title>/details.jsonl`, with a full log in `log.log`.

## Reproducing the experiments

The `exp/` directory contains parameterized runners; each script documents its own
flags in its header comment.

| Script                       | What it varies                                                                  |
|------------------------------|---------------------------------------------------------------------------------|
| `exp/comp_ablation.sh`       | component ablation: Full vs w/o temporal-agg / temporal-attn / spatial-attn / MLP-encoder / id-emb |
| `exp/emb_ablation.sh`        | identity-embedding ablation across `niid` / `diri`                              |
| `exp/qnet_compare.sh`        | Q-network architecture                                                          |
| `exp/recall_compare.sh`      | history (recall) length                                                         |
| `exp/feature_compare.sh`     | feature-reduction method & dimension                                            |
| `exp/selection_ratio.sh`     | selection-ratio robustness (20 / 50 / 80 %)                                     |
| `exp/hetero_compare.sh`      | heterogeneity strength (Dirichlet α / label-skew shards)                        |

Example:

```bash
bash exp/comp_ablation.sh -d cifar10 -S niid -t 1500 -g 0,1,2,3 -P 2
```

## Visualization

The `vis/` directory contains plotting and efficiency-analysis tools (run from the repo
root). The curve/bar tools consume the `<method>: saved results to <path>` lines that
`main.py` prints and that the `exp/` scripts collect into `stdout.log`, so the usual
pattern is to pipe that log in:

```bash
cat result/<run-dir>/stdout.log | python vis/curvevis.py --output-dir result/<run-dir> --plot-type line
```

| Tool                                   | Produces                                                            |
|----------------------------------------|---------------------------------------------------------------------|
| `vis/curvevis.py`                      | accuracy/F1 curves, error bars, shaded variance                     |
| `vis/ablation_vis.py`                  | grouped ablation bars (supports `--delta-vs`)                       |
| `vis/bl_compare_bar.py`                | grouped comparison bars                                             |
| `vis/sel_ratio_bar.py`                 | selection-ratio robustness bars                                     |
| `vis/efficiency_stats.py`              | per-group communication/compute/accuracy CSV                        |
| `vis/efficiency_bar.py`                | communication & server-FLOPs-to-target bars                         |
| `vis/calculate_efficiency_savings.py`  | uplink communication savings summary                                |
| `vis/calculate_server_flops_savings.py`| server-FLOPs savings summary                                        |
| `vis/legend_only.py`                   | standalone legend image                                             |

Run `python vis/<tool>.py -h` for each tool's exact arguments.

## Repository structure

```
main.py              entry point & CLI
servers/             STCS server (stcs.py) + base server infrastructure (base.py)
agent/               DQN agent with the per-class vectorized Q-network (dqn.py)
data_wrapper/        datasets and non-IID partitioning (Dirichlet / label skew)
clients.py           local client training
models.py            client CNN + Q-network architectures
featuredim.py        feature reduction (Random Projection / PCA / ...)
flops.py             communication & compute (FLOPs) accounting
sampling.py          data sampling strategies (IID / non-IID / Dirichlet)
client_sampler.py    client clustering samplers
evaluation.py        model evaluation
exp/                 reproduction / ablation runner scripts
vis/                 plotting & efficiency-analysis tools
```

## Citation

If you use this code, please cite the paper (BibTeX to be added upon acceptance).

## License

Released under the [MIT License](LICENSE).

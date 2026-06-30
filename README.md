# FLOW: Federated Learning with Optimal Client Selection

Official implementation of **FLOW**, a reinforcement-learning (DQN) client-selection
method for federated learning under heterogeneous (non-IID) data. This repository is
the supplementary code release accompanying our paper.

> This release ships **only our method (FLOW)** together with the infrastructure
> needed to run and visualize it. The baseline methods compared against in the paper
> (FedAvg/FedProx, Edgeflow/HA-Edgeflow, FedAWAC, F3AST) are **not** included.

## Method overview

Each communication round, FLOW learns and applies a Q-network policy that selects
which clients train from the current cluster. Core ideas:

- **Per-class vectorized Q-values** (`-q multistep_vec`): the Q-network outputs one
  Q-value per class, scalarized for selection via `--balance-select` (default `mean`).
- **Scalar encoder input** (`--enc-input scalar`): each client is encoded from the
  compact signal `[gradient-norm, training-loss, |D|]` instead of raw parameters.
- **Temporal self-attention** over a recall history, **spatial self-attention** across
  the cluster, and **temporal parameter aggregation** over a sliding window.
- Reward is proxy-data macro-F1 (or accuracy), EMA-smoothed; an optional
  progress-relative reward (`--progress-relative-reward`) removes the monotonic
  training-progress trend.

The default configuration (the method used in the paper) is:
`-q multistep_vec --enc-input scalar --balance-select mean`.

## Installation

```bash
conda create -n flow python=3.10 -y && conda activate flow
pip install -r requirements.txt
```

A CUDA build of PyTorch is recommended for training speed.

## Data

All datasets are read from `~/data/` (the `path_to_data` constant in
`data_wrapper/base.py` — edit it to relocate).

| Dataset        | `-d`       | How to provide                                              |
|----------------|------------|-------------------------------------------------------------|
| CIFAR-10       | `cifar10`  | torchvision format under `~/data/` (provide yourself)       |
| CIFAR-100      | `cifar100` | torchvision format under `~/data/`                          |
| Fashion-MNIST  | `fashion`  | torchvision format under `~/data/`                          |
| UCI HAR        | `har`      | place the `UCI HAR Dataset/` folder under `~/data/`         |
| Shakespeare    | `shake`    | auto-downloaded on first use                                |

For CIFAR / Fashion-MNIST the loaders use `download=False`; either place
torchvision-format data under `~/data/` once, or flip `download=False` → `True` in
the corresponding `data_wrapper/<dataset>.py`.

## Running FLOW

Run from the repository root:

```bash
python main.py -a Ours -d cifar10 --setting niid \
    -g 0,1,2,3 -c 10 -s 5 -t 1500 -k 3 \
    -F 128 --recall 5 --enc-input scalar -q multistep_vec \
    -z clsrand --workers-per-gpu 1
```

Key arguments (see `python main.py -h` for the full list):

| Flag                | Meaning                                                          | Default     |
|---------------------|------------------------------------------------------------------|-------------|
| `-a`                | method (`Ours` = FLOW; the `FLOW` alias also works)              | `Ours`      |
| `-d`                | dataset                                                          | `cifar10`   |
| `--setting`         | heterogeneity: `niid` (label skew) / `diri` (Dirichlet)          | `niid`      |
| `-g`                | GPU ids, comma-separated                                         | `0,1,2,3`   |
| `-c` / `-s`         | cluster size / selection size                                    | `10` / `5`  |
| `-t` / `-k`         | communication rounds / local epochs                              | `50` / `3`  |
| `-F`                | feature dimension                                                | `64`        |
| `--recall`          | history length                                                   | `1`         |
| `-q`                | Q-net: `multistep_vec` (per-class) / `multistep` / `noemb` / `singleT` | `multistep_vec` |
| `--enc-input`       | `scalar` (default) / `abs` / `grad`                              | `scalar`    |
| `--balance-select`  | `mean` (default) / `maxmin` (favor worst class)                  | `mean`      |
| `--alpha`           | Dirichlet α (smaller = more heterogeneous)                       | `0.1`       |
| `--num-shards`      | label-skew shards per client (fewer = more heterogeneous)        | `2`         |
| `--workers-per-gpu` | process-pool workers per GPU                                     | `2`         |

Per-round metrics (and an `efficiency` header record) are written to
`./result/<run-title>/details.jsonl`, with a full log in `log.log`.

## Reproducing the ablations

The `exp/` directory contains parameterized runners; each script documents its own
flags in its header comment.

| Script                       | What it varies                                                            |
|------------------------------|---------------------------------------------------------------------------|
| `exp/comp_ablation.sh`       | component ablation: Full vs w/o temporal-agg / temporal-attn / spatial-attn / MLP-encoder / id-emb |
| `exp/emb_ablation.sh`        | identity-embedding ablation across `niid` / `diri`                        |
| `exp/qnet_compare.sh`        | Q-network architecture                                                    |
| `exp/recall_compare.sh`      | history (recall) length                                                   |
| `exp/feature_compare.sh`     | feature-reduction method & dimension                                      |
| `exp/selection_ratio.sh`     | selection-ratio robustness (20 / 50 / 80 %)                               |
| `exp/hetero_compare.sh`      | heterogeneity strength (Dirichlet α / label-skew shards)                  |

Example:

```bash
bash exp/comp_ablation.sh -d cifar10 -S niid -t 1500 -g 0,1,2,3 -P 2
```

## Visualization

The `vis/` directory contains plotting and efficiency-analysis tools (run from the
repo root). The curve/bar tools consume the `<method>: saved results to <path>`
lines that `main.py` prints and that the `exp/` scripts collect into `stdout.log`,
so the usual pattern is to pipe that log in:

```bash
cat result/<run-dir>/stdout.log | python vis/curvevis.py --output-dir result/<run-dir> --plot-type line
```

| Tool                                   | Produces                                                        |
|----------------------------------------|-----------------------------------------------------------------|
| `vis/curvevis.py`                      | accuracy/F1 curves, error bars, shaded variance                 |
| `vis/ablation_vis.py`                  | grouped ablation bars (supports `--delta-vs`)                   |
| `vis/bl_compare_bar.py`                | grouped comparison bars                                         |
| `vis/sel_ratio_bar.py`                 | selection-ratio robustness bars                                 |
| `vis/efficiency_stats.py`              | per-group communication/compute/accuracy CSV                    |
| `vis/efficiency_bar.py`                | communication & server-FLOPs-to-target bars                     |
| `vis/calculate_efficiency_savings.py`  | uplink communication savings summary                            |
| `vis/calculate_server_flops_savings.py`| server-FLOPs savings summary                                    |
| `vis/legend_only.py`                   | standalone legend image                                         |

Run `python vis/<tool>.py -h` for each tool's exact arguments.

## Repository structure

```
main.py              entry point & CLI
servers/             FLOW server (flow.py) + base server infrastructure (base.py)
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

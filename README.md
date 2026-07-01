# Federated Client Selection under Partial Visibility: A POMDP Approach with Spatio-Temporal Attention

[English](README.md) | [中文](README_zh.md)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg)

Official code release — the core implementation of our method (code key `STCS`) for
federated client selection. Full method details are in the paper:
[arXiv:2605.11752](https://arxiv.org/abs/2605.11752).

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{hou2026federatedclientselectionpartial,
      title={Federated Client Selection under Partial Visibility: A POMDP Approach with Spatio-Temporal Attention},
      author={Qijun Hou and Yuchen Shi and Pingyi Fan and Khaled B. Letaief},
      year={2026},
      eprint={2605.11752},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.11752},
}
```

## Installation

```bash
conda create -n stcs python=3.10 -y && conda activate stcs
pip install -r requirements.txt
```

A CUDA build of PyTorch is recommended for training speed.

## Datasets

All datasets are read from `~/data/` (the `path_to_data` constant in
`data_wrapper/base.py` — edit it to relocate).

| Dataset       | `-d`       | How to provide                                        |
|---------------|------------|-------------------------------------------------------|
| CIFAR-10      | `cifar10`  | torchvision format under `~/data/` (provide yourself) |
| Fashion-MNIST | `fashion`  | torchvision format under `~/data/`                    |
| UCI HAR       | `har`      | place the `UCI HAR Dataset/` folder under `~/data/`   |

For CIFAR-10 / Fashion-MNIST the loaders use `download=False`; either place
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
| `-d`                | dataset (`cifar10` / `fashion` / `har`)                              | `cifar10`        |
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
```

## License

Released under the [MIT License](LICENSE).

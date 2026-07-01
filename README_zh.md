# Federated Client Selection under Partial Visibility: A POMDP Approach with Spatio-Temporal Attention

[English](README.md) | [中文](README_zh.md)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg)

官方代码仓库 —— 我们提出的联邦学习客户端选择方法（代码标识 `STCS`）的核心实现。方法的完整细节见论文：
[arXiv:2605.11752](https://arxiv.org/abs/2605.11752)。

## 引用

如果您使用了本代码，请引用我们的论文：

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

## 安装

```bash
conda create -n stcs python=3.10 -y && conda activate stcs
pip install -r requirements.txt
```

建议安装支持 CUDA 的 PyTorch 以加速训练。

## 数据集

所有数据集从 `~/data/` 读取 (由 `data_wrapper/base.py` 中的 `path_to_data` 常量控制，可按需修改)。

| 数据集        | `-d`       | 获取方式                                       |
|---------------|------------|------------------------------------------------|
| CIFAR-10      | `cifar10`  | torchvision 格式，置于 `~/data/` (需自行准备)  |
| Fashion-MNIST | `fashion`  | torchvision 格式，置于 `~/data/`               |
| UCI HAR       | `har`      | 将 `UCI HAR Dataset/` 文件夹置于 `~/data/`     |

CIFAR-10 / Fashion-MNIST 的加载器使用 `download=False`；可自行将 torchvision 格式数据放入
`~/data/`，或将对应 `data_wrapper/<dataset>.py` 中的 `download=False` 改为 `True`。

## 快速开始

在仓库根目录运行：

```bash
python main.py -a STCS -d cifar10 --setting niid \
    -g 0,1,2,3 -c 10 -s 5 -t 1500 -k 3 \
    -F 128 --recall 5 --enc-input scalar -q multistep_vec \
    -z clsrand --workers-per-gpu 1
```

主要参数 (完整列表见 `python main.py -h`)：

| 参数                | 含义                                                                | 默认值           |
|---------------------|---------------------------------------------------------------------|------------------|
| `-a`                | 方法 (默认 `STCS`)                                                  | `STCS`           |
| `-d`                | 数据集 (`cifar10` / `fashion` / `har`)                              | `cifar10`        |
| `--setting`         | 异构设置：`niid` (标签倾斜) / `diri` (Dirichlet)                    | `niid`           |
| `-g`                | GPU 编号，逗号分隔                                                  | `0,1,2,3`        |
| `-c` / `-s`         | 簇大小 / 选择数量                                                   | `10` / `5`       |
| `-t` / `-k`         | 通信轮数 / 本地训练轮数                                             | `50` / `3`       |
| `-F`                | 特征维度                                                            | `64`             |
| `--recall`          | 历史长度                                                            | `1`              |
| `-q`                | Q 网络：`multistep_vec` (逐类别) / `multistep` / `noemb` / `singleT` | `multistep_vec` |
| `--enc-input`       | `scalar` (默认) / `abs` / `grad`                                    | `scalar`         |
| `--balance-select`  | `mean` (默认) / `maxmin` (偏向最差类别)                             | `mean`           |
| `--alpha`           | Dirichlet α (越小越异构)                                            | `0.1`            |
| `--num-shards`      | 标签倾斜时每客户端分片数 (越少越异构)                               | `2`              |
| `--workers-per-gpu` | 每 GPU 进程池工作进程数                                             | `2`              |

每轮指标 (以及 `efficiency` 头部记录) 写入 `./result/<运行名>/details.jsonl`，完整日志见 `log.log`。

## 仓库结构

```
main.py              入口与命令行
servers/             STCS 服务端 (stcs.py) + 基础服务端设施 (base.py)
agent/               DQN 智能体，逐类别向量化 Q 网络 (dqn.py)
data_wrapper/        数据集与 non-IID 划分 (Dirichlet / 标签倾斜)
clients.py           客户端本地训练
models.py            客户端 CNN + Q 网络结构
featuredim.py        特征降维 (随机投影 / PCA / ...)
flops.py             通信与计算 (FLOPs) 统计
sampling.py          数据采样策略 (IID / non-IID / Dirichlet)
client_sampler.py    客户端聚类采样器
evaluation.py        模型评估
```

## 许可证

以 [MIT 许可证](LICENSE) 发布。

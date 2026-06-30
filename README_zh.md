# STCS

**基于深度强化学习的联邦学习客户端选择**

[English](README.md) | [中文](README_zh.md)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg)

> 官方补充代码仓库。本仓库**仅包含我们的方法 (STCS)** 及其运行与可视化所需的基础设施。
> 论文中用于对比的基线方法 (FedAvg/FedProx、Edgeflow、FedAWAC、F3AST) **不包含**在内。

---

## 概述

在联邦学习中，服务器每轮需要决定*哪些*客户端参与训练。在异构 (non-IID) 数据下做好
这一选择并不容易：选择不当会浪费通信资源，并把全局模型推向强势客户端。**STCS** 将客户端
选择建模为一个序列决策问题，并学习一个 **策略 (DQN Q 网络)**，从当前簇中选择客户端，以
最大化代理数据集上的奖励。

每个通信轮，STCS 依次完成：

1. 采样一个客户端簇并执行本地更新；
2. 用**紧凑、低带宽的信号**刻画每个客户端 —— `[梯度范数, 训练损失, |D|]` —— 而非原始模型参数；
3. 用一个**带时序自注意力** (回顾历史) 与**空间自注意力** (跨簇) 的 Q 网络对客户端打分，
   输出**逐类别 Q 值**并标量化用于选择；
4. 聚合被选中客户端的更新，并在滑动窗口上做**时序参数平滑**；
5. 计算经 EMA 平滑的代理数据奖励并更新 Q 网络。

默认配置 (论文所用方法) 为：
`-q multistep_vec --enc-input scalar --balance-select mean`。

## 主要特性

- **基于强化学习的选择** —— 用 DQN 学习每轮训练哪些客户端，而非随机或启发式选择。
- **逐类别向量化 Q 值** (`-q multistep_vec`)，由 `--balance-select` 标量化
  (`mean` 兼顾均衡准确率，`maxmin` 偏向最差类别)。
- **低带宽编码器输入** (`--enc-input scalar`) —— 客户端仅上传少量标量用于选择，仅在被选中时
  才上传完整参数。
- Q 网络中的**时空注意力** (`--recall` 控制历史长度)。
- 跨滑动窗口的**时序参数聚合** (可用 `--no-temporal-agg` 关闭)。
- 内置**通信 / 计算 (FLOPs) 统计** (`flops.py`)，写入每次运行的 `details.jsonl`。

## 安装

```bash
conda create -n stcs python=3.10 -y && conda activate stcs
pip install -r requirements.txt
```

建议安装支持 CUDA 的 PyTorch 以加速训练。

## 数据集

所有数据集从 `~/data/` 读取 (由 `data_wrapper/base.py` 中的 `path_to_data` 常量控制，可按需修改)。

| 数据集        | `-d`       | 获取方式                                                       |
|---------------|------------|----------------------------------------------------------------|
| CIFAR-10      | `cifar10`  | torchvision 格式，置于 `~/data/` (需自行准备)                  |
| CIFAR-100     | `cifar100` | torchvision 格式，置于 `~/data/`                               |
| Fashion-MNIST | `fashion`  | torchvision 格式，置于 `~/data/`                               |
| UCI HAR       | `har`      | 将 `UCI HAR Dataset/` 文件夹置于 `~/data/`                     |
| Shakespeare   | `shake`    | 首次使用时自动下载                                             |

CIFAR / Fashion-MNIST 的加载器使用 `download=False`；可自行将 torchvision 格式数据放入
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
| `-d`                | 数据集                                                              | `cifar10`        |
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

## 复现实验

`exp/` 目录下为参数化运行脚本，每个脚本的开头注释中说明了各自支持的参数。

| 脚本                         | 变化的因素                                                                     |
|------------------------------|--------------------------------------------------------------------------------|
| `exp/comp_ablation.sh`       | 组件消融：Full vs 去掉 时序聚合 / 时序注意力 / 空间注意力 / MLP 编码器 / 身份嵌入 |
| `exp/emb_ablation.sh`        | 身份嵌入消融，跨 `niid` / `diri`                                               |
| `exp/qnet_compare.sh`        | Q 网络结构                                                                     |
| `exp/recall_compare.sh`      | 历史 (recall) 长度                                                             |
| `exp/feature_compare.sh`     | 特征降维方法与维度                                                             |
| `exp/selection_ratio.sh`     | 选择比例鲁棒性 (20 / 50 / 80 %)                                                |
| `exp/hetero_compare.sh`      | 异构强度 (Dirichlet α / 标签倾斜分片数)                                        |

示例：

```bash
bash exp/comp_ablation.sh -d cifar10 -S niid -t 1500 -g 0,1,2,3 -P 2
```

## 可视化

`vis/` 目录下为绘图与效率分析工具 (在仓库根目录运行)。其中曲线/柱状工具会读取 `main.py`
输出的 `<方法>: saved results to <路径>` 行，`exp/` 脚本会将这些行汇总到 `stdout.log`，
因此通常的做法是把该日志通过管道传入：

```bash
cat result/<运行目录>/stdout.log | python vis/curvevis.py --output-dir result/<运行目录> --plot-type line
```

| 工具                                   | 产出                                                                |
|----------------------------------------|---------------------------------------------------------------------|
| `vis/curvevis.py`                      | 准确率/F1 曲线、误差棒、阴影方差                                    |
| `vis/ablation_vis.py`                  | 分组消融柱状图 (支持 `--delta-vs`)                                  |
| `vis/bl_compare_bar.py`                | 分组对比柱状图                                                      |
| `vis/sel_ratio_bar.py`                 | 选择比例鲁棒性柱状图                                                |
| `vis/efficiency_stats.py`              | 每组的通信/计算/准确率 CSV                                          |
| `vis/efficiency_bar.py`                | 到目标准确率的通信 & 服务器 FLOPs 柱状图                            |
| `vis/calculate_efficiency_savings.py`  | 上行通信节省汇总                                                    |
| `vis/calculate_server_flops_savings.py`| 服务器 FLOPs 节省汇总                                               |
| `vis/legend_only.py`                   | 独立图例图                                                          |

运行 `python vis/<工具>.py -h` 查看每个工具的具体参数。

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
exp/                 复现 / 消融运行脚本
vis/                 绘图与效率分析工具
```

## 引用

如果您使用了本代码，请引用我们的论文 (BibTeX 将在论文录用后补充)。

## 许可证

以 [MIT 许可证](LICENSE) 发布。

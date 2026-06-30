import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from pathlib import Path
from typing import List, Dict
import logging
import sys
import re
import argparse
from pathlib import Path

# 输出图表的默认 DPI
DPI = 500

def parse_results_from_stdin(sort_by_h=False):
    """
    Read lines from stdin like:
        "FedGMI: saved results to /path/details.jsonl"
    Return:
        methods = [ ... ]
        paths   = [ ... ]
    """
    methods = []
    paths = []

    # Regex: <method>: saved results to <path>
    pattern = re.compile(r"^(.*?):\s*saved results to\s*(.*)$")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        m = pattern.match(line)
        if m:
            method = m.group(1).strip()
            path   = m.group(2).strip()
            methods.append(method)
            paths.append(path)
        else:
            print(f"[WARN] Unrecognized line format: {line}", file=sys.stderr)

    combined = list(zip(methods, paths))

    if sort_by_h:
        h_pattern = re.compile(r'H=(\d+)')
        def sort_key(item):
            method, _ = item
            m = h_pattern.search(method)
            return int(m.group(1)) if m else 0
    else:
        def sort_key(item):
            method, _ = item
            return (0, "") if method == "STCS" else (1, method)

    combined_sorted = sorted(combined, key=sort_key)
    methods_sorted, paths_sorted = zip(*combined_sorted) if combined_sorted else ([], [])

    return list(methods_sorted), list(paths_sorted)

def load_data_from_paths(log_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Loads data from a list of JSONL files, skipping the first (meta info) line 
    of each file, and returns a dictionary mapping file path to DataFrame.
    """
    data_frames = {}
    for path in log_paths:
        try:
            # 1. 读取文件内容
            with open(path, 'r', encoding='utf-8') as f:
                # 2. 跳过第一行 (Meta Info)
                f.readline() 
                # 3. 读取剩余的所有行
                remaining_content = f.read()
            
            # 4. 使用 io.StringIO 将剩余内容作为文件对象传递给 pandas
            # 这样 pandas 就能直接读取剩下的 JSONL 格式数据
            data_io = io.StringIO(remaining_content)
            
            # 5. 使用 lines=True 读取 JSONL 格式
            df = pd.read_json(data_io, lines=True)
            
            data_frames[path] = df
            # print(f"Successfully loaded data from: {path} ({len(df)} epochs), skipping meta info line.")
        
        except FileNotFoundError:
            # print(f"Error: Log file not found at {path}")
            pass
        except Exception as e:
            # 捕获所有其他异常，包括 pandas 读取时的格式错误
            # print(f"Error processing file {path}: {e}")
            pass
            
    if not data_frames:
        # print("No data loaded. Exiting.")
        return
    return data_frames

# --- 绘图函数更新：接收 Path 对象参数 ---

def plot_agent_loss(data_frames: Dict[Path, pd.DataFrame], output_dir: Path, output_prefix: str, legends: List[str]):
    """
    Plots the DQN Agent Loss (loss_agent) for all experiments and saves it 
    to the specified directory using pathlib.
    """
    plt.figure(figsize=(10, 6))
    n_curves = 0
    
    for i, (path, df) in enumerate(data_frames.items()):
        # path.stem 获取文件名（不含后缀）
        # label = path.stem 
        epochs = df['epoch']
        if 'loss_agent' not in df.columns:
            continue
        plt.plot(epochs, df['loss_agent'], label=legends[i], alpha=0.7)
        n_curves += 1
    
    if n_curves == 0:
        plt.close()
        return
    plt.title('DQN Agent Loss (Training Stability) - Comparison', fontsize=14)
    plt.xlabel('Federated Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(title="Experiment")
    
    # 构造完整的输出路径：使用 / 运算符拼接 Path 对象
    filename = f"{output_prefix}_agent_loss_comparison.png"
    output_filename = output_dir / filename  # 使用 / 运算符进行路径拼接
    
    plt.savefig(output_filename, dpi=DPI)
    plt.close() 
    # print(f"Successfully generated plot and saved to {output_filename}")

def plot_test_loss(data_frames: Dict[Path, pd.DataFrame], output_dir: Path, output_prefix: str, legends: List[str]):
    """
    Plots the Federated Test Loss (test_loss) for all experiments and saves it 
    to the specified directory using pathlib.
    """
    plt.figure(figsize=(10, 6))
    
    for i, (path, df) in enumerate(data_frames.items()):
        label = path.stem
        epochs = df['epoch']
        plt.plot(epochs, df['test_loss'], label=legends[i])
    
    plt.title('Global Model Test Loss - Comparison', fontsize=14)
    plt.xlabel('Federated Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend(title="Experiment")
    
    # 构造完整的输出路径
    filename = f"{output_prefix}_test_loss_comparison.png"
    output_filename = output_dir / filename
    
    plt.savefig(output_filename, dpi=DPI)
    plt.close()
    # print(f"Successfully generated plot and saved to {output_filename}")

def plot_test_accuracy_with_variance(data_frames: Dict[Path, pd.DataFrame], output_dir: Path, output_prefix: str, window_size: int = 10, legends: List[str]=None, nolegend: bool = False):
    """
    Plots the Federated Test Accuracy (test_accuracy) with moving average and variance 
    for all experiments, and saves it to the specified directory using pathlib.
    """
    plt.figure(figsize=(10, 7.5))
    colors = plt.get_cmap('tab10').colors  # 预定义颜色列表
    
    for i, (path, df) in enumerate(data_frames.items()):
        label_base = legends[i] if legends else path.stem
        epochs = df['epoch']
        
        # Calculate Moving Average and Standard Deviation
        rolling_mean = df['test_accuracy'].rolling(window=window_size, min_periods=1, center=False).mean()
        rolling_std = df['test_accuracy'].rolling(window=window_size, min_periods=1, center=False).std().fillna(0)

        upper_bound = rolling_mean + rolling_std
        lower_bound = rolling_mean - rolling_std
        color = colors[i]
        
        # 绘图并获取颜色
        line, = plt.plot(epochs, rolling_mean, 
                         label=label_base, color=color,
                         linewidth=2, alpha=0.85)
        color = line.get_color()

        # 阴影区域
        plt.fill_between(epochs, lower_bound, upper_bound, 
                         color=color, alpha=0.10,
                         label=f'{label_base} (± 1 Std Dev)')

    # plt.title('Global Model Test Accuracy (Smoothed) - Comparison', fontsize=14)
    plt.xlabel('Communication Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 80)
    plt.grid(True, ls="--", alpha=0.6)
    if not nolegend:
        plt.legend(loc='lower right', fontsize=14)

    # 构造完整的输出路径
    filename = f"{output_prefix}_test_accuracy_comparison.pdf"
    output_filename = output_dir / filename
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=DPI)
    plt.close()
    # print(f"Successfully generated plot and saved to {output_filename}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

def plot_test_accuracy_error_bar(data_frames: Dict[Path, pd.DataFrame],
                                output_dir: Path,
                                output_prefix: str,
                                interval: int = 100,
                                legends: List[str] = None,
                                nolegend: bool = False):
    """
    绘制联邦测试准确率的 Error Bar 图。
    每 100 个 round 取一个点，点为均值，bar 为标准差。
    """
    plt.figure(figsize=(10, 7.5))
    colors = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']

    first_df = next(iter(data_frames.values()))
    total_epochs = len(first_df)
    num_points = total_epochs // interval
    if num_points < 10:
        interval = total_epochs // 10
        num_points = total_epochs // interval

    x_axis = np.arange(interval, num_points * interval + 1, interval)
    max_val = 0

    for i, (path, df) in enumerate(data_frames.items()):
        label_base = legends[i] if legends else path.stem

        acc_data = df['test_accuracy'].values[:num_points * interval]
        reshaped_acc = acc_data.reshape(num_points, interval)

        means = np.mean(reshaped_acc, axis=1)
        stds = np.std(reshaped_acc, axis=1)
        max_val = max(max_val, (means + stds).max())

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        x_offset = (i - len(data_frames)/2) * (interval * 0.05)

        plt.errorbar(x_axis + x_offset, means, yerr=stds,
                     label=label_base,
                     color=color,
                     marker=marker,
                     markersize=7,
                     capsize=3,
                     elinewidth=1,
                     linewidth=2,
                     markeredgewidth=1.5,
                     markerfacecolor='white',
                     alpha=0.9)

    plt.xlabel('Communication Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(20, 100) if max_val > 80 else plt.ylim(0, 80)
    plt.xticks(np.arange(0, num_points * interval + 1, interval))
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=14)
    plt.grid(True, ls='--', alpha=0.4)
    if not nolegend:
        plt.legend(loc='lower right', fontsize=16, frameon=True,
                   title='Mean ± Std per interval', title_fontsize=12)

    # 保存文件
    filename = f"{output_prefix}_errorbar_comparison.pdf"
    output_filename = output_dir / filename
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

def plot_test_accuracy_line(data_frames: Dict[Path, pd.DataFrame],
                           output_dir: Path,
                           output_prefix: str,
                           interval: int = 100,
                           legends: List[str] = None,
                           nolegend: bool = False):
    """
    绘制联邦测试准确率的折线图（无 error bar）。
    每 interval 个 round 取一个均值点。
    """
    plt.figure(figsize=(10, 4.5))
    colors = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']

    first_df = next(iter(data_frames.values()))
    total_epochs = len(first_df)
    num_points = total_epochs // interval
    if num_points < 10:
        interval = total_epochs // 10
        num_points = total_epochs // interval

    x_axis = np.arange(interval, num_points * interval + 1, interval)
    max_val = 0

    for i, (path, df) in enumerate(data_frames.items()):
        label_base = legends[i] if legends else path.stem

        acc_data = df['test_accuracy'].values[:num_points * interval]
        reshaped_acc = acc_data.reshape(num_points, interval)

        means = np.mean(reshaped_acc, axis=1)
        max_val = max(max_val, means.max())

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(x_axis, means,
                 label=label_base,
                 color=color,
                 marker=marker,
                 markersize=8,
                 linewidth=3,
                 markeredgewidth=2,
                 markerfacecolor='white',
                 alpha=0.9)

    plt.ylim(50, 100) if max_val > 80 else plt.ylim(30, 80)
    plt.yticks(np.arange((50 if max_val > 80 else 30), (101 if max_val > 80 else 81), 10))
    plt.xticks(np.arange(0, num_points * interval + 1, interval))
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(axis='y', ls='--', alpha=0.4)
    if not nolegend:
        plt.legend(loc='lower right', fontsize=16, frameon=True)

    filename = f"{output_prefix}_line_comparison.pdf"
    output_filename = output_dir / filename

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

def plot_test_accuracy_shaded(data_frames: Dict[Path, pd.DataFrame],
                             output_dir: Path,
                             output_prefix: str,
                             interval: int = 100,
                             legends: List[str] = None,
                             nolegend: bool = False):
    """
    绘制等间隔采样点 + 阴影的准确率图。
    每 interval 个轮次计算一个均值点和标准差阴影。
    """
    plt.figure(figsize=(10, 7.5))
    colors = plt.get_cmap('tab10').colors

    first_df = next(iter(data_frames.values()))
    total_epochs = len(first_df)
    num_points = total_epochs // interval
    if num_points < 10:
        interval = total_epochs // 10
        num_points = total_epochs // interval

    x_axis = np.arange(interval / 2, num_points * interval, interval)
    max_val = 0

    for i, (path, df) in enumerate(data_frames.items()):
        label_base = legends[i] if legends else path.stem

        acc_data = df['test_accuracy'].values[:num_points * interval]
        reshaped_acc = acc_data.reshape(num_points, interval)

        means = np.mean(reshaped_acc, axis=1)
        stds = np.std(reshaped_acc, axis=1)
        max_val = max(max_val, (means + stds).max())

        color = colors[i % len(colors)]

        plt.fill_between(x_axis, means - stds, means + stds,
                         color=color, alpha=0.15,
                         edgecolor='none')

        plt.plot(x_axis, means,
                 label=label_base,
                 color=color,
                 linewidth=2.5,
                 marker='o',
                 markersize=6,
                 markerfacecolor='white',
                 markeredgewidth=1,
                 alpha=0.9)

    plt.xlabel('Communication Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(20, 100) if max_val > 80 else plt.ylim(0, 80)
    plt.xticks(np.arange(0, num_points * interval + 1, interval))
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=14)
    plt.grid(True, ls="--", alpha=0.3)
    if not nolegend:
        plt.legend(loc='lower right', fontsize=16)

    # 保存
    filename = f"{output_prefix}_shaded_comparison.pdf"
    output_filename = output_dir / filename
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

# --- 主执行逻辑更新 ---

PLOT_TYPES = {
    'variance': plot_test_accuracy_with_variance,
    'line': plot_test_accuracy_line,
    'error_bar': plot_test_accuracy_error_bar,
    'shaded': plot_test_accuracy_shaded,
}

def main_plot_script(log_paths: List[Path], legends: List[str], output_dir: Path, output_prefix: str = 'Curve', window_size: int = 6, plot_type: str = 'line', nolegend: bool = False):
    """
    Main function to load data from multiple paths and generate comparative plots,
    saving them to the specified directory using pathlib.
    plot_type: 'variance' | 'line' | 'error_bar' | 'shaded'
    """
    if not log_paths:
        print("Error: Please provide at least one JSONL file path.", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    data_frames = load_data_from_paths(log_paths)
    if not data_frames:
        print("No valid data to plot. Exiting.", file=sys.stderr)
        return

    plot_fn = PLOT_TYPES.get(plot_type)
    if plot_fn is None:
        print(f"Error: unknown plot_type '{plot_type}', choose from {list(PLOT_TYPES.keys())}", file=sys.stderr)
        return

    if plot_type == 'variance':
        plot_fn(data_frames, output_dir, output_prefix, window_size=window_size, legends=legends, nolegend=nolegend)
    else:
        plot_fn(data_frames, output_dir, output_prefix, interval=100, legends=legends, nolegend=nolegend)

if __name__ == '__main__':
    # Example (illustrative; use the CLI args below instead):
    # log_paths = [
    #     Path('result/STCS_cifar10_niid_n[100]_c[10]_s[5]_t1000_k[3]/details.jsonl'),
    # ]
    # legends = ['STCS']
    # output_dir = Path('./result/')
    # main_plot_script(log_paths, legends, output_dir, output_prefix='Curve', window_size=6)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save visualization or summary results.")
    parser.add_argument("--plot-type", type=str, default="line",
                        choices=list(PLOT_TYPES.keys()),
                        help="Plot style: variance, line (default), error_bar, shaded")
    parser.add_argument("--sort-by-h", action="store_true",
                        help="Sort legends by H=N value (for recall comparison)")
    parser.add_argument("--nolegend", action="store_true",
                        help="Omit legend from the plot")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    legends, log_paths = parse_results_from_stdin(sort_by_h=args.sort_by_h)
    log_paths = [Path(p) for p in log_paths]
    main_plot_script(log_paths, legends, output_dir, output_prefix='Curve', window_size=80, plot_type=args.plot_type, nolegend=args.nolegend)

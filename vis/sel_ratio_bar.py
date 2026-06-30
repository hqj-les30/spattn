import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

NAME_MAP = {
    'clsrand': 'MS',
    'onoff': 'RA',
    'niid': 'Label Skew',
    'diri': 'Dirichlet',
}


def parse_results_from_stdin():
    methods, paths = [], []
    pattern = re.compile(r"^(.*?):\s*saved results to\s*(.*)$")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            methods.append(m.group(1).strip())
            paths.append(m.group(2).strip())
    return methods, paths


def load_data(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        content = f.read()
    return pd.read_json(io.StringIO(content), lines=True)


def get_group_info(method: str, path: str):
    c_match = re.search(r'_c\[(\d+)\]_', path)
    cluster_size = int(c_match.group(1)) if c_match else None

    # SelRatio_cifar10_1500_10_<sampler>_<setting>_expN
    m = re.search(r'SelRatio_\w+_\d+_\d+_([a-z]+)_([a-z]+)_', path)
    sampler = m.group(1) if m else "unknown"
    setting = m.group(2) if m else "unknown"

    r_match = re.search(r'_(\d+)%$', method)
    ratio = int(r_match.group(1)) if r_match else None

    return cluster_size, sampler, setting, ratio


def fmt_name(s):
    return NAME_MAP.get(s, s)


def plot_grouped_bar(means: Dict, stds: Dict, groups: List[Tuple], ratios: List[int],
                     display_labels: List[str], output_path: Path, ylabel: str):
    n_groups = len(groups)
    n_bars = len(ratios)
    group_width = 0.7
    bar_width = group_width / n_bars
    x = np.arange(n_groups)
    colors = plt.get_cmap('tab10').colors
    hatches = ['///', '...', '\\\\\\']

    fig, ax = plt.subplots(figsize=(8, 3.5))

    for j, ratio in enumerate(ratios):
        vals = [means.get((grp, ratio), 0) for grp in groups]
        errs = [stds.get((grp, ratio), 0) for grp in groups]
        offset = (j - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, yerr=errs,
                      capsize=3, error_kw={'elinewidth': 1},
                      color=colors[j % len(colors)], edgecolor='black', linewidth=0.8,
                      hatch=hatches[j % len(hatches)])

        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis='y', labelsize=12)

    # Ratio labels in black, positioned just below the axis
    for i, grp in enumerate(groups):
        for j, ratio in enumerate(ratios):
            offset = (j - n_bars / 2 + 0.5) * bar_width
            ax.text(x[i] + offset, -1.5, f'{ratio}%',
                    ha='center', va='top', fontsize=9, color='black')

    # Group names as xtick labels, pushed further down
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.tick_params(axis='x', pad=20)

    ax.grid(axis='y', ls='--', alpha=0.4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Selection ratio grouped bar chart")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="test_accuracy",
                        help="Column name from details.jsonl (default: test_accuracy)")
    parser.add_argument("--ylabel", type=str, default="Accuracy (%)",
                        help="Y-axis label")
    parser.add_argument("--last-n", type=int, default=50,
                        help="Use last N epochs to compute mean (default: 50)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    methods, paths = parse_results_from_stdin()
    if not paths:
        print("No data found in stdin.", file=sys.stderr)
        return

    means, stds_data = {}, {}
    group_labels = {}
    all_ratios = set()

    for method, path in zip(methods, paths):
        cluster_size, sampler, setting, ratio = get_group_info(method, path)
        if ratio is None:
            continue
        print(f"[DEBUG] sampler={sampler}→{fmt_name(sampler)}, setting={setting}→{fmt_name(setting)}, c={cluster_size}, ratio={ratio}", file=sys.stderr)
        df = load_data(path)
        tail = df[args.metric].iloc[-args.last_n:]
        group_key = (sampler, setting, cluster_size)
        if sampler == 'onoff':
            group_labels[group_key] = f"{fmt_name(sampler)}, {fmt_name(setting)}\n$p={cluster_size / 100}$"
        else:
            group_labels[group_key] = f"{fmt_name(sampler)}, {fmt_name(setting)}\n$|\\mathbb{{C}}^t|={cluster_size}$"
        all_ratios.add(ratio)
        means[(group_key, ratio)] = tail.mean()
        stds_data[(group_key, ratio)] = tail.std()

    groups = sorted(group_labels.keys())
    display_labels = [group_labels[g] for g in groups]
    ratios = sorted(all_ratios)

    plot_grouped_bar(means, stds_data, groups, ratios, display_labels,
                     output_dir / "selection_ratio_bar.pdf",
                     args.ylabel)


if __name__ == '__main__':
    main()

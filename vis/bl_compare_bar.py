import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

NAME_MAP = {
    'clsrand': 'MS',
    'onoff': 'RA',
    'niid': 'Label Skew',
    'diri': 'Dirichlet',
}

DATASET_NAMES = {
    'cifar': 'CIFAR-10',
    'fashion': 'Fashion-MNIST',
    'har': 'HAR',
}


H_PATTERN = re.compile(r'H=(\d+)')


def method_sort_key(method, sort_by_h=False):
    if sort_by_h:
        m = H_PATTERN.search(method)
        return int(m.group(1)) if m else 0
    return (0, '') if method == 'Ours' else (1, method)


def load_data(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        content = f.read()
    return pd.read_json(io.StringIO(content), lines=True)


def parse_stdout(stdout_path: str):
    """Parse stdout.log, return list of (method, details_path).
    Handles concatenated lines where multiple entries appear on one line.
    """
    pattern = re.compile(r'([A-Za-z0-9]+?):\s*saved results to\s*(\S+?/details\.jsonl)')
    entries = []
    with open(stdout_path, 'r') as f:
        content = f.read()
    for m in pattern.finditer(content):
        method = m.group(1).strip()
        path = m.group(2).strip()
        entries.append((method, path))
    return entries


def parse_group_name(img_name: str):
    """Extract dataset, setting, sampler from image name like cifar_diri_clsrand_bs_2."""
    parts = img_name.split('_')
    # har only has dataset + sampler
    if parts[0] == 'har':
        return 'har', 'niid', parts[1]
    # cifar/fashion: dataset_setting_sampler_bs_...
    return parts[0], parts[1], parts[2]


def fmt_name(s):
    return NAME_MAP.get(s, s)


def plot_grouped_bar(means: Dict, stds: Dict, groups: List[Tuple], methods: List[str],
                     display_labels: List[str], output_path: Path, nolegend: bool = False,
                     figsize: Tuple = None, ylim: Tuple = None, group_width: float = 0.65):
    n_groups = len(groups)
    n_bars = len(methods)
    bar_width = group_width / n_bars
    x = np.arange(n_groups)
    colors = plt.get_cmap('tab10').colors
    hatches = ['', '///', '...', '\\\\\\']

    fig, ax = plt.subplots(figsize=figsize or (9, 3.5))

    for j, method in enumerate(methods):
        vals = [means.get((grp, method), 0) for grp in groups]
        errs = [stds.get((grp, method), 0) for grp in groups]
        offset = (j - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, yerr=errs,
                      capsize=3, error_kw={'elinewidth': 1},
                      color=colors[j % len(colors)], edgecolor='black', linewidth=0.8,
                      hatch=hatches[j % len(hatches)],
                      label=method)

        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.grid(axis='y', ls='--', alpha=0.4)
    if not nolegend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=len(methods), fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline compare grouped bar chart")
    parser.add_argument("--last-n", type=int, default=50,
                        help="Use last N epochs to compute mean (default: 50)")
    parser.add_argument("--sort-by-h", action="store_true",
                        help="Sort methods by H=N value (for recall comparison)")
    parser.add_argument("--nolegend", action="store_true",
                        help="Omit legend from the plot")
    args = parser.parse_args()

    # (img_name, source_dirs) — independent-seed reps (s4*) only; old rep1/rep2
    # share the same seed and are NOT independent, so they are excluded here.
    entries = [
        ('cifar_diri_clsrand', [
            'BLCompare_cifar10_1500_10_5_clsrand_diri_s41',
            'BLCompare_cifar10_1500_10_5_clsrand_diri_s43',
            'BLCompare_cifar10_1500_10_5_clsrand_diri_s44',
        ]),
        ('cifar_diri_onoff', [
            'BLCompare_cifar10_1500_10_5_onoff_diri_s41',
            'BLCompare_cifar10_1500_10_5_onoff_diri_s43',
            'BLCompare_cifar10_1500_10_5_onoff_diri_s44',
        ]),
        ('cifar_niid_clsrand', [
            'BLCompare_cifar10_1500_10_5_clsrand_niid_s41',
            'BLCompare_cifar10_1500_10_5_clsrand_niid_s43',
            'BLCompare_cifar10_1500_10_5_clsrand_niid_s44',
        ]),
        ('cifar_niid_onoff', [
            'BLCompare_cifar10_1500_10_5_onoff_niid_s41',
            'BLCompare_cifar10_1500_10_5_onoff_niid_s43',
            'BLCompare_cifar10_1500_10_5_onoff_niid_s44',
        ]),
        ('fashion_diri_clsrand', [
            'BLCompare_fashion_600_10_5_clsrand_diri_s41',
            'BLCompare_fashion_600_10_5_clsrand_diri_s43',
            'BLCompare_fashion_600_10_5_clsrand_diri_s44',
        ]),
        ('fashion_diri_onoff', [
            'BLCompare_fashion_600_10_5_onoff_diri_s41',
            'BLCompare_fashion_600_10_5_onoff_diri_s43',
            'BLCompare_fashion_600_10_5_onoff_diri_s44',
        ]),
        ('fashion_niid_clsrand', [
            'BLCompare_fashion_600_10_5_clsrand_niid_s41',
            'BLCompare_fashion_600_10_5_clsrand_niid_s43',
            'BLCompare_fashion_600_10_5_clsrand_niid_s44',
        ]),
        ('fashion_niid_onoff', [
            'BLCompare_fashion_600_10_5_onoff_niid_s41',
            'BLCompare_fashion_600_10_5_onoff_niid_s43',
            'BLCompare_fashion_600_10_5_onoff_niid_s44',
        ]),
        ('har_clsrand_niid', [
            'BLCompare_har_600_7_3_clsrand_niid_s41',
            'BLCompare_har_600_7_3_clsrand_niid_s42',
        ]),
        ('har_onoff_niid', [
            'BLCompare_har_600_10_3_onoff_niid_s41',
            'BLCompare_har_600_10_3_onoff_niid_s42',
        ]),
    ]

    output_dir = Path('vis/images')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all entries into structured data
    parsed = []  # (dataset, setting, sampler, src_dir)
    for img_name, src_dir in entries:
        dataset, setting, sampler = parse_group_name(img_name)
        parsed.append((dataset, setting, sampler, src_dir))

    # Load all data: key=(dataset, setting, sampler, method) -> (mean, std)
    all_means, all_stds = {}, {}
    all_methods = set()
    run_stats = {}  # key -> {'means': [...], 'stds': [...]}
    for dataset, setting, sampler, src_dirs in parsed:
        for src_dir in src_dirs:
            stdout_path = f'result/{src_dir}/stdout.log'
            if not Path(stdout_path).exists():
                print(f"SKIP: {stdout_path} not found", file=sys.stderr)
                continue
            for method, details_path in parse_stdout(stdout_path):
                if not Path(details_path).exists():
                    print(f"SKIP: {details_path} not found", file=sys.stderr)
                    continue
                df = load_data(details_path)
                tail = df['test_accuracy'].iloc[-args.last_n:]
                key = (dataset, setting, sampler, method)
                run_stats.setdefault(key, {'means': [], 'stds': []})
                run_stats[key]['means'].append(tail.mean())
                run_stats[key]['stds'].append(tail.std())
                all_methods.add(method)

    for key, stats in run_stats.items():
        all_means[key] = np.mean(stats['means'])
        all_stds[key] = np.mean(stats['stds'])

    methods = sorted(all_methods, key=lambda m: method_sort_key(m, sort_by_h=args.sort_by_h))

    # Chart 1: Label Skew (cifar + fashion, grouped by dataset then sampler)
    for chart_setting, chart_name in [('niid', 'labelskew'), ('diri', 'dirichlet')]:
        means, stds = {}, {}
        group_keys, display_labels = [], []
        for dataset in ['cifar', 'fashion']:
            for sampler in ['clsrand', 'onoff']:
                group_key = (dataset, sampler)
                has_data = False
                for m in methods:
                    k = (dataset, chart_setting, sampler, m)
                    if k in all_means:
                        means[(group_key, m)] = all_means[k]
                        stds[(group_key, m)] = all_stds[k]
                        has_data = True
                if has_data:
                    group_keys.append(group_key)
                    display_labels.append(f'{DATASET_NAMES[dataset]}, {fmt_name(sampler)}')

        if group_keys:
            output_path = output_dir / f'{chart_name}_bl_bar.pdf'
            plot_grouped_bar(means, stds, group_keys, methods, display_labels, output_path, nolegend=args.nolegend)
            print(f'OK: {output_path}')

    # Chart 3: HAR (only sampler grouping)
    means, stds = {}, {}
    group_keys, display_labels = [], []
    for sampler in ['clsrand', 'onoff']:
        group_key = (sampler,)
        has_data = False
        for m in methods:
            k = ('har', 'niid', sampler, m)
            if k in all_means:
                means[(group_key, m)] = all_means[k]
                stds[(group_key, m)] = all_stds[k]
                has_data = True
        if has_data:
            group_keys.append(group_key)
            display_labels.append(fmt_name(sampler))

    if group_keys:
        output_path = output_dir / 'har_bl_bar.pdf'
        # HAR has only two groups (MS/RA); use narrower bars and a mildly zoomed
        # y-axis (50-100) so the bars are not stretched too wide across the canvas.
        plot_grouped_bar(means, stds, group_keys, methods, display_labels, output_path,
                         nolegend=args.nolegend, group_width=0.30, ylim=(50, 100))
        print(f'OK: {output_path}')


if __name__ == '__main__':
    main()

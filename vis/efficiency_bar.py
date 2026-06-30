#!/usr/bin/env python3
"""
Efficiency-to-target grouped bar charts (curvevis/bl_compare_bar style).

Two figures per dataset family:
  - cumulative uplink / client to reach Acc_t  (linear, GB)
  - cumulative server FLOPs to reach Acc_t       (log, GFLOPs)
Each figure: the 4 CIFAR scenarios on the x-axis, 5 methods as grouped bars.
No title, no legend, no y-axis label (legend is a separate bar image).
Methods sorted/colored/hatched per the curvevis standard (STCS first, tab10).
Methods that never reach Acc_t use full-run cumulative + a red dagger.

Acc_t = median of all runs' final accuracies (last --final-window epochs) within
each scenario. Target round = first round where the seed-mean accuracy curve,
smoothed by a --smooth-window rolling mean, reaches Acc_t.

Usage (from the repo root):
    python vis/efficiency_bar.py \
        --input-dir result/efficiency_csv --datasets cifar10 \
        --output-dir vis/images
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# curvevis/bl_compare_bar standard: STCS first, then alphabetical
METHODS_SORTED = ['STCS', 'F3AST', 'FedAWAC', 'FedProx', 'HA']
COLORS = plt.get_cmap('tab10').colors
HATCHES = ['', '///', '...', '\\\\\\']
NAME_MAP = {'niid': 'Label Skew', 'diri': 'Dirichlet',
            'clsrand': 'MS', 'onoff': 'RA'}


def scen_label(scen):
    """p.stem like efficiency_cifar10_diri_clsrand -> 'MS, Dirichlet'."""
    # Extract SCN part from 'efficiency_{dataset}_{setting}_{sampler}_C{cluster}_K{selection}'
    p = scen.rsplit('_', 5)  # ['efficiency', 'cifar10', 'diri', 'clsrand', 'C10', 'K5']
    if len(p) >= 4:
        sampler = NAME_MAP.get(p[3], p[3])
        setting = NAME_MAP.get(p[2], p[2])
        return f"{sampler}, {setting}"
    return scen


def scen_sort_key(scen):
    p = scen.split('_')
    setting_o = {'niid': 0, 'diri': 1}.get(p[1], 9)
    sampler_o = {'clsrand': 0, 'onoff': 1}.get(p[2], 9)
    return (setting_o, sampler_o)


def read_group(csv_path):
    """{method: {seed_dir: [(round, acc, uplink_total, server_total)]}}."""
    runs = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            runs[row['method']].setdefault(row['seed_dir'], []).append(
                (int(row['round']), float(row['test_accuracy']),
                 int(row['uplink_total_bytes']), int(row['server_total_flops'])))
    for m in runs:
        for s in runs[m]:
            runs[m][s].sort(key=lambda x: x[0])
    return runs


def final_accs(runs, fw):
    out = []
    for m in runs:
        for s in runs[m]:
            accs = [x[1] for x in runs[m][s]]
            out.append(float(np.mean(accs[-fw:])) if accs else 0.0)
    return out


def cumulative_to_target(runs, method, target, w):
    """(uplink_bytes, server_flops, reached, r_star) for one method (seed-mean)."""
    if method not in runs or not runs[method]:
        return 0.0, 0.0, False, 0
    sa = [np.array([x[1] for x in runs[method][s]], float) for s in runs[method]]
    su = [np.array([x[2] for x in runs[method][s]], float) for s in runs[method]]
    ss = [np.array([x[3] for x in runs[method][s]], float) for s in runs[method]]
    T = min(len(c) for c in sa)
    acc = np.stack([c[:T] for c in sa]).mean(axis=0)
    up = np.stack([c[:T] for c in su]).mean(axis=0)
    sf = np.stack([c[:T] for c in ss]).mean(axis=0)
    if w > 1:
        acc = np.convolve(acc, np.ones(w) / w, mode='full')[:T]
    idx = np.where(acc >= target)[0]
    if len(idx) == 0:
        r = T - 1; reached = False
    else:
        r = int(idx[0]); reached = True
    return float(up[:r + 1].sum()), float(sf[:r + 1].sum()), reached, r


def compute_scenarios(input_dir, datasets, fw, sw):
    """Return ordered [scen], {scen: Acc_t}, {scen: {method: (up, sf, reached, r)}}."""
    scens, targets, data = [], {}, {}

    # Target accuracies for different datasets
    TARGET_ACCURACY = {
        'cifar10': {'diri': 70.0, 'niid': 40.0},
        'fashion': {'diri': 80.0, 'niid': 60.0},
    }

    for ds in datasets:
        for csvf in sorted(Path(input_dir).glob(f'efficiency_{ds}_*.csv')):
            scen = csvf.stem
            runs = read_group(csvf)

            # Parse setting and sampler from scen name
            parts = scen.rsplit('_', 5)
            if len(parts) >= 4:
                setting = parts[2]  # diri or niid
                sampler = parts[3]  # clsrand or onoff

                # Use specified target accuracy
                target = TARGET_ACCURACY[ds].get(setting, 70.0)

                scens.append(scen)
                targets[scen] = target
                data[scen] = {m: cumulative_to_target(runs, m, target, sw)
                              for m in METHODS_SORTED if m in runs}
    scens.sort(key=scen_sort_key)
    return scens, targets, data


def plot_grouped(scens, data, metric, unit, scale, log, out_dir, fname):
    """metric in {'up','sf'}; values = data[scen][method][idx] / scale."""
    idx = 0 if metric == 'up' else 1

    # Filter out methods with very low server FLOPs (F3AST, FedProx)
    methods_to_plot = METHODS_SORTED
    if metric == 'sf':
        methods_to_plot = [m for m in METHODS_SORTED if m not in ('F3AST', 'FedProx')]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    n_groups = len(scens)
    n_bars = len(methods_to_plot)
    group_width = 0.7
    bw = group_width / n_bars
    x = np.arange(n_groups)
    for j, m in enumerate(methods_to_plot):
        vals, dags = [], []
        for s in scens:
            if m in data[s]:
                v = data[s][m][idx] / scale
                dags.append(not data[s][m][2])
            else:
                v, dags = 0.0, False
            vals.append(v)
        off = (j - n_bars / 2 + 0.5) * bw
        bars = ax.bar(x + off, vals, bw, color=COLORS[j % len(COLORS)],
                      edgecolor='black', linewidth=0.7,
                      hatch=HATCHES[j % len(HATCHES)])
        for b, v, dg in zip(bars, vals, dags):
            label = f'{v:.1f}$\\dagger$' if dg else (f'{v:.1f}' if v < 100 else f'{v:.0f}')
            ax.text(b.get_x() + b.get_width() / 2, v, label,
                    ha='center', va='bottom', fontsize=6.5,
                    color='red' if dg else 'black')
    if log:
        ax.set_yscale('log')

    # Add y-axis unit labels (no title, just units)
    ylabel = 'GB' if metric == 'up' else 'GFLOPs'
    ax.text(0.02, 0.98, ylabel, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontweight='normal')

    ax.set_xticks(x)
    ax.set_xticklabels([scen_label(s) for s in scens], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', ls='--', alpha=0.35)
    plt.tight_layout()
    out = Path(out_dir) / fname
    plt.savefig(out, dpi=400, bbox_inches='tight')
    plt.close()
    print(f'OK: {out}')


def plot_legend_bar(out_dir, fname):
    fig, ax = plt.subplots(figsize=(8, 0.4))
    ax.set_axis_off()
    handles = [Patch(facecolor=COLORS[j % len(COLORS)], edgecolor='black',
                     hatch=HATCHES[j % len(HATCHES)], label=m)
               for j, m in enumerate(METHODS_SORTED)]
    fig.legend(handles=handles, loc='center', ncol=len(METHODS_SORTED),
               fontsize=12, frameon=False)
    out = Path(out_dir) / fname
    plt.savefig(out, dpi=500, bbox_inches='tight')
    plt.close()
    print(f'OK: {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', default='result/efficiency_csv')
    ap.add_argument('--datasets', nargs='+', default=['cifar10'])
    ap.add_argument('--output-dir', default='vis/images')
    ap.add_argument('--final-window', type=int, default=50)
    ap.add_argument('--smooth-window', type=int, default=50)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    scens, targets, data = compute_scenarios(
        args.input_dir, args.datasets, args.final_window, args.smooth_window)
    if not scens:
        print('No scenarios found.'); return
    tag = '_'.join(args.datasets)
    # uplink (linear, GB), server FLOPs (log, GFLOPs)
    plot_grouped(scens, data, 'up', 'GB', 1e9, False, args.output_dir,
                 f'efficiency_uplink_{tag}.pdf')
    plot_grouped(scens, data, 'sf', 'GFLOPs', 1e9, True, args.output_dir,
                 f'efficiency_serverflops_{tag}.pdf')
    plot_legend_bar(args.output_dir, f'efficiency_legend_{tag}.pdf')
    print('\nAcc_t per scenario:')
    for s in scens:
        print(f'  {scen_label(s):<20} Acc_t={targets[s]:.3f}')


if __name__ == '__main__':
    main()

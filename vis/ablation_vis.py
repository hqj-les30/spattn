#!/usr/bin/env python3
"""
Ablation visualization: grouped bar chart showing last-N epoch mean±std accuracy.
  - Same setting → grouped together
  - Bars for each ablation variant (e.g. with/without id_emb)

Input: same stdin format as curvevis.py
  "Method Legend: saved results to /path/to/details.jsonl"
"""

import sys
import re
import io
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DPI = 500


def parse_stdin():
    """Parse stdin for 'Method Legend: saved results to /path' entries."""
    pattern = re.compile(r"(.+?):\s*saved results to\s*(\S+?/details\.jsonl)")
    entries = []
    content = sys.stdin.read()
    for m in pattern.finditer(content):
        entries.append((m.group(1).strip(), m.group(2).strip()))
    return entries


def load_df(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        remaining = f.read()
    return pd.read_json(io.StringIO(remaining), lines=True)


def parse_entry(legend: str):
    """Extract (setting, variant) from a legend string.

    The variant is the text inside the *last* parenthesised group, e.g.
        'STCS Label Skew (Full)'          -> ('Label Skew', 'Full')
        'STCS Dirichlet (w/o Temporal Agg)' -> ('Dirichlet', 'w/o Temporal Agg')
    If no parens are present the variant defaults to 'Full'.
    """
    m = re.search(r'\(([^()]*)\)\s*$', legend)
    if m:
        variant = m.group(1).strip()
        setting = legend[:m.start()].replace('STCS', '').strip()
    else:
        variant = 'Full'
        setting = legend.replace('STCS', '').strip()
    return setting or 'Full', variant


# preferred left-to-right bar order (Full first); unknown variants appended after
_VARIANT_ORDER = ['Full', 'w/o Temporal Agg', 'w/o Temporal Attn',
                  'w/o Spatial Attn', 'MLP Encoder', 'w/o id_emb']


def variant_sort_key(v: str):
    if v in _VARIANT_ORDER:
        return (0, _VARIANT_ORDER.index(v))
    return (1, v)


def plot_grouped_bar(means: Dict, stds: Dict, groups: List[str], variants: List[str],
                     output_path: Path):
    n_groups = len(groups)
    n_bars = len(variants)
    group_width = 0.25
    bar_width = group_width / n_bars
    x = np.arange(n_groups) * 0.4
    colors = plt.get_cmap('tab10').colors
    hatches = ['', '///', '...', '\\\\\\', 'xxx', '+++']

    fig_w = max(4.5, 1.5 * n_groups * max(n_bars, 2))
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))

    for j, variant in enumerate(variants):
        vals = [means.get((grp, variant), 0) for grp in groups]
        errs = [stds.get((grp, variant), 0) for grp in groups]
        offset = (j - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, yerr=errs,
                      capsize=3, error_kw={'elinewidth': 1},
                      color=colors[j % len(colors)], edgecolor='black', linewidth=0.8,
                      hatch=hatches[j % len(hatches)],
                      label=variant)

        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.grid(axis='y', ls='--', alpha=0.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=len(variants), fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"OK: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ablation grouped bar chart")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--last-n", type=int, default=10,
                        help="Use last N epochs to compute mean (default: 10)")
    parser.add_argument("--delta-vs", type=str, default=None,
                        help="Print Delta (mean acc) of each variant vs this baseline variant (e.g. Full)")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = parse_stdin()
    if not entries:
        print("No results found on stdin.", file=sys.stderr)
        sys.exit(1)

    # Collect data: key=(setting, variant) → (mean, std)
    means, stds = {}, {}
    all_settings, all_variants = [], []

    for legend, path in entries:
        if not Path(path).exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            continue
        df = load_df(path)
        tail = df['test_accuracy'].iloc[-args.last_n:]
        setting, variant = parse_entry(legend)

        means[(setting, variant)] = tail.mean()
        stds[(setting, variant)] = tail.std()

        if setting not in all_settings:
            all_settings.append(setting)
        if variant not in all_variants:
            all_variants.append(variant)

    # deterministic bar order: Full first, then preferred order, then alphabetical
    all_variants = sorted(set(all_variants), key=variant_sort_key)
    all_settings = sorted(set(all_settings))

    # optional Delta-vs summary (Table 2 style), printed to stdout
    if args.delta_vs:
        print(f"\n=== Accuracy (mean over last {args.last_n} epochs) — Delta vs '{args.delta_vs}' ===")
        for setting in all_settings:
            base = means.get((setting, args.delta_vs))
            print(f"\n[{setting}]")
            for variant in all_variants:
                m = means.get((setting, variant))
                if m is None:
                    continue
                delta = (m - base) if base is not None else float('nan')
                delta_str = f"{delta:+.2f}" if base is not None else "  —"
                print(f"  {variant:<20s} {m:6.2f}  Δ={delta_str}")
        print()

    output_path = output_dir / 'ablation_bar.pdf'
    plot_grouped_bar(means, stds, all_settings, all_variants, output_path)

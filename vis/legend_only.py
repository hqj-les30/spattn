#!/usr/bin/env python3
"""
Generate a standalone legend bar image for layout use.
Reads one stdout.log to extract method names, then draws a legend-only PDF.

Usage:
    python vis/legend_only.py --source result/BLCompare_xxx/stdout.log --output vis/images/legend_bar.pdf
"""

import re
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_methods(stdout_path: str):
    pattern = re.compile(r'^(.*?):\s*saved results to\s*(.*)$')
    methods = []
    with open(stdout_path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                methods.append(m.group(1).strip())
    # Same sort order as curvevis.py: "STCS" first, then alphabetical
    methods.sort(key=lambda m: (0, "") if m == "STCS" else (1, m))
    return methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to a stdout.log')
    parser.add_argument('--output', type=str, default='vis/images/legend_bar.pdf')
    args = parser.parse_args()

    methods = parse_methods(args.source)
    if not methods:
        print("No methods found.", file=sys.stderr)
        sys.exit(1)

    colors = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']

    fig, ax = plt.subplots(figsize=(12, 0.3))
    ax.set_axis_off()

    handles = []
    for i, method in enumerate(methods):
        h, = ax.plot([], [], color=colors[i % len(colors)],
                     marker=markers[i % len(markers)],
                     markersize=7, linewidth=2,
                     markeredgewidth=1.5, markerfacecolor='white',
                     label=method)
        handles.append(h)

    fig.legend(handles=handles, loc='center', ncol=len(methods),
               fontsize=14, frameon=False)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=500, bbox_inches='tight')
    plt.close()
    print(f'Saved legend to {out}')


if __name__ == '__main__':
    main()

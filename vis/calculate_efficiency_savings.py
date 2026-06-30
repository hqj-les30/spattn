#!/usr/bin/env python3
"""
Calculate uplink communication savings relative to the best baseline.
"""
import csv
import statistics
from pathlib import Path
from collections import defaultdict
import numpy as np

METHODS = ['Ours', 'FedProx', 'HA', 'FedAWAC', 'F3AST']

def read_group(csv_path):
    """Read CSV, return {method: {seed_dir: [(round, acc, uplink_total)]}}."""
    runs = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            runs[row['method']].setdefault(row['seed_dir'], []).append(
                (int(row['round']), float(row['test_accuracy']),
                 int(row['uplink_total_bytes'])))
    for m in runs:
        for s in runs[m]:
            runs[m][s].sort(key=lambda x: x[0])
    return runs

def cumulative_to_target(runs, method, target, w=50):
    """Calculate cumulative uplink to reach target accuracy."""
    if method not in runs or not runs[method]:
        return 0.0, False

    sa = [np.array([x[1] for x in runs[method][s]], float) for s in runs[method]]
    su = [np.array([x[2] for x in runs[method][s]], float) for s in runs[method]]

    T = min(len(c) for c in sa)
    acc = np.stack([c[:T] for c in sa]).mean(axis=0)
    up = np.stack([c[:T] for c in su]).mean(axis=0)

    if w > 1:
        acc = np.convolve(acc, np.ones(w) / w, mode='full')[:T]

    idx = np.where(acc >= target)[0]
    if len(idx) == 0:
        r = T - 1
        reached = False
    else:
        r = int(idx[0])
        reached = True

    return float(up[:r + 1].sum()), reached

def analyze_dataset(dataset, input_dir):
    """Analyze one dataset with specified target accuracies."""
    results = {}

    TARGET_ACCURACY = {
        'cifar10': {'diri': 70.0, 'niid': 40.0},
        'fashion': {'diri': 80.0, 'niid': 60.0},
    }

    for csvf in sorted(Path(input_dir).glob(f'efficiency_{dataset}_*.csv')):
        scen = csvf.stem
        parts = scen.rsplit('_', 5)
        setting = parts[2]
        sampler = parts[3]

        runs = read_group(csvf)
        target = TARGET_ACCURACY[dataset][setting]

        uplink_by_method = {}
        for m in METHODS:
            if m in runs:
                up, reached = cumulative_to_target(runs, m, target)
                uplink_by_method[m] = (up, reached)

        results[(setting, sampler)] = {
            'target': target,
            'uplink': uplink_by_method
        }

    return results

def main():
    input_dir = Path('result/efficiency_csv')

    print("=" * 90)
    print("CIFAR-10 (Target: Dirichlet=70%, Label Skew=40%)")
    print("=" * 90)
    cifar_results = analyze_dataset('cifar10', input_dir)

    cifar_diri_savings = []
    cifar_niid_savings = []

    for (setting, sampler), data in sorted(cifar_results.items()):
        print(f"\n{setting.upper()} + {sampler.upper()}:")
        print(f"  Target Accuracy: {data['target']:.1f}%")

        ours_up, ours_reached = data['uplink'].get('Ours', (0, False))
        ours_gb = ours_up / 1e9

        print(f"  Uplink (GB):")
        baseline_ups = []
        for m in METHODS[1:]:  # Skip Ours
            if m in data['uplink']:
                up, reached = data['uplink'][m]
                up_gb = up / 1e9
                baseline_ups.append((m, up_gb, reached))
                print(f"    {m:10s}: {up_gb:6.2f} GB {'†' if not reached else ''}")

        if baseline_ups and ours_up > 0:
            # Find best baseline (minimum uplink)
            best_baseline = min(baseline_ups, key=lambda x: x[1])
            best_name, best_up, best_reached = best_baseline
            savings = (best_up - ours_gb) / best_up * 100

            print(f"\n  Ours: {ours_gb:.2f} GB {'†' if not ours_reached else ''}")
            print(f"  Best baseline: {best_name} = {best_up:.2f} GB")
            print(f"  Savings vs best baseline: {savings:.1f}%")

            if setting == 'diri':
                cifar_diri_savings.append(savings)
            else:
                cifar_niid_savings.append(savings)

    print("\n" + "=" * 90)
    print("Fashion-MNIST (Target: Dirichlet=80%, Label Skew=60%)")
    print("=" * 90)
    fashion_results = analyze_dataset('fashion', input_dir)

    fashion_diri_savings = []
    fashion_niid_savings = []

    for (setting, sampler), data in sorted(fashion_results.items()):
        print(f"\n{setting.upper()} + {sampler.upper()}:")
        print(f"  Target Accuracy: {data['target']:.1f}%")

        ours_up, ours_reached = data['uplink'].get('Ours', (0, False))
        ours_gb = ours_up / 1e9

        print(f"  Uplink (GB):")
        baseline_ups = []
        for m in METHODS[1:]:  # Skip Ours
            if m in data['uplink']:
                up, reached = data['uplink'][m]
                up_gb = up / 1e9
                baseline_ups.append((m, up_gb, reached))
                print(f"    {m:10s}: {up_gb:6.2f} GB {'†' if not reached else ''}")

        if baseline_ups and ours_up > 0:
            # Find best baseline (minimum uplink)
            best_baseline = min(baseline_ups, key=lambda x: x[1])
            best_name, best_up, best_reached = best_baseline
            savings = (best_up - ours_gb) / best_up * 100

            print(f"\n  Ours: {ours_gb:.2f} GB {'†' if not ours_reached else ''}")
            print(f"  Best baseline: {best_name} = {best_up:.2f} GB")
            print(f"  Savings vs best baseline: {savings:.1f}%")

            if setting == 'diri':
                fashion_diri_savings.append(savings)
            else:
                fashion_niid_savings.append(savings)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY: Average Savings vs Best Baseline")
    print("=" * 90)

    if cifar_diri_savings:
        print(f"CIFAR-10 Dirichlet: {np.mean(cifar_diri_savings):.1f}%")
    if cifar_niid_savings:
        print(f"CIFAR-10 Label Skew: {np.mean(cifar_niid_savings):.1f}%")
    if fashion_diri_savings:
        print(f"Fashion-MNIST Dirichlet: {np.mean(fashion_diri_savings):.1f}%")
    if fashion_niid_savings:
        print(f"Fashion-MNIST Label Skew: {np.mean(fashion_niid_savings):.1f}%")

    overall_diri = np.mean(cifar_diri_savings + fashion_diri_savings) if (cifar_diri_savings + fashion_diri_savings) else 0
    overall_niid = np.mean(cifar_niid_savings + fashion_niid_savings) if (cifar_niid_savings + fashion_niid_savings) else 0

    print(f"\nOverall Dirichlet: {overall_diri:.1f}%")
    print(f"Overall Label Skew: {overall_niid:.1f}%")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Calculate server FLOPs savings vs FedAWAC and HA-EdgeFlow by setting.
"""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

METHODS = ['Ours', 'FedProx', 'HA', 'FedAWAC', 'F3AST']

def read_group(csv_path):
    """Read CSV, return {method: {seed_dir: [(round, acc, server_flops)]}}."""
    runs = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            runs[row['method']].setdefault(row['seed_dir'], []).append(
                (int(row['round']), float(row['test_accuracy']),
                 int(row['server_total_flops'])))
    for m in runs:
        for s in runs[m]:
            runs[m][s].sort(key=lambda x: x[0])
    return runs

def cumulative_to_target(runs, method, target, w=50):
    """Calculate cumulative server FLOPs to reach target accuracy."""
    if method not in runs or not runs[method]:
        return 0.0, False

    sa = [np.array([x[1] for x in runs[method][s]], float) for s in runs[method]]
    ss = [np.array([x[2] for x in runs[method][s]], float) for s in runs[method]]

    T = min(len(c) for c in sa)
    acc = np.stack([c[:T] for c in sa]).mean(axis=0)
    sf = np.stack([c[:T] for c in ss]).mean(axis=0)

    if w > 1:
        acc = np.convolve(acc, np.ones(w) / w, mode='full')[:T]

    idx = np.where(acc >= target)[0]
    if len(idx) == 0:
        r = T - 1
        reached = False
    else:
        r = int(idx[0])
        reached = True

    return float(sf[:r + 1].sum()), reached

TARGET_ACCURACY = {
    'cifar10': {'diri': 70.0, 'niid': 40.0},
    'fashion': {'diri': 80.0, 'niid': 60.0},
}

def main():
    input_dir = Path('result/efficiency_csv')

    print("Server FLOPs Savings vs FedAWAC and HA-EdgeFlow")
    print("=" * 90)

    diri_savings = []
    niid_savings = []

    for dataset in ['cifar10', 'fashion']:
        print(f"\n{dataset.upper()}:")
        for setting in ['diri', 'niid']:
            target = TARGET_ACCURACY[dataset][setting]
            print(f"  {setting.upper()} (Target: {target}%):")

            setting_savings = []

            for sampler in ['clsrand', 'onoff']:
                csvf = Path(input_dir) / f'efficiency_{dataset}_{setting}_{sampler}_C10_K5.csv'
                if not csvf.exists():
                    continue

                runs = read_group(csvf)

                ours_sf, ours_reached = cumulative_to_target(runs, 'Ours', target)
                fedawac_sf, fedawac_reached = cumulative_to_target(runs, 'FedAWAC', target)
                ha_sf, ha_reached = cumulative_to_target(runs, 'HA', target)

                ours_gf = ours_sf / 1e9
                fedawac_gf = fedawac_sf / 1e9
                ha_gf = ha_sf / 1e9

                print(f"    {sampler.upper():6s}:")
                print(f"      Ours:       {ours_gf:.2f} GFLOPs {'†' if not ours_reached else ''}")
                print(f"      FedAWAC:    {fedawac_gf:.2f} GFLOPs {'†' if not fedawac_reached else ''}")
                print(f"      HA:         {ha_gf:.2f} GFLOPs {'†' if not ha_reached else ''}")

                # Calculate savings vs each baseline
                if ours_sf > 0:
                    if fedawac_sf > 0:
                        fedawac_saving = (fedawac_sf - ours_sf) / fedawac_sf * 100
                        print(f"      Savings vs FedAWAC: {fedawac_saving:.1f}%")
                        if fedawac_saving > 0:
                            setting_savings.append(fedawac_saving)

                    if ha_sf > 0:
                        ha_saving = (ha_sf - ours_sf) / ha_sf * 100
                        print(f"      Savings vs HA:      {ha_saving:.1f}%")
                        if ha_saving > 0:
                            setting_savings.append(ha_saving)

            if setting_savings:
                avg = np.mean(setting_savings)
                print(f"    Average {setting.upper()} savings: {avg:.1f}%")

                if setting == 'diri':
                    diri_savings.extend(setting_savings)
                else:
                    niid_savings.extend(setting_savings)

    print("\n" + "=" * 90)
    print("SUMMARY: Average Server FLOPs Savings")
    print("=" * 90)
    print(f"Dirichlet:      {np.mean(diri_savings):.1f}%")
    print(f"Label Skew:     {np.mean(niid_savings):.1f}%")
    print(f"Overall:        {np.mean(diri_savings + niid_savings):.1f}%")

if __name__ == '__main__':
    main()
#!/usr/bin/env bash

# Heterogeneity comparison experiment
# Runs a single method across different data heterogeneity settings:
#   Dirichlet:  alpha = 0.1, 0.5
#   Label Skew: num_shards = 2, 5
#
# Usage:
#   bash exp/hetero_compare.sh -a Ours -d cifar10 -g 0,1,2 -t 1500 -c 10 -s 5

# ---- Defaults ----
g="0,1,2,3"
t=50
c=10
s=5
n=100
k=3
z="clsrand"

# ---- Parse args ----
while getopts "a:d:g:t:c:s:n:k:z:" opt; do
  case $opt in
    a) method=$OPTARG ;;
    d) dataset=$OPTARG ;;
    g) g=$OPTARG ;;
    t) t=$OPTARG ;;
    c) c=$OPTARG ;;
    s) s=$OPTARG ;;
    n) n=$OPTARG ;;
    k) k=$OPTARG ;;
    z) z=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

if [ -z "$method" ] || [ -z "$dataset" ]; then
    echo "Usage: $0 -a METHOD -d DATASET [-g GPUs] [-t T] [-c C] [-s S] [-n N] [-k K] [-z Z]"
    exit 1
fi

# ---- Dirichlet experiments ----
for alpha in 0.1 0.5; do
  outdir="./result/HeteroCompare_${dataset}_${t}_${c}_${s}_${method}_diri_a${alpha}"
  mkdir -p "$outdir"
  outfile="$outdir/stdout.log"
  : > "$outfile"

  echo "=== Dirichlet alpha=$alpha ==="
  echo "F3AST ${dataset} diri alpha=${alpha}" >> "$outfile"
  python main.py -a "$method" -d "$dataset" -g "$g" -t "$t" -c "$c" -s "$s" -n "$n" -k "$k" -z "$z" \
      --setting diri --alpha "$alpha" -p "$outdir" \
      >> "$outfile" &

  wait

  cat "$outfile" | python vis/curvevis.py --output-dir "$outdir" --nolegend
done

# ---- Label Skew experiments ----
for ns in 2 5; do
  outdir="./result/HeteroCompare_${dataset}_${t}_${c}_${s}_${method}_niid_ns${ns}"
  mkdir -p "$outdir"
  outfile="$outdir/stdout.log"
  : > "$outfile"

  echo "=== Label Skew num_shards=$ns ==="
  echo "F3AST ${dataset} niid num_shards=${ns}" >> "$outfile"
  python main.py -a "$method" -d "$dataset" -g "$g" -t "$t" -c "$c" -s "$s" -n "$n" -k "$k" -z "$z" \
      --setting niid --num-shards "$ns" -p "$outdir" \
      >> "$outfile" &

  wait

  cat "$outfile" | python vis/curvevis.py --output-dir "$outdir" --nolegend
done

echo "All heterogeneity experiments done."

#!/usr/bin/env bash

# Selection Ratio Robustness Experiment
# Compare selection_size / cluster_size = 20%, 50%, 80%

# ---- Parse args ----
g="0,1,2,3"
while getopts "t:c:z:i:d:S:g:" opt; do
  case $opt in
    t) t=$OPTARG ;;
    c) c=$OPTARG ;;
    z) z=$OPTARG ;;
    i) i=$OPTARG ;;
    d) d=$OPTARG ;;
    S) S=$OPTARG ;;
    g) g=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

# ---- Check argument completeness ----
if [ -z "$t" ] || [ -z "$c" ] || [ -z "$z" ] || [ -z "$i" ] || [ -z "$d" ] || [ -z "$S" ]; then
    echo "Usage: $0 -t T -c C -z Z -i I -d DATASET -S SETTING [-g GPUs]"
    echo "  -t  total epochs"
    echo "  -c  cluster size"
    echo "  -z  cluster sampler"
    echo "  -i  experiment identifier"
    echo "  -d  dataset (cifar10/cifar100/fashion/har/shake)"
    echo "  -S  data heterogeneity setting (niid/diri)"
    exit 1
fi

# ---- Shakespeare only supports diri ----
if [ "$d" = "shake" ] && [ "$S" != "diri" ]; then
    echo "Error: shake (Shakespeare) dataset only supports diri setting, got '$S'"
    exit 1
fi

# ---- Compute selection sizes ----
s20=$(( c * 20 / 100 ))   # 20% of cluster
s50=$(( c * 50 / 100 ))   # 50% of cluster
s80=$(( c * 80 / 100 ))   # 80% of cluster

echo "Cluster size: $c"
echo "Selection sizes: 20%=$s20, 50%=$s50, 80%=$s80"

# ---- Create output directory ----
outdir="./result/SelRatio_${d}_${t}_${c}_${z}_${S}_${i}"
mkdir -p "$outdir"

# merged output file
outfile="$outdir/stdout.log"
: > "$outfile"

# ---- Run 3 experiments in parallel ----
echo "Running selection ratio experiments with T=$t, C=$c, Z=$z, dataset=$d, setting=$S"
echo "Results will be saved to $outdir"

echo "Start ratio=20% (s=$s20)"
python main.py -t "$t" -c "$c" -s "$s20" -z "$z" -d "$d" -g "$g" -a STCS -p "${outdir}" --setting "$S" -F 128 --recall 5 -i "ratio20" --legend "_20%" \
    >> "$outfile" &

echo "Start ratio=50% (s=$s50)"
python main.py -t "$t" -c "$c" -s "$s50" -z "$z" -d "$d" -g "$g" -a STCS -p "${outdir}" --setting "$S" -F 128 --recall 5 -i "ratio50" --legend "_50%" \
    >> "$outfile" &

echo "Start ratio=80% (s=$s80)"
python main.py -t "$t" -c "$c" -s "$s80" -z "$z" -d "$d" -g "$g" -a STCS -p "${outdir}" --setting "$S" -F 128 --recall 5 -i "ratio80" --legend "_80%" \
    >> "$outfile" &

# ---- Wait for all jobs ----
wait

# ---- Feed all collected stdout to curvevis.py ----
cat "$outfile" | python vis/curvevis.py --output-dir "$outdir"

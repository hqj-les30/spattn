#!/usr/bin/env bash

# ---- Embedding ablation: multistep (with id_emb) vs noemb (without id_emb)
# ---- Runs both diri and niid settings in parallel
# ---- Usage: bash exp/emb_ablation.sh -d <dataset> [-t T] [-c C] [-s S] [-z Z] [-g GPUs]

# ---- Defaults ----
dataset=""
t=50
c=10
s=5
z="clsrand"
g="0,1,2,3"

# ---- Parse args ----
while getopts "d:t:c:s:z:g:" opt; do
  case $opt in
    d) dataset=$OPTARG ;;
    t) t=$OPTARG ;;
    c) c=$OPTARG ;;
    s) s=$OPTARG ;;
    z) z=$OPTARG ;;
    g) g=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

if [ -z "$dataset" ]; then
    echo "Usage: $0 -d <dataset> [-t T] [-c C] [-s S] [-z Z] [-g GPUs]"
    exit 1
fi

# ---- Shakespeare only supports diri, emb_ablation runs both diri and niid ----
if [ "$dataset" = "shake" ]; then
    echo "Error: shake (Shakespeare) dataset only supports diri setting, emb_ablation requires both diri and niid"
    exit 1
fi

# ---- Create output directory ----
outdir="./result/EmbAblation_${dataset}_${t}_${c}_${s}_${z}"
mkdir -p "$outdir"

outfile="$outdir/stdout.log"
: > "$outfile"

echo "Running embedding ablation: dataset=$dataset, T=$t, C=$c, S=$s, Z=$z"
echo "Results will be saved to $outdir"

# ---- Dirichlet setting ----
echo "Start Dirichlet + multistep (with id_emb)"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$dataset" -g "$g" -a STCS -p "${outdir}" \
    --setting diri -q multistep -F 128 --recall 5 -i "diri_emb" --legend " Dirichlet" \
    >> "$outfile" &

echo "Start Dirichlet + noemb (without id_emb)"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$dataset" -g "$g" -a STCS -p "${outdir}" \
    --setting diri -q noemb -F 128 --recall 5 -i "diri_noemb" --legend " Dirichlet (w/o id_emb)" \
    >> "$outfile" &

# ---- Label Skew (niid) setting ----
echo "Start Label Skew + multistep (with id_emb)"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$dataset" -g "$g" -a STCS -p "${outdir}" \
    --setting niid -q multistep -F 128 --recall 5 -i "niid_emb" --legend " Label Skew" \
    >> "$outfile" &

echo "Start Label Skew + noemb (without id_emb)"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$dataset" -g "$g" -a STCS -p "${outdir}" \
    --setting niid -q noemb -F 128 --recall 5 -i "niid_noemb" --legend " Label Skew (w/o id_emb)" \
    >> "$outfile" &

# ---- Wait for all jobs ----
wait

# ---- Feed all collected stdout to ablation_vis.py ----
cat "$outfile" | python vis/ablation_vis.py --output-dir "$outdir"

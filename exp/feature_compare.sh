#!/usr/bin/env bash

# ---- Parse args ----
g="0,1,2,3"
while getopts "t:c:s:z:i:g:" opt; do
  case $opt in
    t) t=$OPTARG ;;
    c) c=$OPTARG ;;
    s) s=$OPTARG ;;
    z) z=$OPTARG ;;
    i) i=$OPTARG ;;
    g) g=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

# ---- Check argument completeness ----
if [ -z "$t" ] || [ -z "$c" ] || [ -z "$s" ] || [ -z "$z" ] || [ -z "$i" ]; then
    echo "Usage: $0 -t T -c C -s S -z Z -i I"
    exit 1
fi

# ---- Create output directory ----
outdir="./result/${i}_${t}_${c}_${s}_${z}"
mkdir -p "$outdir"

# merged output file
outfile="$outdir/stdout.log"
: > "$outfile"    # empty the file before writing

# ---- Run 4 commands in parallel ----
echo "Running experiments with T=$t, C=$c, S=$s, Z=$z, I=$i"
echo "Results will be saved to $outdir"

echo "Start PCA"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -g "$g" -i "PCA16" -a Ours -p "${outdir}" \
    --feature "PCA" --legend "_PCA16" -F 16\
    >> "$outfile" &

echo "Start RPCA"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -g "$g" -i "PCA32" -a Ours -p "${outdir}" \
    --feature "PCA" --legend "_PCA32" -F 32\
    >> "$outfile" &

echo "Start TRUNC"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -g "$g" -i "PCA64" -a Ours -p "${outdir}" \
    --feature "PCA" --legend "_PCA64" -F 64\
    >> "$outfile" &

# echo "Start HA"
# python main.py -t "$t" -c "$c" -s "$s" -z "$z" -i "$i" -a Ours -p "${outdir}" \
#     >> "$outfile" &

# ---- Wait for all jobs ----
wait

# ---- Feed all collected stdout to curvevis.py ----
cat "$outfile" | python vis/curvevis.py --output-dir "$outdir"

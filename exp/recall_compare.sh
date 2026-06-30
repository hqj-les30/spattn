#!/usr/bin/env bash

# ---- Defaults ----
g="0,1,2,3"

# ---- Parse args ----
while getopts "t:c:s:z:i:d:S:g:" opt; do
  case $opt in
    t) t=$OPTARG ;;
    c) c=$OPTARG ;;
    s) s=$OPTARG ;;
    z) z=$OPTARG ;;
    i) i=$OPTARG ;;
    d) d=$OPTARG ;;
    S) S=$OPTARG ;;
    g) g=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

# ---- Check argument completeness ----
if [ -z "$t" ] || [ -z "$c" ] || [ -z "$s" ] || [ -z "$z" ] || [ -z "$i" ] || [ -z "$d" ] || [ -z "$S" ]; then
    echo "Usage: $0 -t T -c C -s S -z Z -i I -d DATASET -S SETTING [-g GPUs]"
    exit 1
fi

# ---- Shakespeare only supports diri ----
if [ "$d" = "shake" ] && [ "$S" != "diri" ]; then
    echo "Error: shake (Shakespeare) dataset only supports diri setting, got '$S'"
    exit 1
fi

# ---- Create output directory ----
outdir="./result/RCCompare_${d}_${t}_${c}_${s}_${z}_${S}_${i}"
mkdir -p "$outdir"

# merged output file
outfile="$outdir/stdout.log"
: > "$outfile"    # empty the file before writing

# ---- Run 4 commands in parallel ----
echo "Running experiments with T=$t, C=$c, S=$s, Z=$z, I=$i, dataset=$d, setting=$S"
echo "Results will be saved to $outdir"

echo "Start recall 1"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$d" -g "$g" -a Ours -p "${outdir}" -F 128 --setting "$S" --recall 1 -i "r1" --legend "_r1" \
    >> "$outfile" &

echo "Start recall 2"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$d" -g "$g" -a Ours -p "${outdir}" -F 128 --setting "$S" --recall 2 -i "r2" --legend "_r2" \
    >> "$outfile" &

echo "Start recall 3"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$d" -g "$g" -a Ours -p "${outdir}" -F 128 --setting "$S" --recall 3 -i "r3" --legend "_r3" \
    >> "$outfile" &

echo "Start recall 4"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$d" -g "$g" -a Ours -p "${outdir}" -F 128 --setting "$S" --recall 5 -i "r5" --legend "_r5" \
    >> "$outfile" &

echo "Start recall 8"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$d" -g "$g" -a Ours -p "${outdir}" -F 128 --setting "$S" --recall 8 -i "r8" --legend "_r8" \
    >> "$outfile" &

echo "Start recall 10"
python main.py -t "$t" -c "$c" -s "$s" -z "$z" -d "$d" -g "$g" -a Ours -p "${outdir}" -F 128 --setting "$S" --recall 10 -i "r10" --legend "_r10" \
    >> "$outfile" &

# echo "Start FedAVG"
# python main.py -t "$t" -c "$c" -s "$s" -z "$z" -a FedAVG -p "${outdir}" \
#     >> "$outfile" &

# ---- Wait for all jobs ----
wait

# ---- Rename method labels in stdout for visualization ----
sed -i -e 's/Ours_r1:/H=1(MDP-Based):/' \
       -e 's/Ours_r2:/H=2:/' \
       -e 's/Ours_r3:/H=3:/' \
       -e 's/Ours_r5:/H=5:/' \
       -e 's/Ours_r8:/H=8:/' \
       -e 's/Ours_r10:/H=10:/' \
       "$outfile"

# ---- Feed all collected stdout to curvevis.py ----
cat "$outfile" | python vis/curvevis.py --output-dir "$outdir"

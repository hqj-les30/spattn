#!/usr/bin/env bash

# ---- Component-level ablation (review item C2): isolate where the accuracy
# ---- gain comes from. Full vs A1c/A2c/A3c/A5c (+ A6c id_emb as reference).
# ----
# ---- Phase 1 (Label Skew, -S niid): Full + A1c + A2c + A3c + A5c + A6c
# ---- Phase 2 (Dirichlet,  -S diri): Full + A1c + A5c + A6c
# ----
# ---- Condition -> flag map:
# ----   Full  : (none)
# ----   A1c   : --no-temporal-agg     (w/o Temporal Parameter Aggregation)
# ----   A2c   : --no-temporal-attn    (w/o Temporal Attention)
# ----   A3c   : --no-spatial-attn     (w/o Spatial Attention)
# ----   A5c   : --mlp-encoder         (MLP history encoder)
# ----   A6c   : -q noemb              (w/o Identity Embedding, existing)
# ----
# ---- Usage: bash exp/comp_ablation.sh -d <dataset> -S <niid|diri> \
# ----          [-t T -c C -s S -z Z -g GPUs -F Fdim -R recall -w workers -P maxparallel -e seed]
# ---- -e sets the random seed (main.py --seed, default 40); when given, it is also
# ---- appended to the output dir (..._s<seed>) so multiple seeds don't collide.
# ---- Resource note: each main spawns (num_gpus * workers) pool procs. Conditions
# ---- launch in parallel but are capped at -P concurrent (default 2). With 3 GPUs
# ---- and -w 1 that is 4 procs/condition, so -P 2 => 8 procs per experiment. Use
# ---- PARALLEL=0 to run all conditions strictly serially (1 at a time). Conditions
# ---- whose details.jsonl already reached -t epochs are skipped (resume-friendly).

set -u

# ---- Defaults (match emb_ablation.sh; final paper runs use -t 1500 per req 3.2) ----
dataset=""
setting="niid"
t=50
c=10
s=5
z="clsrand"
g="0,1,2,3"
F=128
R=5
w=1
P=2
e=""

while getopts "d:S:t:c:s:z:g:F:R:w:P:e:" opt; do
  case $opt in
    d) dataset=$OPTARG ;;
    S) setting=$OPTARG ;;
    t) t=$OPTARG ;;
    c) c=$OPTARG ;;
    s) s=$OPTARG ;;
    z) z=$OPTARG ;;
    g) g=$OPTARG ;;
    F) F=$OPTARG ;;
    R) R=$OPTARG ;;
    w) w=$OPTARG ;;
    P) P=$OPTARG ;;
    e) e=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

if [ -z "$dataset" ]; then
    echo "Usage: $0 -d <dataset> -S <niid|diri> [-t T -c C -s S -z Z -g GPUs -F Fdim -R recall -w workers -P maxparallel -e seed]"
    exit 1
fi

case "$setting" in
    niid) setting_label="Label Skew" ;;
    diri) setting_label="Dirichlet" ;;
    *) echo "Error: -S must be niid or diri (got '$setting')"; exit 1 ;;
esac

if [ "$dataset" = "shake" ] && [ "$setting" != "diri" ]; then
    echo "Error: shake (Shakespeare) only supports the diri setting"
    exit 1
fi

sshort="$setting"
outdir="./result/CompAblation_${dataset}_${sshort}_${t}_${c}_${s}_${z}"
seed_flag=""
if [ -n "$e" ]; then
    outdir="${outdir}_s${e}"
    seed_flag="--seed $e"
fi
mkdir -p "$outdir"
outfile="$outdir/stdout.log"
: > "$outfile"

echo "Running component ablation: dataset=$dataset, setting=$setting ($setting_label), T=$t, C=$c, S=$s, Z=$z, F=$F, recall=$R, workers/gpu=$w, seed=${e:-default}"
echo "Results -> $outdir"

# shared base command (word-split intentionally; args contain no spaces)
# Use combo method: multistep_vec + scalar input (per-class Q)
base="python main.py -t $t -c $c -s $s -z $z -d $dataset -g $g -a Ours -p ${outdir} --setting $setting -F $F --recall $R --workers-per-gpu $w --enc-input scalar ${seed_flag}"

# run_cond <tag> <variant-label> <qnet> [extra-flags...]
# Skips conditions already finished (resume-friendly), and caps concurrency at $P.
RUNNING=0
run_cond() {
    local tag="$1"; local variant="$2"; local qnet="$3"; shift 3
    # skip if this condition already finished (details.jsonl has >= t epochs)
    local prev
    prev=$(ls -d "${outdir}"/*"${sshort}_${tag}" 2>/dev/null | head -1)
    if [ -n "$prev" ] && [ -f "$prev/details.jsonl" ]; then
        local done_n=$(($(wc -l < "$prev/details.jsonl") - 1))
        if [ "$done_n" -ge "$t" ]; then
            echo "Skip ${setting_label} + ${variant} (already ${done_n}/${t})"
            return
        fi
    fi
    echo "Start ${setting_label} + ${variant}"
    if [ "${PARALLEL:-1}" = "0" ]; then
        $base -q "$qnet" "$@" -i "${sshort}_${tag}" --legend " ${setting_label} (${variant})" >> "$outfile"
    else
        $base -q "$qnet" "$@" -i "${sshort}_${tag}" --legend " ${setting_label} (${variant})" >> "$outfile" &
        RUNNING=$((RUNNING + 1))
        # concurrency gate: at most $P conditions running at once
        while [ "$RUNNING" -ge "$P" ]; do
            wait -n
            RUNNING=$((RUNNING - 1))
        done
    fi
}

if [ "$setting" = "niid" ]; then
    # Phase 1: full attribution on the setting where Ours wins most
    run_cond full "Full"               multistep_vec
    run_cond a1c  "w/o Temporal Agg"   multistep_vec --no-temporal-agg
    run_cond a2c  "w/o Temporal Attn"  multistep_vec --no-temporal-attn
    run_cond a3c  "w/o Spatial Attn"   multistep_vec --no-spatial-attn
    run_cond a5c  "MLP Encoder"        multistep_vec --mlp-encoder
    run_cond a6c  "w/o id_emb"         noemb
else
    # Phase 2: generalization check — only the two most critical conditions
    run_cond full "Full"               multistep_vec
    run_cond a1c  "w/o Temporal Agg"   multistep_vec --no-temporal-agg
    run_cond a5c  "MLP Encoder"        multistep_vec --mlp-encoder
    run_cond a6c  "w/o id_emb"         noemb
fi

wait

# grouped bar chart + Delta-vs-Full summary (Table 2 style)
cat "$outfile" | python vis/ablation_vis.py --output-dir "$outdir" --delta-vs Full

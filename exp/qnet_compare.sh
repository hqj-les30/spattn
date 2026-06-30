#!/bin/bash

# --- 1. 定义变量和默认值 ---
T_VAL="" # -t 参数的值
C_VAL="" # -c 参数的值
S_VAL="" # -s 参数的值
Z_VAL="" # -z 参数的值
i=""     # 实验标识符
g="0,1,2,3" # GPU编号

# Python 脚本文件名
PYTHON_SCRIPT="main.py"
# PYTHON_SCRIPT_B="main_b.py"

# --- 2. 解析命名参数 ---
# getopts 字符串 't:c:s:z:' 中的冒号表示该选项（flag）需要一个参数（argument）。
while getopts 't:c:s:z:i:g:' flag; do
    case "${flag}" in
        t) T_VAL="${OPTARG}" ;;
        c) C_VAL="${OPTARG}" ;;
        s) S_VAL="${OPTARG}" ;;
        z) Z_VAL="${OPTARG}" ;;
        i) i="${OPTARG}" ;;
        g) g="${OPTARG}" ;;
        *)
            echo "用法: $0 -t <值> -c <值> -s <值> -z <值> -i <字符串>"
            exit 1
            ;;
    esac
done

# --- 3. 检查关键参数是否传入 ---
# 检查四个参数是否都已赋值
if [ -z "$T_VAL" ] || [ -z "$C_VAL" ] || [ -z "$S_VAL" ] || [ -z "$Z_VAL" ]; then
    echo "错误：必须同时提供 -t, -c, -s, -z 四个参数。"
    echo "用法: $0 -t <值> -c <值> -s <值> -z <值>"
    exit 1
fi

# ---- Create output directory ----
outdir="./result/${i}_${T_VAL}_${C_VAL}_${S_VAL}_${Z_VAL}"
mkdir -p "$outdir"

# merged output file
outfile="$outdir/stdout.log"
: > "$outfile"    # empty the file before writing

# --- 4. 并行执行 Python 命令 ---
echo "传递参数: -t ${T_VAL} -c ${C_VAL} -s ${S_VAL} -z ${Z_VAL}"
# 启动第一个 Python 脚本 (使用 -t 和 -c)
echo "--- 1. 启动 DuelMLP ---"
# echo "传递参数: -t ${T_VAL} -c ${C_VAL}"
python ${PYTHON_SCRIPT} -t ${T_VAL} -c ${C_VAL} -s ${S_VAL} -z ${Z_VAL} -g "$g" -q duel -i duel \
    -p "${outdir}" --legend duel \
    >> "$outfile" &

# 启动第二个 Python 脚本 (使用 -s 和 -z)
echo "--- 2. 启动 DuelLSTM ---"
# echo "传递参数: -s ${S_VAL} -z ${Z_VAL}"
python ${PYTHON_SCRIPT} -t ${T_VAL} -c ${C_VAL} -s ${S_VAL} -z ${Z_VAL} -g "$g" -q lstm -i lstm \
    -p "${outdir}" --legend lstm \
    >> "$outfile" &

# 启动第二个 Python 脚本 (使用 -s 和 -z)
echo "--- 3. 启动 DuelLSTM2L ---"
# echo "传递参数: -s ${S_VAL} -z ${Z_VAL}"
python ${PYTHON_SCRIPT} -t ${T_VAL} -c ${C_VAL} -s ${S_VAL} -z ${Z_VAL} -g "$g" -q lstm2l -i lstm2l \
    -p "${outdir}" --legend lstm2l \
    >> "$outfile" &

# `wait` 命令等待所有后台进程完成
echo "--- 等待两个后台脚本完成 ---"
wait

echo "--- 所有脚本执行完毕 ---"
# ---- Feed all collected stdout to curvevis.py ----
cat "$outfile" | python vis/curvevis.py --output-dir "$outdir"
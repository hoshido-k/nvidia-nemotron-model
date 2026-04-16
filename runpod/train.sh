#!/bin/bash
# RTX 6000 Ada (48GB) + QLoRA (4-bit) で学習を実行する
#
# 使い方:
#   bash runpod/train.sh

set -e

WORKSPACE=/workspace
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_DIR="$WORKSPACE/model"
DATA_CSV="$WORKSPACE/data/train_split_with_cot.csv"
OUTPUT_DIR="$WORKSPACE/output/lora-adapter"

echo "=== Nemotron LoRA 学習 (QLoRA 4-bit) ==="
echo "モデル : $MODEL_DIR"
echo "データ : $DATA_CSV"
echo "出力   : $OUTPUT_DIR"
echo ""

python "$REPO_DIR/notebooks/nemotron-train/train.py" \
    --model_dir   "$MODEL_DIR" \
    --data_csv    "$DATA_CSV" \
    --output_dir  "$OUTPUT_DIR" \
    --load_in_4bit \
    --lora_rank   32 \
    --lora_alpha  32 \
    --epochs      2 \
    --batch_size  4 \
    --grad_accum  1 \
    --lr          5e-5 \
    --max_seq_len 2048 \
    --zip_output

echo ""
echo "=== 学習完了 ==="
echo "アダプタ: $OUTPUT_DIR"
echo "zip    : $WORKSPACE/output/submission.zip"
echo ""
echo "次のステップ: アダプタを Kaggle Dataset にアップロード"
echo "  kaggle datasets create -p $OUTPUT_DIR"

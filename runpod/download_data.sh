#!/bin/bash
# モデル・学習データを Kaggle からダウンロードする
#
# 事前準備:
#   mkdir -p ~/.kaggle
#   cp /path/to/kaggle.json ~/.kaggle/kaggle.json
#   chmod 600 ~/.kaggle/kaggle.json

set -e

WORKSPACE=/workspace

echo "=== [1/2] ベースモデルDL (~60GB) ==="
mkdir -p "$WORKSPACE/model"
kaggle models instances versions download \
    metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 \
    -p "$WORKSPACE/model"

# zip が展開されていない場合に展開
if ls "$WORKSPACE/model"/*.zip 1>/dev/null 2>&1; then
    unzip -o "$WORKSPACE/model"/*.zip -d "$WORKSPACE/model"
    rm "$WORKSPACE/model"/*.zip
fi
echo "モデル保存先: $WORKSPACE/model"

echo "=== [2/2] 学習データDL ==="
mkdir -p "$WORKSPACE/data"
kaggle datasets download \
    konbu17/nemotron-sft-lora-cot-selection \
    -p "$WORKSPACE/data" --unzip
echo "データ保存先: $WORKSPACE/data"

echo ""
echo "=== Download complete! ==="
ls -lh "$WORKSPACE/model"
ls -lh "$WORKSPACE/data"

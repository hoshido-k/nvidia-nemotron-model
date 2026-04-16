#!/bin/bash
# RunPod 起動後の軽量セットアップ
# カスタムイメージ (dte59723/nemotron-train:latest) 使用前提
# mamba-ssm / PyTorch / ML ライブラリはイメージ内にビルド済み
#
# 使い方:
#   git clone https://github.com/hoshido-k/nvidia-nemotron-model.git
#   cd nvidia-nemotron-model
#   bash runpod/setup.sh

set -e

echo "=== [1/3] 動作確認 ==="
python - <<'EOF'
import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
import mamba_ssm; print(f"mamba-ssm : OK")
import bitsandbytes; print(f"bitsandbytes : {bitsandbytes.__version__}")
EOF

echo "=== [2/3] Kaggle 認証確認 ==="
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "[!] ~/.kaggle/kaggle.json が見つかりません。"
    echo "    以下を実行して kaggle.json を配置してください:"
    echo "      mkdir -p ~/.kaggle"
    echo "      # kaggle.json を貼り付け or scp でコピー"
    echo "      chmod 600 ~/.kaggle/kaggle.json"
else
    echo "Kaggle 認証: OK"
fi

echo "=== [3/3] Claude Code 認証 ==="
if ! claude --version &>/dev/null 2>&1; then
    echo "[!] Claude Code が見つかりません（イメージを確認してください）"
else
    echo "Claude Code: $(claude --version)"
    echo "未認証の場合は 'claude /login' を実行してください。"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "次のステップ:"
echo "  1. claude /login                 # 初回のみ"
echo "  2. bash runpod/download_data.sh  # モデル・データ DL"
echo "  3. bash runpod/train.sh          # 学習実行"

#!/bin/bash
# RunPod セットアップスクリプト
# Base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# GPU: RTX 6000 Ada (sm_89, 48GB) + --load_in_4bit (QLoRA)
#
# 使い方:
#   git clone https://github.com/hoshido-k/nvidia-nemotron-model.git
#   cd nvidia-nemotron-model
#   bash runpod/setup.sh

set -e

echo "=== [1/4] mamba-ssm + causal-conv1d (sm_89 Ada Lovelace) ==="
pip install mamba-ssm causal-conv1d

echo "=== [2/4] ML ライブラリ ==="
pip install \
    "transformers>=4.45.0" \
    "trl==1.1.0" \
    "peft>=0.12.0" \
    "bitsandbytes>=0.43.0" \
    "accelerate>=0.34.0" \
    "datasets" \
    "pandas"

echo "=== [3/4] Kaggle CLI (モデル・データDL用) ==="
pip install kaggle

echo "=== [4/4] Claude Code ==="
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
curl -fsSL https://claude.ai/install.sh | bash
echo "Claude Code インストール完了。初回は 'claude /login' で認証してください。"

echo "=== [5/5] 確認 ==="
python - <<'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
import mamba_ssm; print(f"mamba-ssm: OK")
import bitsandbytes; print(f"bitsandbytes: {bitsandbytes.__version__}")
EOF

echo ""
echo "=== Setup complete! ==="
echo ""
echo "次のステップ:"
echo "  1. claude /login                 # Claude Code 認証（初回のみ）"
echo "  2. bash runpod/download_data.sh  # モデル・データDL"
echo "  3. bash runpod/train.sh          # 学習実行"

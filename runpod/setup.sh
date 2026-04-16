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

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== [0/5] uv インストール ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

echo "=== [1/5] PyTorch CUDA バージョン確認・修正 ==="
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
echo "PyTorch CUDA version: $TORCH_CUDA"
if [ "$TORCH_CUDA" != "12.4" ]; then
    echo "PyTorch CUDA mismatch (got $TORCH_CUDA, need 12.4). 再インストール..."
    uv pip install --system torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
        --reinstall
    echo "PyTorch 2.4.0+cu124 インストール完了"
else
    echo "PyTorch CUDA 12.4 OK"
fi

echo "=== [2/5] causal-conv1d + mamba-ssm (sm_89 Ada Lovelace, CUDAビルド) ==="
# Kaggle utility script と同じアプローチ:
#   - TORCH_CUDA_ARCH_LIST でアーキテクチャを明示（sm_89 = Ada Lovelace）
#   - --no-build-isolation で既インストールの torch を使用
#   - mamba-ssm は git 最新版（リリース版よりアーキテクチャ対応が広い）
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS="4"
pip install --no-build-isolation nvidia-cutlass
pip install --no-build-isolation "causal-conv1d>=1.4.0"
pip install --no-build-isolation "git+https://github.com/state-spaces/mamba.git"

echo "=== [3/5] ML ライブラリ (uv) ==="
uv pip install --system -r "$REPO_DIR/runpod/requirements.txt"

echo "=== [4/5] Claude Code ==="
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
if ! command -v claude &>/dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash
fi
echo "Claude Code インストール完了。初回は 'claude /login' で認証してください。"

echo "=== [5/5] 動作確認 ==="
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

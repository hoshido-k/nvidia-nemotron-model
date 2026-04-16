# RunPod カスタムイメージ
# GPU: RTX 6000 Ada (sm_89) + QLoRA (--load_in_4bit)
# Docker Hub: dte59723/nemotron-train:latest
#
# ビルド: bash runpod/build_and_push.sh

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# ベースイメージの torch バージョンが異なる場合があるため固定
RUN pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall --quiet

# mamba-ssm + causal-conv1d をインストール
# SKIP_CUDA_BUILD=TRUE でCUDAコンパイルをスキップ → Mac でもビルド可能
# 実行時に Triton JIT が自動コンパイル（Kaggle UTILITY SCRIPTS と同じ仕組み）
RUN pip install nvidia-cutlass && \
    CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE pip install "causal-conv1d>=1.4.0" && \
    MAMBA_SKIP_CUDA_BUILD=TRUE pip install "git+https://github.com/state-spaces/mamba.git"

# ML ライブラリ
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Node.js + Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs --quiet && \
    curl -fsSL https://claude.ai/install.sh | bash

WORKDIR /workspace

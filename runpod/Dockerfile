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

# mamba-ssm + causal-conv1d を Ada Lovelace (sm_89) 向けにビルド
# Kaggle utility script と同じアプローチ（--no-build-isolation + ARCH明示）
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV MAX_JOBS="4"
RUN pip install --no-build-isolation nvidia-cutlass && \
    pip install --no-build-isolation "causal-conv1d>=1.4.0" && \
    pip install --no-build-isolation "git+https://github.com/state-spaces/mamba.git"

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

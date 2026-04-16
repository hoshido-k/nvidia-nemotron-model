#!/bin/bash
# Docker Hub にカスタムイメージをビルド & プッシュする
#
# 事前準備:
#   docker login
#
# 使い方:
#   bash runpod/build_and_push.sh

set -e

IMAGE="dte59723/nemotron-train:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== ビルド: $IMAGE ==="
docker build \
    --platform linux/amd64 \
    -t "$IMAGE" \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$SCRIPT_DIR"

echo "=== プッシュ: $IMAGE ==="
docker push "$IMAGE"

echo ""
echo "=== 完了 ==="
echo "RunPod で使うイメージ名: $IMAGE"
echo "RunPod Pod 作成時に「Custom Image」で上記を指定してください。"

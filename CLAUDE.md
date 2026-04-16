# CLAUDE.md

## コマンド実行

Python スクリプトや CLI ツール（kaggle コマンド等）は必ず `uv run` で実行すること。

```bash
# 例
uv run python script.py
uv run kaggle kernels pull ...
uv run python -c "..."
```

## プロジェクト概要

Kaggle の NVIDIA Nemotron-H モデルを使った LoRA SFT fine-tuning コンペ用リポジトリ。

- **学習スクリプト**: `notebooks/nemotron-train/train.py` が唯一の編集対象
- **自動デプロイ**: main への push で GitHub Actions が `notebooks/nemotron-train/train.py` を Kaggle Dataset (`shotokishida/nemotron-train-scripts`) に自動同期
- `kaggle-datasets/nemotron-train-scripts/train.py` は CI が上書きする自動生成物。直接編集しない

## 環境変数

`.env` に Kaggle・GitHub の認証情報が入っている。

## Kaggle 実行環境

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition（VRAM 95GB）
- Triton パッチ（Blackwell 対応）が必要
- インターネット無効 → trl 等は Dataset 経由のオフラインインストール

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

### 編集対象ファイル

| ファイル | 役割 |
|---|---|
| `notebooks/nemotron-train/train.py` | 学習スクリプト本体（唯一の編集対象） |
| `notebooks/nemotron-train/train.ipynb` | 学習実行ノートブック（パラメータ調整はここ） |
| `notebooks/nvidia-nemotron-model-reasoning-challenge/nvidia-nemotron-model-reasoning-challenge.ipynb` | 推論・提出ノートブック |

### 自動デプロイ

- main への push で GitHub Actions が `notebooks/nemotron-train/train.py` を Kaggle Dataset (`shotokishida/nemotron-train-scripts`) に自動同期
- `kaggle-datasets/nemotron-train-scripts/train.py` は CI が上書きする自動生成物。直接編集しない

### 学習 → 提出フロー

1. `train.py` を編集 → main に push → GitHub Actions が Kaggle Dataset に同期
2. Kaggle で `nemotron-train` ノートブックを Run & All（Commit）で実行
3. 出力された adapter ファイルを `shotokishida/nemotron-adapter` Kaggle モデルにアップロード
4. Kaggle で `nvidia-nemotron-model-reasoning-challenge` ノートブックを Run & All（Commit）で提出

## 環境変数

`.env` に Kaggle・GitHub の認証情報が入っている。

## Kaggle 実行環境

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition（VRAM 95GB）
- Triton パッチ（Blackwell 対応）が必要
- インターネット無効（推論ノートブック）→ trl 等は Dataset 経由のオフラインインストール
- インターネット有効（学習ノートブック）

## 主要ハイパーパラメータ（dgxchen / Tong Hui Kang 準拠）

| パラメータ | 値 | 備考 |
|---|---|---|
| `lora_rank` | 32 | competition 上限（vLLM `max_lora_rank=32`） |
| `lora_dropout` | 0.0 | dropout なし |
| `epochs` | 1 | 過学習回避 |
| `lr` | 2e-4 | linear scheduler |
| `warmup_steps` | 0 | |
| `adam_beta2` | 0.95 | デフォルト 0.999 より小さく設定 |
| `max_grad_norm` | 1e9 | 実質クリッピングなし |
| `grad_accum` | 32 | 実効バッチサイズ = 32 |
| `max_seq_len` | 8192 | CoT が長いため |
| `TARGET_MODULES` | q/k/v/o_proj + in/out/up/down_proj | lm_head は除外（アダプタ巨大化防止） |

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
| `notebooks/nemotron_drLora_reasoning/nemotron_drLora_reasoning.ipynb` | 推論・提出ノートブック（MODE="train"/"infer" で切替） |

### 自動デプロイ

- main への push で GitHub Actions が各ノートブックを自動同期
  - `notebooks/nemotron-train/**` → Kaggle Dataset (`shotokishida/nemotron-train-scripts`)
  - `notebooks/nemotron_drLora_reasoning/**` → Kaggle Kernel (`shotokishida/nemotron-drlora-reasoning`)
- `kaggle-datasets/nemotron-train-scripts/train.py` は CI が上書きする自動生成物。直接編集しない

### 学習 → 提出フロー

1. `train.py` / `train.ipynb` を編集 → main に push → GitHub Actions が Kaggle Dataset に同期
2. Kaggle で `nemotron-train` ノートブックを Run & All（Commit）で実行（`MODE="train"` 相当）
3. 出力された adapter ファイルを `shotokishida/nemotron-adapter` Kaggle モデルにアップロード
4. Kaggle で `nemotron-drlora-reasoning` ノートブック（`MODE="infer"`）を Run & All（Commit）で提出

### ノートブックの MODE 制御

`nemotron_drLora_reasoning.ipynb` は cell-0 の `MODE` 定数で動作を切り替える：

| MODE | 動作 |
|---|---|
| `"train"` | trl インストール → train.py 実行 → アダプタ出力 |
| `"infer"` | PEFT→vLLM 変換 → test.csv 推論 → submission.zip 作成 |

## 環境変数

`.env` に Kaggle・GitHub の認証情報が入っている。

## Kaggle 実行環境

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition（VRAM 95GB）
- Triton パッチ（Blackwell 対応）が必要
- インターネット無効（`nemotron-drlora-reasoning` ノートブック）→ trl は Dataset 経由のオフラインインストール
- インターネット有効（`nemotron-train` ノートブック）

## 主要ハイパーパラメータ

| パラメータ | 値 | 備考 |
|---|---|---|
| `lora_rank` | 32 | competition 上限（vLLM `max_lora_rank=32`） |
| `lora_dropout` | 0.0 | dropout なし |
| `epochs` | 2 | |
| `lr` | 2e-5 | linear scheduler |
| `warmup_ratio` | 0.05 | 全ステップの5%でLRを0から線形増加 |
| `weight_decay` | 0.01 | L2 正則化（過学習防止） |
| `max_grad_norm` | 1.0 | 勾配クリッピング |
| `adam_beta2` | 0.95 | デフォルト 0.999 より小さく設定 |
| `grad_accum` | 32 | 実効バッチサイズ = 32 |
| `max_seq_len` | 8192 | CoT が長いため |
| `TARGET_MODULES` | q/k/v/o_proj + in/out/up/down_proj | lm_head は除外（アダプタ巨大化防止） |

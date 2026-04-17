# NVIDIA Nemotron Model Reasoning Challenge

Kaggle コンペ [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) の実験リポジトリ。

ベースモデル `nemotron-3-nano-30b-a3b-bf16`（30B パラメータ Hybrid Mamba-Transformer MoE）に対して LoRA アダプタを学習・提出し、数学推論の精度を競う。

---

## リポジトリ構成

```
.
├── notebooks/
│   ├── nemotron-train/
│   │   ├── train.py             # 学習スクリプト本体（唯一の編集対象）
│   │   ├── train.ipynb          # 学習実行ノートブック
│   │   └── kernel-metadata.json
│   └── nvidia-nemotron-model-reasoning-challenge/
│       ├── nvidia-nemotron-model-reasoning-challenge.ipynb  # 推論・提出ノートブック
│       └── kernel-metadata.json
├── kaggle-datasets/
│   └── nemotron-train-scripts/  # CI 自動生成物（直接編集しない）
├── .github/
│   └── workflows/               # GitHub Actions（Kaggle Dataset 自動同期）
├── CLAUDE.md
├── .gitignore
└── .env                         # API キー（Git 管理外）
```

---

## 学習 → 提出フロー

```
train.py 編集
    ↓ git push main
GitHub Actions → Kaggle Dataset (shotokishida/nemotron-train-scripts) に同期
    ↓
Kaggle: nemotron-train ノートブックを Run & All
    ↓ adapter ファイルが /kaggle/working/adapter/ に出力
Kaggle モデル (shotokishida/nemotron-adapter) にアップロード
    ↓
Kaggle: nvidia-nemotron-model-reasoning-challenge ノートブックを Run & All → 提出
```

---

## 実行環境

| 環境 | 用途 | GPU |
|------|------|-----|
| Kaggle Notebook | 本番学習・推論・提出 | RTX Pro 6000 Blackwell (95GB VRAM) |
| MacBook | コード設計・EDA | なし |

---

## セットアップ

### 前提条件

- `uv` がインストール済み
- `~/.kaggle/kaggle.json` が設定済み
- `.env` に以下が設定済み:
  ```
  KAGGLE_API_TOKEN=...
  GH_TOKEN=...
  ```

### ローカル（MacBook）

```bash
export $(cat .env | xargs)

# コンペデータ
mkdir -p data/nvidia-nemotron-competition
uv run kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge \
  -p data/nvidia-nemotron-competition

# 公式ユーティリティ
uv run kaggle kernels output metric/nvidia-metric-utility-script \
  -p data/nvidia-metric-utility-script
```

---

## Kaggle への Push

`main` ブランチに push すると GitHub Actions が自動で Kaggle Dataset にデプロイする。

```bash
git add notebooks/nemotron-train/train.py
git commit -m "update train.py"
git push
```

推論ノートブックの手動 Push:
```bash
export $(cat .env | xargs)
uv run kaggle kernels push -p notebooks/nvidia-nemotron-model-reasoning-challenge
```

---

## 学習設定（dgxchen / Tong Hui Kang 準拠）

| パラメータ | 値 | 備考 |
|---|---|---|
| `lora_rank` | 32 | competition 上限 |
| `epochs` | 1 | 過学習回避 |
| `lr` | 2e-4 | linear scheduler |
| `grad_accum` | 32 | 実効バッチサイズ 32 |
| `max_seq_len` | 8192 | CoT が長いため |
| `TARGET_MODULES` | q/k/v/o_proj + in/out/up/down_proj | Attention + Mamba 全層 |
| `enable_thinking` | True | `apply_chat_template` で CoT 思考モード有効化 |

# NVIDIA Nemotron Model Reasoning Challenge

Kaggle コンペ [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) の実験リポジトリ。

ベースモデル `nemotron-3-nano-30b-a3b-bf16` に対して LoRA アダプタを学習・提出し、数学推論の精度を競う。

---

## リポジトリ構成

```
.
├── notebooks/
│   └── nvidia-nemotron-model-reasoning-challenge/
│       ├── nvidia-nemotron-model-reasoning-challenge.ipynb  # メインノートブック
│       └── kernel-metadata.json                             # Kaggle Push 設定
├── data/                        # ローカルデータ（.gitignore 済み）
│   ├── nvidia-nemotron-competition/   # コンペデータ
│   ├── nvidia-nemotron-all-linear/    # 公開ベスト解アダプタ
│   └── nvidia-metric-utility-script/ # 公式ユーティリティ
├── .github/
│   └── workflows/
│       └── kaggle-push-nvidia-nemotron-model-reasoning-challenge.yml
├── PLAN.md       # 実験計画
├── .gitignore
└── .env          # API キー（Git 管理外）
```

---

## 実行環境

| 環境 | 用途 | GPU |
|------|------|-----|
| Kaggle Notebook | 本番推論・提出 | RTX Pro 6000 (96GB) |
| RunPod | 動作確認・実験 | RTX 6000 Ada (48GB) |
| MacBook | EDA・コード設計 | なし |

ノートブックは `ENV` 変数で実行環境を自動検出し、パスを切り替える。

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
# データダウンロード
export $(cat .env | xargs)

mkdir -p data/nvidia-nemotron-competition
uv run kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge \
  -p data/nvidia-nemotron-competition

uv run kaggle kernels output huikang/nvidia-nemotron-all-linear \
  -p data/nvidia-nemotron-all-linear

uv run kaggle kernels output metric/nvidia-metric-utility-script \
  -p data/nvidia-metric-utility-script
```

### RunPod

```bash
# Pod 内で実行（KAGGLE_USERNAME / KAGGLE_KEY は Pod 環境変数に設定済みの前提）
pip install vllm bitsandbytes peft transformers accelerate kaggle safetensors tqdm pandas

git clone https://github.com/hoshido-k/nvidia-nemotron-model.git /workspace/repo
cd /workspace/repo

# ノートブックの ENV セットアップセルを実行するとデータが自動ダウンロードされる
```

---

## Kaggle への Push

`main` ブランチに push すると GitHub Actions が自動で Kaggle にデプロイする。

```bash
git add notebooks/nvidia-nemotron-model-reasoning-challenge/
git commit -m "update notebook"
git push
```

手動 Push:
```bash
export $(cat .env | xargs)
uv run kaggle kernels push -p notebooks/nvidia-nemotron-model-reasoning-challenge
```

---

## 実験計画

詳細は [PLAN.md](./PLAN.md) を参照。

| フェーズ | 内容 | 目標 |
|---------|------|------|
| Phase 1 | 環境構築・ベースライン提出 | スコアボードに乗る |
| Phase 2 | LoRA ファインチューニング | ベースライン + 数ポイント |
| Phase 3 | Gemini 2.5 Pro 蒸留 | さらに + 数ポイント |
| Phase 4 | TTT・アンサンブル等 | 上位を狙う |

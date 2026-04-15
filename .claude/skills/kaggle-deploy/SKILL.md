---
name: kaggle-deploy
description: Create GitHub Actions auto-deploy workflow for a Kaggle notebook directory. Use when wiring up a new or existing notebook directory to automated Kaggle kernel push on git push.
argument-hint: <dir> [--existing <user/slug>]
disable-model-invocation: true
---

Kaggle Notebook の自動デプロイワークフロー（GitHub Actions）をセットアップする。
新規ノートブック作成と、既存カーネルへの紐づけ両方に対応。

## 使い方

```
/kaggle-deploy <dir> [--existing <user/slug>]
```

例:
```
# 新規：notebooks/my-solver/ を作成してデプロイ設定
/kaggle-deploy notebooks/my-solver

# 既存カーネルと紐づけ（Kaggle 上で Copy & Edit 済み）
/kaggle-deploy notebooks/my-solver --existing shotokishida/answer-evolve
```

`<dir>` はリポジトリルートからの相対パス（例: `main`, `notebooks/gpt-oss-t0.5`）。  
`--existing <user/slug>` は Kaggle 上の既存カーネルスラッグ。省略時は新規作成扱い。

---

## 手順

以下を順番に実行すること。手動操作が必要なステップはユーザーに確認を取ること。

---

### Step 1: 引数の解析

`$ARGUMENTS` から以下を取り出す：
- `DIR`: デプロイ対象ディレクトリ（例: `notebooks/my-solver`）
- `KERNEL_SLUG`: `--existing` で指定されたスラッグ（省略時は空）

`DIR` が指定されていない場合はエラーを出して終了する。

---

### Step 2: ディレクトリ・ノートブックの確認

```bash
ls <DIR>/
```

- `<DIR>/` が存在しない場合は作成する
- `.ipynb` ファイルが存在するか確認する
  - 存在しない場合：ユーザーに「ノートブックファイルを `<DIR>/` に配置してください」と伝えて続行
  - 存在する場合：そのファイル名を `NOTEBOOK_FILE` として記録する

---

### Step 3: Kaggle カーネルとの紐づけ

**--existing が指定された場合（既存カーネルと紐づけ）:**

```bash
export $(cat .env | xargs)
uv run kaggle kernels pull <KERNEL_SLUG> -p /tmp/kaggle-deploy-tmp --metadata
cat /tmp/kaggle-deploy-tmp/kernel-metadata.json
```

取得した metadata から以下を記録する：
- `dataset_sources`
- `kernel_sources`
- `competition_sources`
- `model_sources`
- `enable_gpu`, `enable_internet`

**--existing が指定されない場合（新規）:**

ユーザーに以下を確認する：
1. デプロイ先の Kaggle カーネルスラッグ（例: `shotokishida/my-solver`）
2. GPU が必要か（H100 / T4 / なし）
3. 紐づけるデータソース（コンペ名、データセット名等）

---

### Step 4: kernel-metadata.json の作成・更新

`<DIR>/kernel-metadata.json` を以下の内容で作成または更新する。

**新規の場合のテンプレート:**

```json
{
  "id": "<user/slug>",
  "title": "<slug の最後の部分>",
  "code_file": "<NOTEBOOK_FILE>",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

- `enable_gpu: true` の場合、GPU 種別を確認してユーザーに `NvidiaH100` / `NvidiaT4x2` 等を選択させる
- Step 3 で取得したソース情報を反映する

---

### Step 5: ローカルデータのダウンロード

Step 3〜4 で確定した `competition_sources` / `dataset_sources` / `kernel_sources` をもとに、
EDA・ローカル実行用のデータを `data/` に取得する。

```bash
export $(cat .env | xargs)
```

**competition_sources** — コンペデータ（`data/<competition-slug>/` に展開）:

```bash
# 例: ai-mathematical-olympiad-progress-prize-3
mkdir -p data/<competition-slug>
uv run kaggle competitions download -c <competition-slug> -p data/<competition-slug>
unzip -q -o "data/<competition-slug>/<competition-slug>.zip" -d data/<competition-slug>
rm "data/<competition-slug>/<competition-slug>.zip"
```

**dataset_sources** — データセット（`data/<dataset-name>/` に展開）:

```bash
# 例: amanatar/50problems
uv run kaggle datasets download <owner>/<slug> -p data/<dataset-name> --unzip
```

**kernel_sources** — カーネル出力（`data/<kernel-name>/` に保存）:

```bash
# 例: andreasbis/aimo-3-utils
uv run kaggle kernels output <owner>/<slug> -p data/<kernel-name>
```

**model_sources** — **スキップ**（サイズが大きいため `data/` には保存しない。Kaggle 上でのみ利用）

> ダウンロード済みのものは `data/` に存在するため、再実行時は `ls data/` で確認して重複ダウンロードを避ける。

---

### Step 6: GitHub Actions ワークフローの作成

`.github/workflows/kaggle-push-<slug-basename>.yml` を作成する。

```yaml
name: Push <slug-basename> Notebook to Kaggle

on:
  push:
    branches:
      - main
    paths:
      - "<DIR>/**"
  workflow_dispatch:

jobs:
  push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Kaggle CLI
        run: pip install kaggle

      - name: Push notebook to Kaggle
        env:
          KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}
        run: |
          export KAGGLE_TOKEN=$KAGGLE_API_TOKEN
          kaggle kernels push -p <DIR>
```

- `<slug-basename>`: カーネルスラッグの `/` 以降（例: `shotokishida/my-solver` → `my-solver`）
- `paths:` は `<DIR>/**` にすることでそのディレクトリ変更時のみ発火する

---

### Step 7: KAGGLE_API_TOKEN の GitHub Secret 確認

```bash
export $(cat .env | xargs)
gh secret list
```

`KAGGLE_API_TOKEN` が **表示されない** 場合のみ、以下を実行してセットする：

```bash
# .env から KAGGLE_API_TOKEN を取り出して GitHub Secret に登録
export $(cat .env | xargs)
gh secret set KAGGLE_API_TOKEN --body "$KAGGLE_API_TOKEN"
```

> **セキュリティ上の注意**: KAGGLE_API_TOKEN は `.env`（git管理外）にのみ保存し、コードやコミット内にハードコードしない。
> GitHub Actions では `secrets.KAGGLE_API_TOKEN` 経由で参照するため、値が外部に漏れない。

---

### Step 8: コミット・デプロイ確認

```bash
git add <DIR>/ .github/workflows/kaggle-push-<slug-basename>.yml
git status
```

内容をユーザーに確認してから：

```bash
git commit -m "feat: <slug-basename> デプロイワークフロー追加"
git push origin main
```

Actions の実行を確認：

```bash
export $(cat .env | xargs) && sleep 15 && gh run list --limit 5
```

---

### チェックリスト

- [ ] `<DIR>/kernel-metadata.json` の `id` が正しい Kaggle スラッグになっている
- [ ] `<DIR>/` に `.ipynb` ファイルが配置されている
- [ ] `.github/workflows/kaggle-push-<slug-basename>.yml` が作成されている
- [ ] `data/<competition-slug>/` にコンペデータがダウンロードされている
- [ ] `data/<dataset-name>/` にデータセットがダウンロードされている（該当する場合）
- [ ] `data/<kernel-name>/` にカーネル出力がダウンロードされている（該当する場合）
- [ ] `KAGGLE_API_TOKEN` が GitHub Secret に登録されている（`gh secret list` で確認）
- [ ] `git push` 後に GitHub Actions が success になっている
- [ ] Kaggle 上でノートブックが更新されている

---
name: env-adapt
description: Add kaggle/runpod/local environment branching to a notebook copied from Kaggle. Use when you have a Kaggle notebook that needs to run on RunPod or MacBook locally — inserts ENV detection cell, converts /kaggle/input/ paths, adds data download commands.
argument-hint: <notebook-path>
disable-model-invocation: true
---

Kaggle からコピーしてきたノートブックに、kaggle / runpod / local 環境を自動判定する分岐コードを追加する。
パス変換・セットアップセル挿入・データダウンロードまで一括対応。

## 環境構成

| ENV | 実行場所 | データパス |
|-----|---------|-----------|
| `"kaggle"` | Kaggle Notebook | `/kaggle/input/<slug>/` |
| `"runpod"` | RunPod (SSH) | `/workspace/data/<name>/` |
| `"local"` | MacBook | `data/<name>/` |

## 使い方

```
/env-adapt <notebook-path>
```

例:
```
/env-adapt notebooks/my-solver/my-solver.ipynb
/env-adapt main/aimop-solver.ipynb
```

---

## 手順

### Step 1: ノートブックと kernel-metadata.json の読み込み

```bash
cat <notebook-path>
cat <notebook-path のディレクトリ>/kernel-metadata.json
```

`kernel-metadata.json` から以下を記録する（存在しない場合は空リスト扱い）：
- `competition_sources`
- `dataset_sources`
- `kernel_sources`
- `model_sources`

---

### Step 2: ノートブック内の環境依存箇所の調査

全セルを確認し、以下を洗い出す：

1. **`/kaggle/input/...` パス** — ENV 分岐への置き換え対象
2. **`/kaggle/working/` パス** — 出力パスの置き換え対象
3. **すでに ENV 分岐が実装済みか** — 重複挿入を避けるため確認
4. **pip インストールセル** — Kaggle 側で省略されているパッケージの確認
5. **`UserSecretsClient`** — 環境変数読み込みへの変換対象

調査結果をユーザーに報告してから次のステップに進む。

---

### Step 3: セットアップセル（先頭）の挿入

ノートブックの **最初のセル** に以下を挿入する。
`kernel-metadata.json` の内容をもとに、空コメントを残さず具体的なコマンドを埋めること。

```python
# ===== 環境セットアップ =====
import os, subprocess

def _is_kaggle():
    return os.path.exists("/kaggle/input")

def _is_runpod():
    return os.environ.get("RUNPOD_POD_ID") is not None

ENV = "kaggle" if _is_kaggle() else "runpod" if _is_runpod() else "local"
print(f"[env] {ENV}")

# ---------- RunPod ----------
if ENV == "runpod":
    # KAGGLE_USERNAME / KAGGLE_KEY は RunPod Pod 設定の環境変数に登録しておく
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    import json
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        json.dump({"username": os.environ["KAGGLE_USERNAME"],
                   "key": os.environ["KAGGLE_KEY"]}, f)
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    def _kaggle(*args):
        r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[warn] {r.stderr}")
        return r.returncode == 0

    # competition_sources
    # <competition_sources をループして以下を生成>
    # if not os.path.exists("/workspace/data/<competition-slug>"):
    #     os.makedirs("/workspace/data/<competition-slug>", exist_ok=True)
    #     _kaggle("competitions", "download", "-c", "<competition-slug>", "-p", "/workspace/data/<competition-slug>")
    #     subprocess.run(["unzip", "-q", "-o", "/workspace/data/<competition-slug>/<competition-slug>.zip", "-d", "/workspace/data/<competition-slug>"])

    # dataset_sources
    # if not os.path.exists("/workspace/data/<dataset-name>"):
    #     _kaggle("datasets", "download", "<owner>/<slug>", "-p", "/workspace/data/<dataset-name>", "--unzip")

    # kernel_sources
    # if not os.path.exists("/workspace/data/<kernel-name>"):
    #     _kaggle("kernels", "output", "<owner>/<slug>", "-p", "/workspace/data/<kernel-name>")

    print("[runpod] data ready")

# ---------- ローカル（MacBook） ----------
elif ENV == "local":
    def _kaggle(*args):
        r = subprocess.run(["uv", "run", "kaggle", *args], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[warn] {r.stderr}")
        return r.returncode == 0

    # competition_sources
    # if not os.path.exists("data/<competition-slug>"):
    #     os.makedirs("data/<competition-slug>", exist_ok=True)
    #     _kaggle("competitions", "download", "-c", "<competition-slug>", "-p", "data/<competition-slug>")
    #     subprocess.run(["unzip", "-q", "-o", "data/<competition-slug>/<competition-slug>.zip", "-d", "data/<competition-slug>"])

    # dataset_sources
    # if not os.path.exists("data/<dataset-name>"):
    #     _kaggle("datasets", "download", "<owner>/<slug>", "-p", "data/<dataset-name>", "--unzip")

    # kernel_sources
    # if not os.path.exists("data/<kernel-name>"):
    #     _kaggle("kernels", "output", "<owner>/<slug>", "-p", "data/<kernel-name>")

    print("[local] data ready")
```

---

### Step 4: パス定義セルの挿入（セル 2）

```python
# ===== パス定義 =====
if ENV == "kaggle":
    DATA_DIR   = "/kaggle/input/<competition-slug>"
    OUTPUT_DIR = "/kaggle/working"
    # <dataset_sources / kernel_sources / model_sources も同様に定義>
elif ENV == "runpod":
    DATA_DIR   = "/workspace/data/<competition-slug>"
    OUTPUT_DIR = "/workspace/output"
    # <runpod のパスを対応させる>
else:  # local (MacBook)
    DATA_DIR   = "data/<competition-slug>"
    OUTPUT_DIR = "output"
    # <local のパスを対応させる>

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"DATA_DIR={DATA_DIR}")
```

実際のスラッグ・パスを `kernel-metadata.json` の内容で埋めること。

---

### Step 5: `/kaggle/input/...` パスの変換

ノートブック内のすべての `/kaggle/input/...` および `/kaggle/working/` を ENV 分岐に置き換える。

```python
# 変換前
path = "/kaggle/input/<slug>/file.csv"

# 変換後
if ENV == "kaggle":
    _BASE = "/kaggle/input/<slug>"
elif ENV == "runpod":
    _BASE = "/workspace/data/<local-name>"
else:
    _BASE = "data/<local-name>"
path = f"{_BASE}/file.csv"
```

- `/kaggle/working/` → runpod: `/workspace/output/`、local: `output/`
- モデルパス → runpod: `/workspace/models/<name>` または HuggingFace、local: 利用不可（コメントで案内）

---

### Step 6: `UserSecretsClient` の変換（該当する場合）

```python
# 変換前
from kaggle_secrets import UserSecretsClient
secret = UserSecretsClient().get_secret("MY_KEY")

# 変換後
if ENV == "kaggle":
    from kaggle_secrets import UserSecretsClient
    secret = UserSecretsClient().get_secret("MY_KEY")
else:  # runpod / local
    secret = os.environ.get("MY_KEY", "")  # 環境変数から読み込み
```

RunPod では Pod 設定画面の環境変数、local では `.env` から `export $(cat .env | xargs)` で読み込む。

---

### Step 7: 変更サマリーをユーザーに報告

```
## env-adapt 変更サマリー

**挿入したセル:**
- セル 0: 環境セットアップ（kaggle/runpod/local 判定 + データダウンロード）
- セル 1: パス定義（DATA_DIR / OUTPUT_DIR 等）

**変換したパス:**
- セル X: `/kaggle/input/<slug>/test.csv` → ENV 分岐
- セル Y: `/kaggle/working/submission.csv` → ENV 分岐

**RunPod 利用時の準備:**
- Pod 設定画面で以下の環境変数を登録すること
  - KAGGLE_USERNAME
  - KAGGLE_KEY
  - （その他必要なシークレット）

**注意事項:**
- <model_sources に大型モデルがある場合の警告>
- `ENV == "kaggle"` の動作は元のまま変わっていない
```

---

### チェックリスト

- [ ] セットアップセルが先頭に挿入されている
- [ ] `/kaggle/input/` パスがすべて ENV 分岐（kaggle/runpod/local）に変換されている
- [ ] RunPod・local のダウンロードコマンドが具体的に記述されている（空コメント不可）
- [ ] RunPod のパス（`/workspace/data/<name>`）と local のパス（`data/<name>`）が対応している
- [ ] `UserSecretsClient` が `os.environ.get()` に変換されている（該当する場合）
- [ ] `ENV == "kaggle"` の動作が元と変わっていない
- [ ] RunPod で必要な環境変数がユーザーに伝えられている

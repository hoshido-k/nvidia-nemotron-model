---
name: kaggle-start
description: Scaffold a new notebook experiment directory under notebooks/<name>/. Creates kernel-metadata.json, GitHub Actions workflow, and either a blank notebook with ENV branching (kaggle/runpod/local) or pulls an existing Kaggle notebook and converts it.
argument-hint: <name>
disable-model-invocation: true
---

新しいノートブック実験ディレクトリを `notebooks/<name>/` にスキャフォールドする。
ベースとなる Kaggle Notebook URL を指定した場合は、そのノートブックを Pull して RunPod/ローカル対応に変換する。
URL を省略（Enter のみ）した場合は最小ノートブックを生成する。

## 環境構成

| ENV | 実行場所 | 用途 |
|-----|---------|------|
| `"kaggle"` | Kaggle Notebook (H100/A100) | 本番推論・サブミット |
| `"runpod"` | RunPod (SSH) | 試し学習・モデル動作確認 |
| `"local"` | MacBook | EDA・コード設計 |

## 使い方

```
/kaggle-start <name>
```

例:
```
/kaggle-start deepseek-r1-math
/kaggle-start answer-evolve-base
```

---

## 手順

### Step 1: 引数の確認

`$ARGUMENTS` から `NAME` を取り出す。指定がなければエラーを出して終了する。

```
DIR = notebooks/<NAME>
```

`DIR` がすでに存在する場合はユーザーに警告し、上書きするか確認してから続行する。

---

### Step 2: Kaggle ユーザー名の取得

```bash
export $(cat .env | xargs)
uv run kaggle config view
```

出力の `username` を `KAGGLE_USER` として記録する。取得できない場合はユーザーに直接確認する。

---

### Step 3: ベース Notebook URL の確認（インタラクティブ）

ユーザーに以下のメッセージを表示して入力を求める：

```
ベースにする Kaggle Notebook の URL を入力してください。
なければ Enter のみで空白を返してください。
例: https://www.kaggle.com/code/crazyzzz7/answer-evolve
```

- **URL が入力された場合** → `SOURCE_URL` として記録し、Step 3a へ進む
- **空白（Enter のみ）の場合** → `SOURCE_URL = ""` として Step 4 へスキップする

---

### Step 3a: ベース Notebook の Pull（URL あり時のみ）

URL からオーナーとスラッグを抽出する。

```bash
export $(cat .env | xargs)
uv run kaggle kernels pull <owner>/<slug> -p /tmp/kaggle-start-<NAME> --metadata
```

取得した `kernel-metadata.json` から以下を記録する：
- `dataset_sources`, `competition_sources`, `kernel_sources`, `model_sources`
- `machine_shape`, `enable_gpu`, `enable_internet`

---

### Step 3b: ノートブックを DIR にコピー（URL あり時のみ）

```bash
mkdir -p <DIR>
cp /tmp/kaggle-start-<NAME>/<slug>.ipynb <DIR>/<NAME>.ipynb
```

コピー後、**Step 5（ノートブック生成）はスキップして Step 5b（環境変換）へ進む**。

---

### Step 4: kernel-metadata.json の生成

`<DIR>/kernel-metadata.json` を作成する。URL あり時は Pull した metadata のデータソースを反映する。

```json
{
  "id": "<KAGGLE_USER>/<NAME>",
  "title": "<NAME>",
  "code_file": "<NAME>.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "false",
  "dataset_sources": [
    "amanatar/50problems"
  ],
  "competition_sources": [
    "ai-mathematical-olympiad-progress-prize-3"
  ],
  "kernel_sources": [
    "andreasbis/aimo-3-utils"
  ],
  "model_sources": [
    "danielhanchen/gpt-oss-120b/Transformers/default/1"
  ],
  "machine_shape": "NvidiaH100"
}
```

---

### Step 5: 最小ノートブックの生成（URL なし時のみ）

`<DIR>/<NAME>.ipynb` を以下の4セル構成で生成する。

**セル 1: 環境セットアップ**

```python
# ===== 環境セットアップ =====
import os, subprocess

def _is_kaggle():
    return os.path.exists("/kaggle/input")

def _is_runpod():
    return os.environ.get("RUNPOD_POD_ID") is not None

ENV = "kaggle" if _is_kaggle() else "runpod" if _is_runpod() else "local"
print(f"[env] {ENV}")

# ---------- RunPod セットアップ ----------
if ENV == "runpod":
    # KAGGLE_API_TOKEN は RunPod の環境変数（Pod 設定画面）に登録しておく
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

    if not os.path.exists("/workspace/data/aimo3"):
        os.makedirs("/workspace/data/aimo3", exist_ok=True)
        _kaggle("competitions", "download", "-c", "ai-mathematical-olympiad-progress-prize-3", "-p", "/workspace/data/aimo3")
        subprocess.run(["unzip", "-q", "-o", "/workspace/data/aimo3/ai-mathematical-olympiad-progress-prize-3.zip", "-d", "/workspace/data/aimo3"])
    if not os.path.exists("/workspace/data/50problems"):
        _kaggle("datasets", "download", "amanatar/50problems", "-p", "/workspace/data/50problems", "--unzip")
    if not os.path.exists("/workspace/data/aimo-utils"):
        _kaggle("kernels", "output", "andreasbis/aimo-3-utils", "-p", "/workspace/data/aimo-utils")
    print("[runpod] data ready")

# ---------- ローカル（MacBook）セットアップ ----------
elif ENV == "local":
    def _kaggle(*args):
        r = subprocess.run(["uv", "run", "kaggle", *args], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[warn] {r.stderr}")
        return r.returncode == 0

    if not os.path.exists("data/aimo3"):
        os.makedirs("data/aimo3", exist_ok=True)
        _kaggle("competitions", "download", "-c", "ai-mathematical-olympiad-progress-prize-3", "-p", "data/aimo3")
        subprocess.run(["unzip", "-q", "-o", "data/aimo3/ai-mathematical-olympiad-progress-prize-3.zip", "-d", "data/aimo3"])
    if not os.path.exists("data/50problems"):
        _kaggle("datasets", "download", "amanatar/50problems", "-p", "data/50problems", "--unzip")
    if not os.path.exists("data/aimo-utils"):
        _kaggle("kernels", "output", "andreasbis/aimo-3-utils", "-p", "data/aimo-utils")
    print("[local] data ready")
```

**セル 2: パス定義**

```python
# ===== パス定義 =====
if ENV == "kaggle":
    DATA_DIR   = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3"
    PROB_DIR   = "/kaggle/input/amanatar/50problems"
    UTILS_DIR  = "/kaggle/input/andreasbis/aimo-3-utils/aimo-3-utils"
    MODEL_DIR  = "/kaggle/input/danielhanchen/gpt-oss-120b/Transformers/default/1"
    OUTPUT_DIR = "/kaggle/working"
elif ENV == "runpod":
    DATA_DIR   = "/workspace/data/aimo3"
    PROB_DIR   = "/workspace/data/50problems"
    UTILS_DIR  = "/workspace/data/aimo-utils"
    MODEL_DIR  = "/workspace/models/gpt-oss-120b"  # 手動配置 or HuggingFace
    OUTPUT_DIR = "/workspace/output"
else:  # local (MacBook)
    DATA_DIR   = "data/aimo3"
    PROB_DIR   = "data/50problems"
    UTILS_DIR  = "data/aimo-utils"
    MODEL_DIR  = None  # MacBook では大型モデルは使わない
    OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"DATA_DIR={DATA_DIR}")
```

**セル 3: データ読み込み確認**

```python
# ===== データ読み込み =====
import pandas as pd

test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"test shape: {test_df.shape}")
print(test_df.head(2))
```

**セル 4: 提出ファイル雛形**

```python
# ===== 提出ファイル =====
import pandas as pd

submission = pd.DataFrame({
    "id": test_df["id"],
    "answer": 0
})
submission.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
print("submission.csv saved")
```

---

### Step 5b: Pull したノートブックを RunPod/ローカル対応に変換（URL あり時のみ）

`/env-adapt` スキルと同等の変換を行う。

1. **全セルを読んで調査する**（`/kaggle/input/` パス・`UserSecretsClient` 等）
2. **セットアップセルをノートブック先頭に挿入する**（上記セル 1 のテンプレートを使い、データソースを実際のスラッグで埋める）
3. **パス定義セルを挿入する**（上記セル 2 のテンプレートを使う）
4. **`/kaggle/input/...` パスをすべて ENV 分岐に変換する**

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

5. **`UserSecretsClient` を ENV 分岐に変換する**

   ```python
   if ENV == "kaggle":
       from kaggle_secrets import UserSecretsClient
       secret = UserSecretsClient().get_secret("MY_KEY")
   else:  # runpod / local
       secret = os.environ.get("MY_KEY", "")  # 環境変数から読み込み
   ```

6. **変換サマリーをユーザーに報告する**

---

### Step 6: Python スクリプトの有無を確認

`notebooks/<NAME>/` に `.py` ファイルが存在する場合は **Kaggle Dataset として管理**する。

#### Step 6a: Python スクリプトがある場合

`kaggle-datasets/<NAME>-scripts/` を作成し、スクリプトをコピーして Dataset を作成する。

```bash
mkdir -p kaggle-datasets/<NAME>-scripts

# .py ファイルをコピー
cp notebooks/<NAME>/*.py kaggle-datasets/<NAME>-scripts/

# dataset-metadata.json を作成
cat > kaggle-datasets/<NAME>-scripts/dataset-metadata.json << 'EOF'
{
  "title": "<NAME>-scripts",
  "id": "<KAGGLE_USER>/<NAME>-scripts",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

export $(cat .env | xargs)
uv run kaggle datasets create -p kaggle-datasets/<NAME>-scripts
```

`kernel-metadata.json` の `dataset_sources` に追加する：

```json
"dataset_sources": [
  "<KAGGLE_USER>/<NAME>-scripts"
]
```

ノートブックのパス定義セルに以下を追加する：

```python
# Python スクリプトを Input からコピー
if ENV == "kaggle":
    import shutil
    for py_file in os.listdir("/kaggle/input/<NAME>-scripts"):
        if py_file.endswith(".py"):
            shutil.copy2(
                f"/kaggle/input/<NAME>-scripts/{py_file}",
                f"/kaggle/working/{py_file}"
            )
    print("[script] copied from input")
```

#### Step 6b: Python スクリプトがない場合

スキップする。

---

### Step 7: GitHub Actions ワークフローの生成

`.github/workflows/kaggle-push-<NAME>.yml` を作成する。

**Python スクリプトがある場合**（`push-notebook` と `push-scripts` の2ジョブ構成）：

```yaml
name: Push <NAME> to Kaggle

on:
  push:
    branches:
      - main
    paths:
      - "notebooks/<NAME>/**"
      - "kaggle-datasets/<NAME>-scripts/**"
  workflow_dispatch:

jobs:
  push-notebook:
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
          kaggle kernels push -p notebooks/<NAME>

  push-scripts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Kaggle CLI
        run: pip install kaggle

      - name: Sync scripts to Kaggle Dataset
        env:
          KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}
        run: |
          export KAGGLE_TOKEN=$KAGGLE_API_TOKEN
          cp notebooks/<NAME>/*.py kaggle-datasets/<NAME>-scripts/
          kaggle datasets version -p kaggle-datasets/<NAME>-scripts \
            -m "auto-sync from git $(git rev-parse --short HEAD)"
```

**Python スクリプトがない場合**（`push` の1ジョブ構成）：

```yaml
name: Push <NAME> Notebook to Kaggle

on:
  push:
    branches:
      - main
    paths:
      - "notebooks/<NAME>/**"
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
          kaggle kernels push -p notebooks/<NAME>
```

---

### Step 8: ローカル（MacBook）データの確認・ダウンロード

```bash
export $(cat .env | xargs)

if [ ! -d "data/aimo3" ]; then
  mkdir -p data/aimo3
  uv run kaggle competitions download -c ai-mathematical-olympiad-progress-prize-3 -p data/aimo3
  unzip -q -o data/aimo3/ai-mathematical-olympiad-progress-prize-3.zip -d data/aimo3
fi

if [ ! -d "data/50problems" ]; then
  uv run kaggle datasets download amanatar/50problems -p data/50problems --unzip
fi

if [ ! -d "data/aimo-utils" ]; then
  uv run kaggle kernels output andreasbis/aimo-3-utils -p data/aimo-utils
fi
```

モデル（120B）はローカル・RunPod ともに手動配置または HuggingFace からロード。

---

### Step 9: KAGGLE_API_TOKEN の GitHub Secret 確認

```bash
export $(cat .env | xargs)
gh secret list
```

未登録の場合のみ：

```bash
gh secret set KAGGLE_API_TOKEN --body "$KAGGLE_API_TOKEN"
```

---

### Step 10: サマリー報告

```
## kaggle-start 完了: <NAME>

**モード:** URL あり（Pull + 環境変換） / URL なし（最小ノートブック生成）

**作成・変換したファイル:**
- notebooks/<NAME>/kernel-metadata.json
- notebooks/<NAME>/<NAME>.ipynb  （kaggle/runpod/local ENV 分岐付き）
- .github/workflows/kaggle-push-<NAME>.yml  （ノートブック + スクリプト自動デプロイ）
- （.py あり）kaggle-datasets/<NAME>-scripts/  → Kaggle Dataset として管理

**ローカルデータ（MacBook）:**
- data/<source>/  → 各データソース（あり / 新規取得）

**自動デプロイの流れ:**
- notebooks/<NAME>/ または kaggle-datasets/<NAME>-scripts/ を変更して git push
  → GitHub Actions が Kaggle Notebook と Dataset を同時に更新

**RunPod 利用時の注意:**
- Pod 設定画面で KAGGLE_USERNAME / KAGGLE_KEY を環境変数に登録しておくこと
- データは /workspace/data/ に自動ダウンロードされる

**次のステップ:**
1. notebooks/<NAME>/<NAME>.ipynb を開いてメイン実装を追記
2. Kaggle 上で Copy & Edit 済みの場合は kernel-metadata.json の id を更新
3. git push → GitHub Actions で自動デプロイ → Kaggle で Run All
```

---

### チェックリスト

- [ ] `notebooks/<NAME>/kernel-metadata.json` が生成されている
- [ ] `notebooks/<NAME>/<NAME>.ipynb` が生成されている（kaggle/runpod/local 分岐付き）
- [ ] `.github/workflows/kaggle-push-<NAME>.yml` が生成されている（.py ありなら2ジョブ構成）
- [ ] （.py あり）`kaggle-datasets/<NAME>-scripts/` が作成され Kaggle Dataset が存在する
- [ ] （.py あり）`kernel-metadata.json` の `dataset_sources` にスクリプト Dataset が追加されている
- [ ] （.py あり）ノートブックに Input からスクリプトをコピーするセルが追加されている
- [ ] （URL あり）`/kaggle/input/...` パスがすべて ENV 分岐に変換されている
- [ ] ローカルデータ（MacBook 用）が `data/` に揃っている
- [ ] `KAGGLE_API_TOKEN` が GitHub Secret に登録されている
- [ ] RunPod を使う場合、Pod 設定に `KAGGLE_USERNAME` / `KAGGLE_KEY` が登録されている

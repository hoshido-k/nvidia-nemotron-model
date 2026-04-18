# ネクストアクション計画

最終更新: 2026-04-17

---

## 現状

- 学習: `nemotron-train` ノートブックで Run & All（Commit）実行中
- 推論ノートブック: 我々のアダプタ向けに整備済み
- LB スコア: 未確認（学習完了待ち）

---

## Step 1: 学習完了 → アダプタアップロード → LB 確認

1. 学習ノートブックの完走を待つ
2. 出力された adapter ファイルを Kaggle モデル `shotokishida/nemotron-adapter` にアップロード
   ```bash
   # adapter_config.json + adapter_model.safetensors を手動アップロード
   # または kaggle models create / version create コマンドで
   ```
3. 推論ノートブック（`nvidia-nemotron-model-reasoning-challenge`）を Run & All で提出
4. **LB スコアを確認**

---

## Step 2: スコアに応じた改善（LB が低い場合）

### 優先度 高: CoT データ蒸留

**方針:** 全 9,500 問を Gemini 2.5 Pro で CoT 生成し、現行データセットを置き換え

| 項目 | 詳細 |
|---|---|
| 対象 | `train.csv` の全 9,500 件 |
| 使用モデル | Gemini 2.5 Pro（高品質）または Flash（高速・安価） |
| フィルタ | 答えが正解 → 採用、不正解 → konbu17 既存 CoT にフォールバック |
| 現行データとの比較 | konbu17 は 69%（6,558件）しかカバーしていない・CoT が短くシンプル |

**実装するもの:**
- `scripts/generate_cot.py`: Gemini API で CoT を一括生成
  - 非同期・並列リクエスト
  - 答えの正誤検証（数値は誤差許容）
  - チェックポイント保存（途中再開できるように）
- 出力: `train_split_with_cot_gemini.csv`（konbu17 と同じ形式）

**API レート制限の目安:**
- 無料枠: ~5 RPM → 9,500件で約 32時間（並列化で短縮可能）
- Paid: ~60 RPM → 約 2.5時間

---

### 優先度 中: DR-LoRA（Dynamic Rank LoRA）

**概要:** 層ごとに異なる LoRA rank を割り当てる。MoE × Hybrid 構造に直接マッチ。

**このモデルへの有効性: 高**

| コンポーネント | 現行 | DR-LoRA |
|---|---|---|
| Attention 層（q/k/v/o_proj） | rank=32 | rank 高め（推論に重要） |
| Mamba 層（in/out_proj） | rank=32 | rank 中程度 |
| MoE FFN（up/down_proj × experts） | rank=32 | expert ごとに差をつける |

**実装:** PEFT の `rank_pattern` / `lora_alpha_pattern` で指定可能。現行の SFT フレームワークそのまま。

```python
LoraConfig(
    r=32,
    rank_pattern={"q_proj": 32, "k_proj": 32, "v_proj": 32, "in_proj": 16, ...},
    ...
)
```

**判断基準:** CoT 蒸留後もスコアが伸び悩んだら検討

---

### 優先度 低: GRPO

**概要:** 強化学習ベースの学習手法。正解/不正解を報酬にしてポリシーを最適化。

**このモデルへの有効性: 不確か（リスク高）**

- on-policy 生成（学習中に推論）が必要 → Mamba のカスタム CUDA カーネルとの競合リスク
- Hybrid Mamba-Transformer での GRPO 動作報告がほぼなし
- SFT の 3〜5 倍の学習時間

**判断基準:** DR-LoRA でも頭打ちになったら、動作確認を兼ねて試す

---

## Step 3: その他の改善候補

| アイデア | 効果予測 | コスト |
|---|---|---|
| 複数回生成して多数決（self-consistency） | 中〜高 | 推論時間 3〜5倍 |
| lora_alpha 調整 | 小 | 低 |
| 問題タイプ別に学習データを重み付け | 中 | 中 |

---

## 参考情報

- コンペ: [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
- 参照実装: dgxchen（LB 0.81〜0.85）, Tong Hui Kang
- 学習データ: konbu17/nemotron-sft-lora-cot-selection（9,500件中 6,558件・CoT 質は低め）
- LoRA rank 上限: 32（vLLM の `max_lora_rank=32` による制約）

# NVIDIA Nemotron Model Reasoning Challenge - 実験計画

## コンペ概要

- **目標**: nemotron-3-nano-30b に LoRA アダプタを学習・提出し、数学推論の精度を競う
- **提出物**: `adapter_model.safetensors` + `adapter_config.json` を zip 化したもの
- **評価**: テスト問題の正答率（Accuracy）

---

## フェーズ別計画

### Phase 1: 環境構築 + ベースライン把握

**目標**: スコアボードに乗る

- [ ] RunPod（RTX 6000 Ada, 48GB）セットアップ
- [ ] 現ノートブックが RunPod 上で動くことを確認
- [ ] 既存アダプタ（`huikang/nvidia-nemotron-all-linear`）をそのまま提出してベーススコアを把握
- [ ] `data/nvidia-nemotron-all-linear/submission.zip` の中身を解析（学習コードの確認）
- [ ] Kaggle 上で Run All が通ることを確認

---

### Phase 2: LoRA ファインチューニング

**目標**: ベースラインより +数ポイント

**使用データ**
- `amanatar/50problems`（ダウンロード済み）
- 公開数学データセット（MATH, AMC/AIME 過去問）
- コンペ提供トレーニングデータ

**実装ステップ**
- [ ] `huikang/nvidia-nemotron-all-linear` の学習コードを読んで理解
- [ ] LoRA rank=8 で動作確認
- [ ] rank 8 → 16 → 32 で精度変化を計測
- [ ] 学習率・epoch・データ量のパラメータ探索
- [ ] Kaggle RTX Pro 6000（96GB）で本番学習

**試すパラメータ**
| パラメータ | 候補値 |
|-----------|--------|
| LoRA rank | 8, 16, 32 |
| learning_rate | 1e-4, 5e-5, 2e-5 |
| epochs | 1, 2, 3 |
| target_modules | q_proj/v_proj のみ → 全 linear |

---

### Phase 3: Gemini 2.5 Pro 蒸留

**目標**: Phase 2 からさらに +数ポイント  
**発動条件**: Phase 2 で伸び悩んだ場合

**フロー**
```
[Kaggle Notebook A: internet: true]
  Gemini 2.5 Pro API
    → テスト問題に対する CoT 推論トレースを生成
    → /kaggle/working/train_data.jsonl として保存
    → Kaggle Dataset として公開（private）

[Kaggle Notebook B: internet: false]
  生成データ Dataset をマウント
    → nemotron-30b + LoRA で学習
    → adapter を /kaggle/working に保存 → submission.zip
```

**コスト試算**
- 問題数が少ないため Gemini API 代は数百円程度の見込み

---

### Phase 4: 高度な手法（余裕があれば）

- **テスト時学習（TTT）**: テスト問題に特化してファインチューニング
- **セルフプレイ**: モデル自身が生成した正解例で反復学習
- **QLoRA 4-bit**: メモリ削減で大きなバッチサイズを試す
- **アンサンブル**: 複数アダプタの出力を統合

---

## 実行環境

| 用途 | 環境 | GPU |
|------|------|-----|
| コード動作確認・デバッグ | RunPod | RTX 6000 Ada 48GB |
| 蒸留データ生成 | Kaggle (internet: true) | 不要 |
| LoRA 本番学習 | Kaggle | RTX Pro 6000 96GB |
| 長時間実験 | RunPod | RTX 6000 Ada 48GB |

---

## スケジュール（目安）

| 週 | 内容 |
|----|------|
| 今週 | Phase 1（環境整備・ベースライン提出） |
| 来週 | Phase 2 前半（学習コード理解・LoRA rank=8 で動作確認） |
| 再来週 | Phase 2 後半（パラメータ探索・スコア改善） |
| 残り期間 | スコア次第で Phase 3 / Phase 4 へ |

---

## 参考リソース

- 公開ベスト解: `huikang/nvidia-nemotron-all-linear`
- 公式ユーティリティ: `metric/nvidia-metric-utility-script`
- コンペデータ: `nvidia-nemotron-model-reasoning-challenge`
- 追加問題セット: `amanatar/50problems`

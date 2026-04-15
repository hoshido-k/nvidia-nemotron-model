"""
LoRA SFT fine-tuning script for nemotron-3-nano-30b.

学習データ形式（CSV）:
  - prompt      : 問題文
  - answer      : 答え
  - generated_cot: 推論過程（CoT）※任意。なければ answer のみで学習

Usage:
    python train.py \
        --model_dir /path/to/model \
        --data_csv  /path/to/train_split_with_cot.csv \
        --output_dir /path/to/output \
        --lora_rank 32 \
        --epochs 2
"""

import argparse
import os
import sys
import json
import shutil
import stat
import zipfile
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",   required=True,  help="ベースモデルのパス")
    p.add_argument("--data_csv",    required=True,  help="学習データ CSV のパス")
    p.add_argument("--output_dir",  required=True,  help="アダプタの保存先")
    p.add_argument("--extra_csv",   default=None,   help="追加データ CSV（任意）")
    p.add_argument("--lora_rank",   type=int,   default=32)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--lora_dropout",type=float, default=0.0)
    p.add_argument("--epochs",      type=int,   default=2)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--max_seq_len", type=int,   default=2048)
    p.add_argument("--subsample",   type=int,   default=None, help="データをサブサンプリング（動作確認用）")
    p.add_argument("--zip_output",  action="store_true", help="アダプタを submission.zip に圧縮")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Kaggle Blackwell (RTX Pro 6000) 向け Triton パッチ
# ---------------------------------------------------------------------------

def apply_triton_patch():
    """Kaggle の RTX Pro 6000 (Blackwell) で学習するために必要なパッチ。"""
    import torch.nn.functional as F

    # RMSNorm を pure PyTorch 実装に置き換え（Triton カーネルのクラッシュ回避）
    def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                         group_size=None, norm_before_gate=True, upcast=True):
        dtype = x.dtype
        if upcast:
            x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        out = x_normed * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None:
            out = out * F.silu(z.float())
        return out.to(dtype)

    for name, mod in list(sys.modules.items()):
        if hasattr(mod, "rmsnorm_fn"):
            mod.rmsnorm_fn = _pure_rmsnorm_fn

    # ptxas-blackwell バイナリをコピー
    ptxas_src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell"
    ptxas_dst = "/tmp/ptxas-blackwell"
    if os.path.exists(ptxas_src) and not os.path.exists(ptxas_dst):
        shutil.copy2(ptxas_src, ptxas_dst)
        os.chmod(ptxas_dst, os.stat(ptxas_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print("[patch] ptxas-blackwell copied to /tmp")

    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_dst
    print("[patch] Triton patch applied")


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------

PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

def load_dataset(data_csv: str, extra_csv: str | None, subsample: int | None) -> Dataset:
    """CSV を読み込み HuggingFace Dataset に変換する。"""
    dfs = []

    for path in [data_csv, extra_csv]:
        if path and Path(path).exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"[data] {path}: {len(df)} rows")
        elif path:
            print(f"[warn] not found: {path}")

    if not dfs:
        raise FileNotFoundError(f"No data found: {data_csv}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["prompt", "answer"])

    if subsample:
        df = df.sample(n=min(subsample, len(df)), random_state=42)
        print(f"[data] subsampled to {len(df)} rows")

    print(f"[data] total: {len(df)} rows")
    if "type" in df.columns:
        print(df["type"].value_counts().to_string())

    return Dataset.from_pandas(df.reset_index(drop=True))


def build_training_text(tokenizer, example: dict) -> str:
    """1サンプルを chat template 形式に変換する。

    CoT あり: 推論過程 + \\boxed{answer}
    CoT なし: answer のみ
    """
    cot = example.get("generated_cot", "")
    answer = str(example["answer"])
    prompt = example["prompt"]

    user_msg = prompt + PROMPT_SUFFIX

    if cot and str(cot).strip():
        assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"
    else:
        assistant_msg = f"\\boxed{{{answer}}}"

    messages = [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # chat template が使えない場合のフォールバック
        return f"User: {user_msg}\nAssistant: {assistant_msg}"


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

def mock_mamba_ssm():
    """mamba-ssm/causal-conv1d のモックを sys.modules に差し込む。

    trust_remote_code=True で読み込まれるカスタムコードが mamba-ssm を
    import しようとするが、RTX Pro 6000 環境ではインストール不可。
    モックを差し込むことで ImportError を回避し、ナイーブ実装にフォールバックさせる。
    """
    from unittest.mock import MagicMock

    for mod_name in [
        "mamba_ssm",
        "mamba_ssm.ops",
        "mamba_ssm.ops.triton",
        "mamba_ssm.ops.triton.layernorm_gated",
        "mamba_ssm.ops.selective_scan_interface",
        "causal_conv1d",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    print("[patch] mamba-ssm mock injected (naive fallback enabled)")


def load_model_and_tokenizer(model_dir: str):
    """BF16 でモデルとトークナイザを読み込む。"""
    print(f"[model] loading from {model_dir} ...")

    # mamba-ssm が不要になるよう事前にモックを注入
    mock_mamba_ssm()

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Nemotron 特有の fast path を無効化（学習時の安定性確保）
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = False

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] loaded. dtype={next(model.parameters()).dtype}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj", "up_proj", "down_proj",
    "lm_head",
]

def apply_lora(model, args):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Kaggle 環境なら Triton パッチを適用
    if os.path.exists("/kaggle"):
        apply_triton_patch()

    # データ
    dataset = load_dataset(args.data_csv, args.extra_csv, args.subsample)

    # モデル・トークナイザ
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    model = apply_lora(model, args)

    # テキスト変換
    def formatting_func(samples):
        return [
            build_training_text(
                tokenizer,
                {k: samples[k][i] for k in samples}
            )
            for i in range(len(samples["prompt"]))
        ]

    # 学習設定
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="no",        # Kaggle の容量節約
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_len,
    )

    print("[train] start ...")
    trainer.train()

    # アダプタ保存
    print(f"[train] saving adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # adapter_config 確認
    cfg_path = Path(args.output_dir) / "adapter_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            print("[train] adapter_config:", json.dumps(json.load(f), indent=2))

    # submission.zip 作成
    if args.zip_output:
        zip_path = str(Path(args.output_dir).parent / "submission.zip")
        required = ["adapter_config.json", "adapter_model.safetensors"]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(args.output_dir):
                fpath = os.path.join(args.output_dir, fname)
                if os.path.isfile(fpath):
                    zf.write(fpath, fname)
        for req in required:
            if req not in os.listdir(args.output_dir):
                raise AssertionError(f"CRITICAL: {req} が見つかりません。提出に失敗します。")
        size_mb = os.path.getsize(zip_path) / 1024 / 1024
        print(f"[train] submission.zip saved ({size_mb:.1f} MB): {zip_path}")

    print("[train] done.")


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)

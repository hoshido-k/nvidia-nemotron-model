"""
LoRA / QLoRA fine-tuning script for nemotron-3-nano-30b.

Usage:
    python train.py \
        --model_dir /path/to/model \
        --data_dir  /path/to/competition/data \
        --output_dir /path/to/output \
        --lora_rank 16 \
        --epochs 2 \
        --use_4bit
"""

import argparse
import os
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",   required=True,  help="ベースモデルのパス")
    p.add_argument("--data_dir",    required=True,  help="train.csv があるディレクトリ")
    p.add_argument("--output_dir",  required=True,  help="アダプタの保存先")
    p.add_argument("--extra_data",  default=None,   help="追加データCSVのパス（任意）")
    p.add_argument("--lora_rank",   type=int, default=16)
    p.add_argument("--lora_alpha",  type=int, default=32)
    p.add_argument("--lora_dropout",type=float, default=0.05)
    p.add_argument("--epochs",      type=int, default=2)
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--grad_accum",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--use_4bit",    action="store_true", help="QLoRA 4-bit 量子化")
    p.add_argument("--use_8bit",    action="store_true", help="8-bit 量子化")
    return p.parse_args()


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str, extra_data: str | None) -> Dataset:
    """train.csv を読み込み、chat 形式に変換する。"""
    dfs = []

    train_csv = Path(data_dir) / "train.csv"
    if train_csv.exists():
        dfs.append(pd.read_csv(train_csv))
        print(f"[data] train.csv: {len(dfs[-1])} rows")
    else:
        print(f"[warn] train.csv not found: {train_csv}")

    if extra_data and Path(extra_data).exists():
        dfs.append(pd.read_csv(extra_data))
        print(f"[data] extra_data: {len(dfs[-1])} rows")

    if not dfs:
        raise FileNotFoundError(f"No data found in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["prompt", "answer"])
    print(f"[data] total: {len(df)} rows")

    return Dataset.from_pandas(df[["prompt", "answer"]])


def format_sample(tokenizer, sample: dict) -> str:
    """1サンプルを chat template 形式に変換する。"""
    messages = [
        {"role": "user",      "content": sample["prompt"]},
        {"role": "assistant", "content": str(sample["answer"])},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_dir: str, use_4bit: bool, use_8bit: bool):
    """ベースモデルとトークナイザを読み込む。"""
    print(f"[model] loading from {model_dir} ...")

    # 量子化設定
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("[model] QLoRA 4-bit (NF4) enabled")
    elif use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print("[model] 8-bit quantization enabled")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not (use_4bit or use_8bit) else None,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # 学習時は無効化

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

# 参照アダプタ（huikang/nvidia-nemotron-all-linear）に合わせた target_modules
TARGET_MODULES = [
    "k_proj", "o_proj", "in_proj", "q_proj",
    "up_proj", "v_proj", "down_proj", "out_proj", "lm_head",
]

def apply_lora(model, args):
    """LoRA を適用して学習対象パラメータを絞る。"""
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

    # データ
    dataset = load_dataset(args.data_dir, args.extra_data)

    # モデル
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir, args.use_4bit, args.use_8bit
    )
    model = apply_lora(model, args)

    # テキスト変換（SFTTrainer に渡す形式）
    def formatting_func(samples):
        return [format_sample(tokenizer, {"prompt": p, "answer": a})
                for p, a in zip(samples["prompt"], samples["answer"])]

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
        save_strategy="epoch",
        max_seq_length=args.max_seq_len,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
    )

    # 学習
    print("[train] start training ...")
    trainer.train()

    # アダプタ保存
    print(f"[train] saving adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 設定ファイルのサマリー出力
    adapter_config_path = Path(args.output_dir) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            print("[train] adapter_config:", json.dumps(json.load(f), indent=2))

    print("[train] done.")


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)

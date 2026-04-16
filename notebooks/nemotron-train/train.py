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

    # 4bit量子化（QLoRA）— VRAM 節約・高速化
    python train.py ... --load_in_4bit

    # 動作確認用サブサンプリング
    python train.py ... --subsample 100
"""

import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import json
import shutil
import stat
import time
import zipfile
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",    required=True,  help="ベースモデルのパス")
    p.add_argument("--data_csv",     required=True,  help="学習データ CSV のパス")
    p.add_argument("--output_dir",   required=True,  help="アダプタの保存先")
    p.add_argument("--extra_csv",    default=None,   help="追加データ CSV（任意）")
    p.add_argument("--lora_rank",    type=int,   default=32)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--epochs",       type=int,   default=2)
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--max_seq_len",  type=int,   default=4096)
    p.add_argument("--subsample",    type=int,   default=None, help="データをサブサンプリング（動作確認用）")
    p.add_argument("--zip_output",   action="store_true", help="アダプタを submission.zip に圧縮")
    p.add_argument("--load_in_4bit", action="store_true", help="4bit量子化でロード（QLoRA）。VRAM 節約・高速化")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Kaggle 環境セットアップ
# ---------------------------------------------------------------------------

def setup_kaggle_env():
    """Kaggle 環境の依存パッケージをセットアップする。

    mamba_ssm は UTILITY SCRIPTS に含まれており自動的に使用される。
    ただし UTILITY SCRIPTS の mamba_ssm (Mamba3版) は cutlass を必要とするため
    cutlass パスを sys.path に追加する。

    4bit QLoRA を使う場合は bitsandbytes が必要。
    dennisfong/nvidia-nemotron-offline-packages を Input に追加しておく。
    """
    import site

    # cutlass パスを追加（UTILITY SCRIPTS の mamba_ssm が必要とする）
    cutlass_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages/"
    if os.path.exists(cutlass_path):
        site.addsitedir(cutlass_path)
        print(f"[setup] cutlass path added: {cutlass_path}")
    else:
        raise FileNotFoundError(
            f"cutlass が見つかりません: {cutlass_path}\n"
            "UTILITY SCRIPTS (ryanholbrook/nvidia-utility-script) を Input に追加してください。"
        )

    # bitsandbytes（QLoRA 用）: dennisfong/nvidia-nemotron-offline-packages から
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        bnb_wheels = sorted(glob.glob("/kaggle/input/**/bitsandbytes*.whl", recursive=True))
        if bnb_wheels:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", bnb_wheels[-1]],
                check=True,
            )
            print(f"[setup] installed: {Path(bnb_wheels[-1]).name}")
        else:
            print("[setup] bitsandbytes wheel not found (QLoRA --load_in_4bit は使用不可)")


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
    os.environ["TRITON_PTXAS_PATH"] = ptxas_dst

    # Triton JIT キャッシュを永続化（セッション再起動時の再コンパイルを回避）
    triton_cache = "/kaggle/working/.triton_cache"
    os.makedirs(triton_cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = triton_cache
    print(f"[patch] Triton cache dir: {triton_cache}")

    # Triton のバージョンチェックをパッチ（Blackwell では ptxas 12.0 扱いにする）
    try:
        import triton.backends.nvidia.compiler as nv_compiler
        nv_compiler.get_ptxas_version = lambda arch: "12.0"
    except Exception:
        pass

    print("[patch] Triton patch applied")


def optimize_gpu():
    """RTX Pro 6000 (Blackwell) の性能を最大限に引き出す GPU 設定。"""
    if not torch.cuda.is_available():
        return

    # TF32: Ampere 以降で FP32 演算を TF32 で高速化（精度低下はほぼなし）
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN: 同一サイズの入力が続く学習では benchmark モードが最速
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # CUDA メモリアロケータの最適化
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("[gpu] TF32 enabled, cuDNN benchmark enabled, expandable_segments enabled")


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------

PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

def _compute_cache_key(data_csv: str, extra_csv: str | None, subsample: int | None) -> str:
    """CSV ファイルの内容とパラメータからキャッシュキーを生成する。"""
    h = hashlib.sha256()
    for path in [data_csv, extra_csv]:
        if path and Path(path).exists():
            h.update(Path(path).read_bytes())
    h.update(f"subsample={subsample}".encode())
    return h.hexdigest()[:16]


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


def load_formatted_dataset(
    data_csv: str,
    extra_csv: str | None,
    subsample: int | None,
    tokenizer,
    cache_dir: str = "/kaggle/working/.dataset_cache",
) -> Dataset:
    """前処理済みデータセットをキャッシュ付きでロードする。

    初回: CSV → chat template 変換 → Arrow 形式でディスク保存
    2回目以降: ディスクから即座にロード（数秒）

    キャッシュキーは CSV の内容ハッシュで決定するため、
    データが変わらなければセッション再起動後も再利用される。
    """
    cache_key = _compute_cache_key(data_csv, extra_csv, subsample)
    cache_path = Path(cache_dir) / cache_key

    if cache_path.exists():
        t0 = time.time()
        text_dataset = Dataset.load_from_disk(str(cache_path))
        print(f"[data] cache hit: {cache_path} ({len(text_dataset)} samples, {time.time() - t0:.1f}s)")
        return text_dataset

    # キャッシュがない場合は通常ロード → 変換 → 保存
    dataset = load_dataset(data_csv, extra_csv, subsample)

    t0 = time.time()
    text_dataset = dataset.map(
        lambda example: {"text": build_training_text(tokenizer, example)},
        remove_columns=dataset.column_names,
        desc="Formatting dataset",
        num_proc=os.cpu_count(),
    )
    print(f"[data] formatted {len(text_dataset)} samples ({time.time() - t0:.1f}s)")

    # キャッシュ保存
    os.makedirs(cache_dir, exist_ok=True)
    text_dataset.save_to_disk(str(cache_path))
    print(f"[data] cache saved: {cache_path}")

    return text_dataset


def build_training_text(tokenizer, example: dict) -> str:
    """1サンプルを chat template 形式に変換する。

    CoT あり: CoT（\boxed{}除去済み）+ </think>\n\boxed{answer}
    CoT なし: \boxed{answer} のみ

    chat template が <think>\n を自動付加するため assistant は CoT から始まる。
    """
    cot    = example.get("generated_cot", "")
    answer = str(example["answer"])
    prompt = example["prompt"]

    user_msg = prompt + PROMPT_SUFFIX

    if cot and str(cot).strip() and len(str(cot).strip()) >= 5:
        # CoT 内の \boxed{} を除去してクリーンな推論テキストにする
        cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', str(cot)).rstrip()
        assistant_msg = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
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
        return f"User: {user_msg}\nAssistant: {assistant_msg}"


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_dir: str, load_in_4bit: bool = False):
    """モデルとトークナイザを読み込む。

    load_in_4bit=True: NF4 量子化（QLoRA）。VRAM を大幅削減し高速化。
    load_in_4bit=False: BF16 フル精度。本番学習向け。
    """
    print(f"[model] loading from {model_dir} ...")

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        print("[model] loaded. 4-bit NF4 quantized (QLoRA)")
        print(f"[model] device: {next(model.parameters()).device}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        print(f"[model] loaded. dtype={next(model.parameters()).dtype}, device={next(model.parameters()).device}")

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

# Mamba 系レイヤーのみを対象とする（Kaggle ノートブックと同じ）
TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"

def apply_lora(model, args):
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        # Nemotron-H MoE の index_add_ で BF16/FP32 の型不一致が起きるためパッチ
        _orig_index_add_ = torch.Tensor.index_add_
        def _patched_index_add_(self, dim, index, source, *args, **kwargs):
            if source.dtype != self.dtype:
                source = source.to(self.dtype)
            return _orig_index_add_(self, dim, index, source, *args, **kwargs)
        torch.Tensor.index_add_ = _patched_index_add_
        print("[patch] index_add_ dtype mismatch patch applied")
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

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

    # GPU 最適化（TF32, cuDNN benchmark 等）
    optimize_gpu()

    # GPU 確認
    print(f"[env] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[env] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[env] VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[env] WARNING: CUDA not available — running on CPU!")

    # Kaggle 環境: offline wheel から mamba_ssm をインストール → Triton パッチ
    if os.path.exists("/kaggle"):
        setup_kaggle_env()
        apply_triton_patch()

    # モデル・トークナイザ（データロードより先にトークナイザが必要）
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.load_in_4bit)
    model = apply_lora(model, args)

    # データ（キャッシュ付き: 2回目以降は数秒でロード）
    text_dataset = load_formatted_dataset(
        args.data_csv, args.extra_csv, args.subsample, tokenizer,
    )

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
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        max_length=args.max_seq_len,
        remove_unused_columns=False,
        torch_compile=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=text_dataset,
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

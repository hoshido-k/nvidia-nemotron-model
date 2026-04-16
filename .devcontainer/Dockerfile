# Kaggle公式CPUイメージ（GPU不要のローカル開発用）
# GPU依存パッケージ（mamba-ssm, bitsandbytes）は含まれないが
# Python/PyTorch/transformers等のバージョンはKaggleと一致する
FROM gcr.io/kaggle-images/python:latest

# trl / peft はKaggleに入っていないため追加
# （Kaggle上では UTILITY SCRIPTS 経由で提供されるバージョンとは異なる場合あり）
RUN pip install --no-cache-dir \
    trl==1.1.0 \
    peft>=0.12.0 \
    accelerate>=0.34.0

WORKDIR /workspace

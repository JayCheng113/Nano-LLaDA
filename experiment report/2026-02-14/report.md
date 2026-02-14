# Experiment Report

## Date
- 2026-02-14

## Hardware
- NVIDIA H100

## Environment Setup
```bash
pip install uv
uv sync
pip install modelscope
```

## Dataset Preparation
```bash
mkdir -p dataset && modelscope download --dataset gongjy/minimind_dataset pretrain_hq.jsonl --local_dir ./dataset
mkdir -p dataset && modelscope download --dataset gongjy/minimind_dataset sft_mini_512.jsonl --local_dir ./dataset
mkdir -p dataset && modelscope download --dataset gongjy/minimind_dataset dpo.jsonl --local_dir ./dataset
```

```bash
uv run python -m scripts.train.train_dpo_diffusion \
  --data dataset/dpo.jsonl \
  --tokenizer-dir . \
  --load-from weights/diffusion_sft.pt \
  --run-name diffusion_dpo \
  --max-seq-len 512 \
  --batch-size 64 \
  --epochs 1 \
  --dpo-beta 0.1 \
  --llada2-block-size 32 \
  --llada2-alpha-min 0.05 \
  --llada2-alpha-max 0.95
```

uv run python -m scripts.eval.eval_sft_one_prompt   --prompt "你好，请介绍你自己。"   --tokenizer-dir .   --diffusion-checkpoint weights/diffusion_dpo_state_dict.pt   --seq-len 512   --max-new-tokens 128   --gen-steps 64   --gen-cfg-scale 2
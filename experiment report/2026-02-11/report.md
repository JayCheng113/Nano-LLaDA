# Experiment Report

## Date
- 2026-02-11

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
```

## AR Pretraining (MiniMind)
```bash
uv run python -m scripts.train.train_pretrain \
  --data ./dataset/pretrain_hq.jsonl \
  --jsonl-field text \
  --tokenizer-dir . \
  --run-name minimind_pretrain \
  --hidden-size 512 \
  --num-hidden-layers 8 \
  --num-attention-heads 8 \
  --max-seq-len 256 \
  --epochs 1 \
  --batch-size 256
```

## AR Evaluation
```bash
uv run python -m scripts.eval.eval_minimind \
  --checkpoint weights/minimind_pretrain_state_dict.pt \
  --tokenizer-dir . \
  --prompt "请介绍你自己。" \
  --max-new-tokens 200
```

## Diffusion Training (Initialized from AR)
```bash
uv run python -m scripts.train.diffusion \
  --train \
  --use-tokenizer \
  --data ./dataset/pretrain_hq.jsonl \
  --inference-rope-scaling \
  --learning-rate 4e-4 \
  --warmup-steps 5000 \
  --min-lr-ratio 0.025 \
  --mask-schedule iid_t \
  --repeat-penalty-weight 0 \
  --init-from-minimind weights/minimind_pretrain_state_dict.pt \
  --run-name diffusion_from_ar_eq3 \
  --early-stop-patience 5 \
  --early-stop-min-delta 0.001 \
  --max-iters 50000 \
  --batch-size 96
```

### 4. Nano-LLaDA Evaluation

```bash
uv run python -m scripts.eval.eval_diffusion \
  --checkpoint weights/diffusion_from_ar_eq3_best.pt \
  --tokenizer-dir . \
  --seq-len 256 \
  --prompt "请介绍你自己。" \
  --max-new-tokens 200 \
  --gen-steps 64 \
  --gen-cfg-scale 1.5
```

### 5. AR SFT

```bash
uv run python -m scripts.train.train_sft_minimind \
  --data dataset/sft_mini_512.jsonl \
  --tokenizer-dir . \
  --load-from weights/minimind_pretrain_state_dict.pt \
  --run-name minimind_sft \
  --max-seq-len 512 \
  --batch-size 96 \
  --epochs 2
```

### 6. Nano-LLaDA SFT

```bash
uv run python -m scripts.train.train_sft_diffusion \
  --data dataset/sft_mini_512.jsonl \
  --tokenizer-dir . \
  --load-from weights/diffusion_from_ar_eq3_best.pt \
  --run-name diffusion_sft \
  --max-seq-len 512 \
  --batch-size 96 \
  --epochs 2
```

## Output
minimind_pretrain_state_dict.pt
minimind_pretrain.pt
diffusion_from_ar_eq3_best.pt
diffusion_from_ar_eq3_loss.png
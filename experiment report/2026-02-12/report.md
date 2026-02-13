# Experiment Report

## Date
- 2026-02-12

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
  --epochs 2 \
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

## LLaDA2.0-style Diffusion Pretraining (WSD + block curriculum + 2.0 merge/decode switches)
```bash
uv run python -m scripts.train.diffusion \
  --train \
  --use-tokenizer \
  --data ./dataset/pretrain_hq.jsonl \
  --jsonl-field text \
  --seq-len 256 \
  --batch-size 96 \
  --max-iters 50000 \
  --learning-rate 4e-4 \
  --warmup-steps 2000 \
  --min-lr-ratio 0.025 \
  --mask-schedule wsd \
  --use-block-curriculum \
  --wsd-phase-ratios 0.3,0.5,0.2 \
  --wsd-block-sizes-up 32,64,128,256 \
  --wsd-block-sizes-down 256,128,64,32 \
  --time-weighted-loss \
  --llada2-topk-merge \
  --llada2-topk-merge-k 3 \
  --llada2-enable-parallel-decoding \
  --init-from-minimind weights/minimind_pretrain_state_dict.pt \
  --run-name diffusion_llada2_pretrain
```

## LLaDA2.0 style eval on pretrain
```bash
uv run python -m scripts.eval.eval_diffusion \
  --checkpoint weights/diffusion_llada2_pretrain.pt \
  --tokenizer-dir . \
  --seq-len 256 \
  --prompt "请介绍一下你自己。" \
  --max-new-tokens 200 \
  --gen-steps 64 \
  --gen-cfg-scale 1.5 \
  --llada2-enable-parallel-decoding
```

## LLaDA1.0 Evaluation

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

## AR SFT

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

## LLaDA2.0-style Diffusion SFT (Eq.5 block diffusion + CAP + complementary + quantized effective length)
```bash
uv run python -m scripts.train.train_sft_diffusion \
  --data ./dataset/sft_mini_512.jsonl \
  --tokenizer-dir . \
  --load-from weights/diffusion_llada2_pretrain_best.pt \
  --run-name diffusion_llada2_sft \
  --max-seq-len 512 \
  --batch-size 96 \
  --epochs 3 \
  --learning-rate 2.5e-5 \
  --llada2-enable-block-diffusion \
  --llada2-block-size 32 \
  --llada2-alpha-min 0.05 \
  --llada2-alpha-max 0.95 \
  --llada2-enable-cap \
  --llada2-cap-lambda 0.1 \
  --llada2-enable-complementary-masking \
  --llada2-quantize-effective-length
```

## Evaluate with LLaDA2.0 parallel decoding
```bash
uv run python -m scripts.eval.eval_diffusion \
  --checkpoint weights/diffusion_llada2_sft_state_dict.pt \
  --tokenizer-dir . \
  --seq-len 512 \
  --prompt "你好，请介绍你自己。" \
  --max-new-tokens 128 \
  --gen-steps 64 \
  --gen-cfg-scale 1.5 \
  --llada2-enable-parallel-decoding \
  --gen-confidence-threshold 0.9 \
  --gen-top-k 8 \
  --gen-cap-start-ratio 0.08 \
  --gen-cap-end-ratio 0.5 \
  --gen-max-decode-per-step 32
```

# pretrain checkpoint 对比
uv run python -m scripts.eval.compare_llada_styles \
  --checkpoint weights/diffusion_pretrain_state_dict.pt \
  --tokenizer-dir . \
  --task pretrain \
  --prompt "你是谁？" \
  --output outputs/compare_pretrain.jsonl


# sft checkpoint 对比
uv run python -m scripts.eval.compare_llada_styles \
  --checkpoint weights/diffusion_sft_state_dict.pt \
  --tokenizer-dir . \
  --task sft \
  --prompt "你是谁？" \
  --output outputs/compare_sft.jsonl

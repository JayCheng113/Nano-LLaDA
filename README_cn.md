# nano-llada

`nano-llada` 是一个轻量级离散扩散语言模型实验项目。当前版本为 **v0.1.0**，参数规模约 **30M**，在参考开源 `minimind` 与 `tiny-diffusion` 的基础上，实现了 AR 与 Diffusion 双路线的训练、SFT 与评测流程。

## 项目状态

- 当前版本：`v0.1.0`
- 参数规模：`~30M`
- 当前能力：完成 LLaDA 1.0 思路的工程化初步复现
- 进展：经过 SFT 后，`nano-llada` 已初步具备回答问题的能力；当前仍在继续训练与优化，以获得更好的效果。
- 说明：当前实现尚未对 LLaDA 1.0 原文报告中的技术细节做到完全 1:1 复现，相关模块仍在持续修改与迭代中。
- 训练路线：
  - AR（MiniMind-style Causal LM）
  - Diffusion（LLaDA-style masked denoising）

## 总体思路

借用 `minimind` 的模型搭建、数据集与 tokenizer 作为基础配置，先完成一个自回归模型的预训练；随后参考 `LLaDA 2.0` 的训练技巧，搭建一个参数规模与 AR 模型一致的 LLaDA 模型，并加载预训练得到的 AR 权重进行初始化，最后开展后续 SFT 训练与评测。

## 共用架构图（AR + LLaDA）

```mermaid
flowchart TD
    A[Tokenizer Input IDs] --> B[Token Embedding]
    B --> C[N x Transformer Block]

    subgraph C [Shared Transformer Backbone]
      C1[RMSNorm]
      C2[Multi Head Attention with RoPE]
      C3[Residual Add]
      C4[RMSNorm]
      C5[SwiGLU MLP gate up down projections]
      C6[Residual Add]
      C1 --> C2 --> C3 --> C4 --> C5 --> C6
    end

    C --> D[Final RMSNorm]
    D --> E[LM Head Linear to Vocab Logits]

    E --> F1[AR Branch MiniMind]
    E --> F2[LLaDA Branch Diffusion]

    F1 --> G1[Causal Attention and Next Token CE Loss]
    F1 --> H1[Autoregressive Decoding]

    F2 --> G2[Non Causal Masked Denoising Loss]
    F2 --> H2[Iterative Unmask Decoding confidence topk cap]
```

## 环境准备

```bash
pip install uv
uv sync
```

## 数据准备

```bash
pip install modelscope
mkdir -p dataset && modelscope download --dataset gongjy/minimind_dataset pretrain_hq.jsonl --local_dir ./dataset
mkdir -p dataset && modelscope download --dataset gongjy/minimind_dataset sft_mini_512.jsonl --local_dir ./dataset
```

## 训练与评测流程

### 1. AR 预训练

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
  --batch-size 96
```

### 2. AR 评测

```bash
uv run python -m scripts.eval.eval_minimind \
  --checkpoint weights/minimind_pretrain_state_dict.pt \
  --tokenizer-dir . \
  --prompt "请介绍你自己。" \
  --max-new-tokens 200
```

### 3. nano-llada 预训练

```bash
uv run python -m scripts.train.diffusion \
  --train \
  --use-tokenizer \
  --data ./dataset/pretrain_hq.jsonl \
  --inference-rope-scaling \
  --learning-rate 4e-4 \
  --warmup-steps 2000 \
  --min-lr-ratio 0.025 \
  --mask-schedule iid_t \
  --repeat-penalty-weight 0 \
  --init-from-minimind weights/minimind_pretrain_state_dict.pt \
  --run-name diffusion_from_ar_eq3 \
  --early-stop-patience 5 \
  --early-stop-min-delta 0.001 \
  --max-iters 40000 \
  --batch-size 96
```

### 4. nano-llada 评测

```bash
uv run python -m scripts.eval.eval_diffusion \
  --checkpoint weights/diffusion_no_v1.pt \
  --tokenizer-dir . \
  --seq-len 256 \
  --prompt "请介绍你自己。" \
  --max-new-tokens 200
```

## nano-llada 预训练 Loss（中文数据集）

以下为 `nano-llada` 在中文数据集上进行 Diffusion 预训练的 loss 曲线结果：

- 训练 `25k` steps：

![nano-llada diffusion pretrain loss on Chinese dataset (25k steps)](./diffusion_from_ar_eq3_loss_cn_25k.png)

- 训练 `40k` steps：

![nano-llada diffusion pretrain loss on Chinese dataset (40k steps)](./diffusion_from_ar_eq3_loss_cn_40k.png)

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

### 6. nano-llada SFT

```bash
uv run python -m scripts.train.train_sft_diffusion \
  --data dataset/sft_mini_512.jsonl \
  --tokenizer-dir . \
  --load-from weights/diffusion_from_ar_eq3_3g_en.pt \
  --run-name diffusion_sft \
  --max-seq-len 512 \
  --batch-size 96 \
  --epochs 3
```

### 7. SFT 单样本评测

AR SFT:
```bash
uv run python -m scripts.eval.eval_sft_one_prompt \
  --prompt "你好，请介绍你自己。" \
  --tokenizer-dir . \
  --minimind-checkpoint weights/minimind_sft_state_dict.pt \
  --seq-len 512 \
  --max-new-tokens 128
```

Nano-llada SFT:
```bash
uv run python -m scripts.eval.eval_sft_one_prompt \
  --prompt "你好，请介绍你自己。" \
  --tokenizer-dir . \
  --diffusion-checkpoint weights/diffusion_sft_state_dict.pt \
  --seq-len 512 \
  --max-new-tokens 128
```

AR + Nano-llada 对比:
```bash
uv run python -m scripts.eval.eval_sft_one_prompt \
  --prompt "你好，请介绍你自己。" \
  --tokenizer-dir . \
  --minimind-checkpoint weights/minimind_sft_state_dict.pt \
  --diffusion-checkpoint weights/diffusion_sft_state_dict.pt \
  --seq-len 512 \
  --max-new-tokens 128
```

## 技术报告

项目技术报告见：`technical_report.md`

核心定位：
- `nano-llada (~30M)` 的工程化实现
- 当前仅为 `v0.1.0`
- 在本仓库持续精细化实现和迭代优化

## 路线图

1. 继续完善 `v0.1.x`：训练稳定性、解码策略、评测体系。  
2. 复现 `LLaDA 2.0`。  
3. 复现 `LLaDA 2.1`。  
4. 在英文数据集上进行预训练与 SFT，并建立中英文统一评测。  
5. 争取在可控生成质量与综合效果上达到更优结果。

## References

- minimind: https://github.com/jingyaogong/minimind
- tiny-diffusion: https://github.com/nathan-barry/tiny-diffusion

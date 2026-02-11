# Technical Report: nano-llada (~30M, v0.1.0)

## Abstract
本报告介绍了轻量级语言模型 **nano-llada** 的当前实现状态：在参考开源 **MiniMind**（自回归语言建模）与 **tiny diffusion**（离散扩散式文本建模）实现的基础上，构建统一训练与评测流程，完成了约 **30M 参数**规模的 **v0.1.0** 版本。项目包含预训练、SFT（监督微调）与单提示词对比评测三部分，支持 AR 与 Diffusion 两条路线并行实验。现阶段结果表明：在小参数规模下，AR 路线训练与推理链路更直接稳定；Diffusion 路线在迭代解码可控性上具备潜力，但对掩码策略与解码超参数更敏感。

## 1. Introduction
离散文本生成主流方案长期由自回归（AR）模型主导。LLaDA 提供了另一种路径：通过掩码-恢复的扩散式训练与迭代解码完成文本生成。  
本项目目标是：
1. 在统一 tokenizer、数据与模型规模下对比 AR 与 Diffusion。  
2. 基于开源实现快速搭建可复现实验管线。  
3. 验证 LLaDA 1.0 核心机制在小模型场景中的可行性与限制。

## 2. Project Basis and Scope
- 参考项目：
  - MiniMind（Transformer Causal LM 路线）
  - tiny diffusion（离散扩散文本建模路线）
- 本项目定位：**工程复现与对比实验**，非论文级完整复现。
- 当前版本：**nano-llada v0.1.0（约 30M 参数）**。
- 复现范围：
  - AR 预训练 + SFT + 推理
  - Diffusion 预训练 + SFT + 推理
  - 单提示词质量与速度对比评测

## 3. Method
### 3.1 AR Baseline (MiniMind-style)
- 模型：Decoder-only Transformer，RoPE + RMSNorm，因果注意力。
- 训练目标：next-token prediction（标准交叉熵）。
- 流程：`train_pretrain.py` → `train_sft_minimind.py` → `eval_sft_one_prompt.py`。

### 3.2 Diffusion Model (LLaDA-style Approximation)
- 模型骨干与 AR 结构对齐，便于权重迁移与公平对比。
- 训练目标：对 response token 区域随机掩码，学习条件恢复。
- 关键机制：
  - mask schedule（如 `iid_t` / `wsd`）
  - mask token 注入
  - 多轮迭代解码（confidence threshold + top-k + decode budget）
- 流程：`diffusion.py` / `train_sft_diffusion.py` → `eval_sft_one_prompt.py`。

### 3.3 SFT Alignment
- 数据按对话格式整理为 prompt/response。
- AR 与 Diffusion 共享 SFT 数据，分别使用对应 collate 与 loss 设计。
- 目标：比较两种生成范式在指令场景下的行为差异。

## 4. Implementation
- 主要脚本：
  - `train_pretrain.py`: AR 预训练与 tokenizer 处理
  - `diffusion.py`: Diffusion 预训练主流程
  - `train_sft_minimind.py`: AR SFT
  - `train_sft_diffusion.py`: Diffusion SFT
  - `eval_sft_one_prompt.py`: 单 prompt 双模型对比（质量+速度）
- 工程特性：
  - 支持 checkpoint 恢复与权重匹配加载
  - 支持 CUDA/MPS/CPU 自动选择
  - 支持混合精度、梯度累积、学习率调度

## 5. Experiments
### 5.1 Setup
- 数据：
  - 预训练：`pretrain_hq.jsonl`
  - SFT：`sft_mini_512.jsonl`
- 规模：小模型配置（hidden size/layers/heads 对齐）。
- 评测：
  - 单 prompt 生成样例
  - 推理时延与有效 token/s
  - 训练 loss 曲线观察

### 5.2 Observations (Qualitative)
- AR 输出通常更连续，句式稳定，终止行为更可预期。
- Diffusion 在部分样例中具备“后期修正”能力，但容易受阈值和掩码比例影响。
- 当解码参数不合适时，Diffusion 可能出现内容漂移或重复。

### 5.3 Efficiency Notes
- AR：首 token 延迟低，串行解码路径清晰。
- Diffusion：单轮可并行，但总体需要多轮迭代；速度收益依赖解码策略与实现优化。

## 6. Limitations
1. 当前评测偏“小样本 + 单 prompt”，统计显著性有限。  
2. 与原始 LLaDA 1.0 在数据规模、训练预算、评测协议上仍有差距。  
3. Diffusion 路线的最优超参数区域较窄，迁移性有待验证。

## 7. Conclusion
本项目基于 MiniMind 与 tiny diffusion 实现了 **nano-llada v0.1.0（约 30M）**，并完成 AR 与 Diffusion 双路线训练-微调-推理闭环。当前版本可视为 LLaDA 思路在小模型场景下的工程化起点：AR 在稳健性与易用性上更具优势，Diffusion 在可控迭代生成方面展示出研究价值。后续将继续在本仓库基础上精细化实现与迭代优化，逐步提升效果。

## 8. Future Work
1. 在当前仓库基础上持续精细化实现 `nano-llada`，完善训练稳定性、解码策略和评测流程。  
2. 复现 LLaDA 2.0：对齐其训练目标、掩码策略与解码流程，并完成可复现实验。  
3. 复现 LLaDA 2.1：补齐与 2.0 的关键差异模块，形成版本化对比（v0.1.0/2.0/2.1）。  
4. 在英文数据集上开展预训练与 SFT，并建立中英文统一评测协议，争取达到更好的综合效果。


minimind: https://github.com/jingyaogong/minimind
tiny-diffusion: https://github.com/nathan-barry/tiny-diffusion

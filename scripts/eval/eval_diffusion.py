import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from core.trainer_utils import auto_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate diffusion model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion checkpoint")
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Tokenizer directory")
    parser.add_argument("--seq-len", type=int, default=256)

    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--gen-temp",
        type=float,
        default=0.8,
        help="Legacy parameter; ignored by greedy s/t remask decoding",
    )
    parser.add_argument(
        "--gen-confidence-threshold",
        type=float,
        default=0.9,
        help="Legacy parameter; ignored by s/t remask decoding",
    )
    parser.add_argument(
        "--gen-top-k",
        type=int,
        default=8,
        help="Legacy parameter; ignored by greedy s/t remask decoding",
    )
    parser.add_argument("--gen-repeat-penalty", type=float, default=0.0)
    parser.add_argument("--gen-repeat-window", type=int, default=128)
    parser.add_argument(
        "--gen-cap-start-ratio",
        type=float,
        default=0.08,
        help="Legacy parameter; ignored by s/t remask decoding",
    )
    parser.add_argument(
        "--gen-cap-end-ratio",
        type=float,
        default=0.5,
        help="Legacy parameter; ignored by s/t remask decoding",
    )
    parser.add_argument(
        "--gen-max-decode-per-step",
        type=int,
        default=32,
        help="Legacy parameter; ignored by s/t remask decoding",
    )
    parser.add_argument("--gen-steps", type=int, default=64)
    parser.add_argument(
        "--gen-cfg-scale",
        type=float,
        default=1.5,
        help="Unsupervised CFG strength w in Eq.(16); 0 disables CFG (default enabled)",
    )

    parser.add_argument("--eval-data", type=str, default=None, help="Optional .jsonl/.txt")
    parser.add_argument("--jsonl-field", type=str, default="text")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-batches", type=int, default=50)
    parser.add_argument("--eval-mask-ratio", type=float, default=0.5)
    parser.add_argument(
        "--cond-ll-response",
        type=str,
        default=None,
        help="Optional response text for LLaDA-style conditional log-likelihood Monte Carlo evaluation",
    )
    parser.add_argument(
        "--cond-ll-nmc",
        type=int,
        default=64,
        help="Number of Monte Carlo samples for conditional log-likelihood estimation",
    )

    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def norm(x, eps):
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def get_activation(name):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation: {name}")


def precompute_freqs_cis(dim, end, rope_base, rope_scaling=None, device=None):
    if device is None:
        device = "cpu"

    freqs = 1.0 / (
        rope_base ** (torch.arange(0, dim, 2, device=device).float()[: (dim // 2)] / dim)
    )
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:
            inv_dim = lambda b: (
                dim * math.log(orig_max / (b * 2 * math.pi)) / (2 * math.log(rope_base))
            )
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def rotate_half(x):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x, cos, sin):
    return ((x * cos) + (rotate_half(x) * sin)).to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, rms_norm_eps):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps

    def forward(self, x, cos_sin):
        bsz, seq_len, _ = x.size()
        q = self.c_q(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seq_len, self.n_head, self.head_dim)
        v = self.c_v(x).view(bsz, seq_len, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q, self.rms_norm_eps), norm(k, self.rms_norm_eps)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd, intermediate_size, dropout, hidden_act):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=False)
        self.up_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = get_activation(hidden_act)

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, intermediate_size, dropout, hidden_act, rms_norm_eps):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head, head_dim, rms_norm_eps)
        self.mlp = MLP(n_embd, intermediate_size, dropout, hidden_act)
        self.rms_norm_eps = rms_norm_eps

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x, self.rms_norm_eps), cos_sin)
        x = x + self.mlp(norm(x, self.rms_norm_eps))
        return x


class DiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd,
        n_head,
        n_layer,
        intermediate_size,
        dropout,
        hidden_act,
        rms_norm_eps,
        rope_base,
        max_position_embeddings,
        rope_scaling,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.head_dim = n_embd // n_head
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.rotary_seq_len = max(max_position_embeddings, 512)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    head_dim=self.head_dim,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    hidden_act=hidden_act,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(n_layer)
            ]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def _precompute_rotary_embeddings(self, seq_len, device=None):
        if device is None:
            device = self.token_emb.weight.device
        cos, sin = precompute_freqs_cis(
            dim=self.head_dim,
            end=seq_len,
            rope_base=self.rope_base,
            rope_scaling=self.rope_scaling,
            device=device,
        )
        return cos[None, :, None, :], sin[None, :, None, :]

    def _ensure_rope_length(self, seq_len, device):
        if seq_len <= self.cos.size(1) and self.cos.device == device:
            return
        self.rotary_seq_len = max(seq_len, self.rotary_seq_len)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, device=device)
        self.cos = cos
        self.sin = sin

    def forward(self, idx):
        _, seq_len = idx.size()
        self._ensure_rope_length(seq_len, idx.device)

        x = self.token_emb(idx)
        x = norm(x, self.rms_norm_eps)
        cos_sin = (self.cos[:, :seq_len], self.sin[:, :seq_len])
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x, self.rms_norm_eps)
        logits = self.lm_head(x)
        return logits


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, text_field="text", max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []

        p = Path(path)
        if p.suffix == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    value = row.get(text_field, "")
                    if isinstance(value, str) and value:
                        self.texts.append(value)
                    elif value:
                        self.texts.append(str(value))
        else:
            text = p.read_text(encoding="utf-8")
            self.texts = [line for line in text.splitlines() if line.strip()]
            if not self.texts:
                self.texts = [text]

        if not self.texts:
            raise ValueError(f"No usable samples found in {path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        toks = self.tokenizer(
            self.texts[idx],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        ).input_ids
        toks = [self.tokenizer.bos_token_id] + toks + [self.tokenizer.eos_token_id]
        length = len(toks)
        toks = toks + [self.tokenizer.pad_token_id] * (self.max_length - length)
        return torch.tensor(toks, dtype=torch.long)


def extract_model_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("args", None)
    return ckpt, None


def ensure_nonempty_mask(mask, candidate_mask):
    for b in range(mask.size(0)):
        if candidate_mask[b].any() and not mask[b].any():
            valid_pos = torch.nonzero(candidate_mask[b], as_tuple=False).view(-1)
            chosen = valid_pos[torch.randint(valid_pos.numel(), (1,))]
            mask[b, chosen] = True
    return mask


def random_mask_batch(x, tokenizer, mask_token_id, mask_ratio):
    y = x.clone()
    candidate_mask = (
        (x != tokenizer.pad_token_id)
        & (x != tokenizer.bos_token_id)
        & (x != tokenizer.eos_token_id)
    )
    mask = (torch.rand_like(x.float()) < mask_ratio) & candidate_mask
    mask = ensure_nonempty_mask(mask, candidate_mask)
    x_masked = x.clone()
    x_masked[mask] = mask_token_id
    return x_masked, y, mask


@torch.no_grad()
def quick_eval_masked_loss(model, loader, tokenizer, mask_token_id, device, mask_ratio, max_batches):
    model.eval()
    losses = []
    masked_counts = []

    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        x_masked, y, mask = random_mask_batch(x, tokenizer, mask_token_id, mask_ratio)
        logits = model(x_masked)

        bsz, seq_len, vocab = logits.shape
        ce = F.cross_entropy(logits.view(bsz * seq_len, vocab), y.view(-1), reduction="none")
        mask_flat = mask.view(-1)
        count = int(mask_flat.sum().item())
        if count == 0:
            continue
        loss = (ce * mask_flat).sum() / count
        losses.append(loss.item())
        masked_counts.append(count)

    if not losses:
        return float("nan"), 0
    return sum(losses) / len(losses), sum(masked_counts)


@torch.no_grad()
def conditional_log_likelihood_llada(
    model,
    tokenizer,
    prompt,
    response,
    mask_token_id,
    seq_len,
    nmc,
    device,
):
    if nmc <= 0:
        raise ValueError("cond-ll-nmc must be > 0")

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    # Keep response as much as possible while respecting seq_len.
    max_resp_len = max(seq_len - len(prompt_ids), 1)
    response_ids = response_ids[:max_resp_len]
    L = len(response_ids)
    if L == 0:
        return float("nan"), 0

    log_likelihood = 0.0
    for _ in range(nmc):
        l = int(torch.randint(1, L + 1, (1,)).item())
        masked_positions = torch.randperm(L)[:l]

        rt = response_ids.copy()
        for pos in masked_positions.tolist():
            rt[pos] = mask_token_id

        full_ids = prompt_ids + rt
        full_ids = full_ids[:seq_len]
        input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(input_ids)
        log_probs = F.log_softmax(logits[0], dim=-1)

        # Map response indices to full sequence indices.
        offset = len(prompt_ids)
        sum_logp = 0.0
        valid_count = 0
        for pos in masked_positions.tolist():
            full_pos = offset + pos
            if full_pos >= input_ids.size(1):
                continue
            target_token = response_ids[pos]
            sum_logp += float(log_probs[full_pos, target_token].item())
            valid_count += 1

        if valid_count > 0:
            log_likelihood += (L / valid_count) * sum_logp

    return log_likelihood / nmc, L


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt_tokens,
    mask_token_id,
    device,
    block_size,
    max_new_tokens,
    temp,
    confidence_threshold,
    top_k,
    repeat_penalty,
    repeat_window,
    cap_start_ratio,
    cap_end_ratio,
    max_decode_per_step,
    gen_steps,
    cfg_scale,
):
    _ = (temp, confidence_threshold, top_k, cap_start_ratio, cap_end_ratio, max_decode_per_step)
    effective_prompt_len = min(len(prompt_tokens), max(block_size - 1, 1))
    all_tokens = prompt_tokens[:effective_prompt_len]

    while len(all_tokens) - effective_prompt_len < max_new_tokens:
        block_len = min(block_size - effective_prompt_len, max_new_tokens - (len(all_tokens) - effective_prompt_len))
        if block_len <= 0:
            break
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :effective_prompt_len] = torch.tensor(all_tokens[-effective_prompt_len:], device=device)

        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, effective_prompt_len : effective_prompt_len + block_len] = True
        for step_idx in range(gen_steps):
            if not masked.any():
                break
            logits_cond = model(x)
            logits_cond[..., mask_token_id] = -float("inf")
            if cfg_scale > 0 and effective_prompt_len > 0:
                x_uncond = x.clone()
                x_uncond[:, :effective_prompt_len] = mask_token_id
                logits_uncond = model(x_uncond)
                logits_uncond[..., mask_token_id] = -float("inf")
                cond_log_probs = F.log_softmax(logits_cond, dim=-1)
                uncond_log_probs = F.log_softmax(logits_uncond, dim=-1)
                logits = (1.0 + cfg_scale) * cond_log_probs - cfg_scale * uncond_log_probs
            else:
                logits = logits_cond

            if repeat_penalty > 0:
                finalized = x[0][x[0] != mask_token_id]
                if repeat_window > 0:
                    finalized = finalized[-repeat_window:]
                if finalized.numel() > 0:
                    counts = torch.bincount(finalized, minlength=model.vocab_size).to(logits.dtype)
                    logits = logits - repeat_penalty * counts.view(1, 1, -1)

            probs = F.softmax(logits, dim=-1)
            sampled_tokens = torch.argmax(logits, dim=-1)
            sampled_probs = torch.gather(probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

            x_filled = torch.where(masked, sampled_tokens, x)
            conf = torch.where(masked, sampled_probs, torch.tensor(float("inf"), device=device))

            if step_idx == gen_steps - 1:
                x = x_filled
                masked = torch.zeros_like(masked)
                break

            t = 1.0 - (step_idx / gen_steps)
            s = 1.0 - ((step_idx + 1) / gen_steps)
            remask_ratio = s / max(t, 1e-8)

            current_masked_idx = torch.nonzero(masked[0], as_tuple=False).view(-1)
            n_masked = int(current_masked_idx.numel())
            if n_masked <= 0:
                x = x_filled
                masked = torch.zeros_like(masked)
                break
            n_remask = int(round(n_masked * remask_ratio))
            n_remask = min(max(n_remask, 0), n_masked)
            if n_remask == 0:
                x = x_filled
                masked = torch.zeros_like(masked)
                break

            conf_masked = conf[0, current_masked_idx]
            remask_local = torch.topk(conf_masked, k=n_remask, largest=False).indices
            remask_positions = current_masked_idx[remask_local]
            next_masked = torch.zeros_like(masked)
            next_masked[0, remask_positions] = True

            x = x_filled
            x[next_masked] = mask_token_id
            masked = next_masked

        all_tokens.extend(x[0, effective_prompt_len : effective_prompt_len + block_len].tolist())

    return tokenizer.decode(all_tokens, skip_special_tokens=False)


def infer_hparams_from_state_dict(state_dict):
    emb_shape = state_dict["token_emb.weight"].shape
    vocab_size, n_embd = emb_shape

    layer_indices = []
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.append(int(parts[1]))
    n_layer = (max(layer_indices) + 1) if layer_indices else 8

    head_dim = 64
    for cand in [16, 12, 10, 8, 6, 4, 2, 1, 32, 24]:
        if n_embd % cand == 0:
            if n_embd // cand in [32, 48, 64, 80, 96, 128]:
                head_dim = n_embd // cand
                n_head = cand
                break
    else:
        n_head = 8 if n_embd % 8 == 0 else 1
        head_dim = n_embd // n_head

    key = "blocks.0.mlp.up_proj.weight"
    intermediate_size = state_dict[key].shape[0] if key in state_dict else int(n_embd * 8 / 3)

    return {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
    }


def main():
    args = parse_args()
    if args.gen_steps <= 0:
        raise ValueError("--gen-steps must be > 0")
    if args.gen_cfg_scale < 0:
        raise ValueError("--gen-cfg-scale must be >= 0")
    device = auto_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_raw = torch.load(args.checkpoint, map_location="cpu")
    state_dict, ckpt_args = extract_model_state_dict(ckpt_raw)
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format")
    ckpt_vocab_size = state_dict["token_emb.weight"].shape[0]

    mask_token = "<|mask|>"
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
        print("Tokenizer missing <|mask|>; added it automatically for eval.")
    if len(tokenizer) < ckpt_vocab_size:
        extra_count = ckpt_vocab_size - len(tokenizer)
        tokenizer.add_tokens([f"<|extra_eval_{i}|>" for i in range(extra_count)])
        print(f"Tokenizer expanded by {extra_count} tokens to match checkpoint vocab={ckpt_vocab_size}.")
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    if mask_token_id >= ckpt_vocab_size:
        raise ValueError(
            f"mask_token_id ({mask_token_id}) >= checkpoint vocab size ({ckpt_vocab_size}). "
            "Please evaluate with the same tokenizer used in diffusion training."
        )

    h = infer_hparams_from_state_dict(state_dict)
    hidden_act = "silu"
    dropout = 0.0
    rms_norm_eps = 1e-5
    rope_base = 1e6
    max_position_embeddings = 32768
    rope_scaling = None

    if isinstance(ckpt_args, dict):
        hidden_act = ckpt_args.get("hidden_act", hidden_act)
        dropout = ckpt_args.get("dropout", dropout)
        rms_norm_eps = ckpt_args.get("rms_norm_eps", rms_norm_eps)
        rope_base = ckpt_args.get("rope_theta", rope_base)
        max_position_embeddings = ckpt_args.get("max_position_embeddings", max_position_embeddings)
        if ckpt_args.get("inference_rope_scaling", False):
            rope_scaling = {
                "original_max_position_embeddings": 2048,
                "factor": 16,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.0,
                "type": "yarn",
            }

    model = DiffusionModel(
        vocab_size=h["vocab_size"],
        n_embd=h["n_embd"],
        n_head=h["n_head"],
        n_layer=h["n_layer"],
        intermediate_size=h["intermediate_size"],
        dropout=dropout,
        hidden_act=hidden_act,
        rms_norm_eps=rms_norm_eps,
        rope_base=rope_base,
        max_position_embeddings=max_position_embeddings,
        rope_scaling=rope_scaling,
    ).to(device)

    model_state = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(matched, strict=False)

    print("=== Diffusion Checkpoint Report ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"matched_tensors: {len(matched)}")
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")
    print(f"hparams: hidden={h['n_embd']}, layers={h['n_layer']}, heads={h['n_head']}, vocab={h['vocab_size']}")

    if len(matched) == 0:
        raise ValueError("No tensors matched. Check checkpoint and tokenizer.")

    prompt_ids = tokenizer(args.prompt, add_special_tokens=False).input_ids
    if tokenizer.bos_token_id is not None:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids

    out_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_ids,
        mask_token_id=mask_token_id,
        device=device,
        block_size=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        temp=args.gen_temp,
        confidence_threshold=args.gen_confidence_threshold,
        top_k=args.gen_top_k,
        repeat_penalty=args.gen_repeat_penalty,
        repeat_window=args.gen_repeat_window,
        cap_start_ratio=args.gen_cap_start_ratio,
        cap_end_ratio=args.gen_cap_end_ratio,
        max_decode_per_step=args.gen_max_decode_per_step,
        gen_steps=args.gen_steps,
        cfg_scale=args.gen_cfg_scale,
    )
    print("\n=== Generation Sample ===")
    print(out_text)

    if args.eval_data:
        ds = TextDataset(args.eval_data, tokenizer, text_field=args.jsonl_field, max_length=args.seq_len)
        loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
        avg_masked_loss, masked_tokens = quick_eval_masked_loss(
            model=model,
            loader=loader,
            tokenizer=tokenizer,
            mask_token_id=mask_token_id,
            device=device,
            mask_ratio=args.eval_mask_ratio,
            max_batches=args.eval_max_batches,
        )
        print("\n=== Quick Eval (masked reconstruction) ===")
        print(f"eval_data: {args.eval_data}")
        print(f"samples: {len(ds)}, masked_tokens_seen: {masked_tokens}")
        print(f"avg_masked_loss: {avg_masked_loss:.4f}")

    if args.cond_ll_response is not None:
        cond_ll, resp_len = conditional_log_likelihood_llada(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            response=args.cond_ll_response,
            mask_token_id=mask_token_id,
            seq_len=args.seq_len,
            nmc=args.cond_ll_nmc,
            device=device,
        )
        print("\n=== Conditional Log-likelihood (LLaDA Algorithm 3) ===")
        print(f"prompt: {args.prompt}")
        print(f"response_len: {resp_len}, nmc: {args.cond_ll_nmc}")
        print(f"log_likelihood: {cond_ll:.4f}")


if __name__ == "__main__":
    main()

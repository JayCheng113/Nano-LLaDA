import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lm_dataset import PretrainDataset
from model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer_utils import auto_device, extract_state_dict, normalize_state_dict_keys


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind pretrained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to minimind checkpoint (.pt/.pth)")
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Tokenizer directory")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。", help="Prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu/mps; default auto")

    parser.add_argument("--eval-data", type=str, default=None, help="Optional .jsonl/.txt for quick perplexity check")
    parser.add_argument("--jsonl-field", type=str, default="text")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-batches", type=int, default=50)
    return parser.parse_args()


def infer_config_from_state_dict(state_dict, tokenizer, fallback_seq_len):
    token_emb_key = "token_emb.weight"
    if token_emb_key not in state_dict:
        raise ValueError(f"State dict missing {token_emb_key}, cannot infer model config")

    vocab_size, hidden_size = state_dict[token_emb_key].shape

    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    if not layer_indices:
        raise ValueError("State dict has no blocks.* keys, cannot infer num_hidden_layers")
    num_hidden_layers = max(layer_indices) + 1

    q_key = "blocks.0.attn.c_q.weight"
    if q_key not in state_dict:
        raise ValueError(f"State dict missing {q_key}, cannot infer num_attention_heads")
    q_out, _ = state_dict[q_key].shape

    head_candidates = [h for h in (4, 8, 12, 16, 24, 32, 40, 48, 64) if hidden_size % h == 0 and q_out == hidden_size]
    num_attention_heads = 8 if 8 in head_candidates else (head_candidates[0] if head_candidates else 1)

    up_key = "blocks.0.mlp.up_proj.weight"
    intermediate_size = state_dict[up_key].shape[0] if up_key in state_dict else None

    return MiniMindConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max(32768, fallback_seq_len),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attention_is_causal=True,
    )


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=128, temperature=0.8, top_k=20):
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(device)

    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([bos, input_ids], dim=1)

    for _ in range(max_new_tokens):
        logits = model(input_ids=input_ids).logits[:, -1, :]

        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                vals, idx = torch.topk(logits, k=k, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                pick = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(idx, 1, pick)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=False)


@torch.no_grad()
def quick_perplexity(model, tokenizer, data_path, text_field, max_seq_len, batch_size, max_batches, device):
    dataset = PretrainDataset(data_path, tokenizer, text_field=text_field, max_length=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    losses = []
    tokens = 0

    for i, (input_ids, labels, attention_mask) in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss.detach().float().item()
        valid_tokens = int((labels != -100).sum().item())

        losses.append(loss)
        tokens += valid_tokens

    if not losses:
        return float("nan"), float("nan"), 0, len(dataset)

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl, tokens, len(dataset)


def main():
    args = parse_args()
    device = auto_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(str(ckpt_path), map_location="cpu")
    state_dict, meta = extract_state_dict(raw)
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format")
    state_dict = normalize_state_dict_keys(state_dict)

    config = infer_config_from_state_dict(state_dict, tokenizer, args.max_seq_len)
    model = MiniMindForCausalLM(config).to(device)

    model_state = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(matched, strict=False)

    print("=== MiniMind Checkpoint Report ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"device: {device}")
    print(f"matched_tensors: {len(matched)}")
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")
    print(
        f"config: vocab={config.vocab_size}, hidden={config.hidden_size}, "
        f"layers={config.num_hidden_layers}, heads={config.num_attention_heads}"
    )

    if len(matched) == 0:
        raise ValueError("No tensors matched when loading checkpoint. Please check architecture hyperparameters.")

    generated = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\n=== Generation Sample ===")
    print(generated)

    if args.eval_data:
        avg_loss, ppl, seen_tokens, total_samples = quick_perplexity(
            model=model,
            tokenizer=tokenizer,
            data_path=args.eval_data,
            text_field=args.jsonl_field,
            max_seq_len=args.max_seq_len,
            batch_size=args.eval_batch_size,
            max_batches=args.eval_max_batches,
            device=device,
        )
        print("\n=== Quick Eval (subset) ===")
        print(f"eval_data: {args.eval_data}")
        print(f"samples_total: {total_samples}, tokens_seen: {seen_tokens}")
        print(f"avg_loss: {avg_loss:.4f}, ppl: {ppl:.4f}")


if __name__ == "__main__":
    main()

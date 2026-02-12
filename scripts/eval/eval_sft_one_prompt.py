import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from scripts.eval.eval_diffusion import (
    DiffusionModel,
    extract_model_state_dict,
    generate as diffusion_generate,
    infer_hparams_from_state_dict,
)
from scripts.eval.eval_minimind import (
    infer_config_from_state_dict,
)
from core.model_minimind import MiniMindForCausalLM
from core.trainer_utils import auto_device, extract_state_dict as extract_minimind_state_dict, normalize_state_dict_keys


def parse_args():
    parser = argparse.ArgumentParser(description="Quick one-prompt SFT comparison (MiniMind vs Diffusion)")
    parser.add_argument("--prompt", type=str, required=True, help="Single prompt to test")
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Tokenizer directory")

    parser.add_argument("--minimind-checkpoint", type=str, default=None, help="Path to AR SFT checkpoint")
    parser.add_argument("--diffusion-checkpoint", type=str, default=None, help="Path to diffusion SFT checkpoint")

    parser.add_argument("--device", type=str, default=None, help="cuda/cpu/mps; default auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256, help="Diffusion block size")

    parser.add_argument("--ar-temperature", type=float, default=0.8)
    parser.add_argument("--ar-top-k", type=int, default=20)

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
    parser.add_argument("--mask-token", type=str, default="<|mask|>")
    parser.add_argument(
        "--no-chat-wrap",
        action="store_true",
        help="Disable automatic chat prompt wrapping",
    )
    return parser.parse_args()


def maybe_wrap_chat_prompt(prompt: str, no_chat_wrap: bool) -> str:
    if no_chat_wrap:
        return prompt
    if "<|im_start|>" in prompt:
        return prompt
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def _sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def maybe_prepend_bos_token_ids(token_ids, bos_token_id):
    if bos_token_id is None:
        return token_ids
    if token_ids and token_ids[0] == bos_token_id:
        return token_ids
    return [bos_token_id] + token_ids


def count_effective_text_tokens(token_ids, special_ids):
    return sum(1 for tid in token_ids if tid not in special_ids)


@torch.no_grad()
def measure_ar_first_token_latency(model, tokenizer, prompt, device, temperature, top_k):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    prompt_ids = maybe_prepend_bos_token_ids(prompt_ids, tokenizer.bos_token_id)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    _sync_if_cuda(device)
    start = time.perf_counter()
    logits = model(input_ids=input_ids).logits[:, -1, :]
    if temperature <= 0:
        _ = logits.argmax(dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k=k, dim=-1)
            probs = torch.softmax(vals, dim=-1)
            pick = torch.multinomial(probs, num_samples=1)
            _ = torch.gather(idx, 1, pick)
        else:
            probs = torch.softmax(logits, dim=-1)
            _ = torch.multinomial(probs, num_samples=1)
    _sync_if_cuda(device)
    return time.perf_counter() - start


@torch.no_grad()
def ar_generate_text(model, tokenizer, prompt, device, max_new_tokens, temperature, top_k):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    prompt_ids = maybe_prepend_bos_token_ids(prompt_ids, tokenizer.bos_token_id)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

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
def measure_diffusion_first_round_latency(
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
    effective_prompt_len = min(len(prompt_tokens), min(32, block_size // 2))
    _sync_if_cuda(device)
    start = time.perf_counter()
    block_len = min(block_size - effective_prompt_len, max_new_tokens)
    x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
    x[0, :effective_prompt_len] = torch.tensor(prompt_tokens[-effective_prompt_len:], device=device)
    masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
    masked[0, effective_prompt_len : effective_prompt_len + block_len] = True
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
    _ = torch.argmax(logits, dim=-1)
    _ = gen_steps

    _sync_if_cuda(device)
    return time.perf_counter() - start


def print_speed_stats(name, total_time, generated_tokens, effective_tokens, first_latency):
    tps = (effective_tokens / total_time) if total_time > 0 else float("nan")
    print(f"\n=== {name} Speed ===")
    print(f"total_time_s: {total_time:.4f}")
    print(f"new_tokens_raw: {generated_tokens}")
    print(f"new_tokens_effective_text: {effective_tokens}")
    print(f"tokens_per_s_effective: {tps:.2f}")
    if first_latency is not None:
        print(f"first_latency_s: {first_latency:.4f}")


def load_minimind_model(checkpoint_path, tokenizer, max_seq_len, device):
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict, _ = extract_minimind_state_dict(raw)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported minimind checkpoint format: {checkpoint_path}")
    state_dict = normalize_state_dict_keys(state_dict)

    config = infer_config_from_state_dict(state_dict, tokenizer, max_seq_len)
    model = MiniMindForCausalLM(config).to(device)
    model_state = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(matched, strict=False)

    print("=== AR Load Report ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"matched_tensors: {len(matched)}")
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")
    if len(matched) == 0:
        raise ValueError("No tensors matched for MiniMind checkpoint.")
    model.eval()
    return model


def load_diffusion_model(checkpoint_path, tokenizer, mask_token, device):
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict, ckpt_args = extract_model_state_dict(raw)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported diffusion checkpoint format: {checkpoint_path}")

    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    ckpt_vocab_size = state_dict["token_emb.weight"].shape[0]
    if len(tokenizer) < ckpt_vocab_size:
        tokenizer.add_tokens([f"<|extra_eval_{i}|>" for i in range(ckpt_vocab_size - len(tokenizer))])
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    if mask_token_id >= ckpt_vocab_size:
        raise ValueError(
            f"mask_token_id ({mask_token_id}) >= checkpoint vocab size ({ckpt_vocab_size}). "
            "Please use the tokenizer used during diffusion training."
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

    print("=== Diffusion Load Report ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"matched_tensors: {len(matched)}")
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")
    if len(matched) == 0:
        raise ValueError("No tensors matched for diffusion checkpoint.")
    model.eval()
    return model, mask_token_id


def main():
    args = parse_args()
    if args.gen_steps <= 0:
        raise ValueError("--gen-steps must be > 0")
    if args.gen_cfg_scale < 0:
        raise ValueError("--gen-cfg-scale must be >= 0")
    if not args.minimind_checkpoint and not args.diffusion_checkpoint:
        raise ValueError("Please provide at least one checkpoint: --minimind-checkpoint and/or --diffusion-checkpoint.")

    device = auto_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_ids = set(tokenizer.all_special_ids or [])

    wrapped_prompt = maybe_wrap_chat_prompt(args.prompt, args.no_chat_wrap)
    print(f"device: {device}")
    print(f"prompt: {args.prompt}")
    if wrapped_prompt != args.prompt:
        print("chat_wrap: enabled (auto)")

    if args.minimind_checkpoint:
        ar_model = load_minimind_model(args.minimind_checkpoint, tokenizer, args.seq_len, device)
        ar_first_token_latency = measure_ar_first_token_latency(
            model=ar_model,
            tokenizer=tokenizer,
            prompt=wrapped_prompt,
            device=device,
            temperature=args.ar_temperature,
            top_k=args.ar_top_k,
        )
        prompt_ids_ar = tokenizer(wrapped_prompt, add_special_tokens=False).input_ids
        prompt_ids_ar = maybe_prepend_bos_token_ids(prompt_ids_ar, tokenizer.bos_token_id)
        _sync_if_cuda(device)
        ar_t0 = time.perf_counter()
        ar_text = ar_generate_text(
            model=ar_model,
            tokenizer=tokenizer,
            prompt=wrapped_prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.ar_temperature,
            top_k=args.ar_top_k,
        )
        _sync_if_cuda(device)
        ar_total_time = time.perf_counter() - ar_t0
        ar_out_ids = tokenizer(ar_text, add_special_tokens=False).input_ids
        ar_new_tokens = max(0, len(ar_out_ids) - len(prompt_ids_ar))
        ar_gen_ids = ar_out_ids[-ar_new_tokens:] if ar_new_tokens > 0 else []
        ar_effective_tokens = count_effective_text_tokens(ar_gen_ids, special_ids)
        print("\n=== AR Output ===")
        print(ar_text)
        print_speed_stats("AR", ar_total_time, ar_new_tokens, ar_effective_tokens, ar_first_token_latency)

    if args.diffusion_checkpoint:
        diffusion_model, mask_token_id = load_diffusion_model(
            args.diffusion_checkpoint, tokenizer, args.mask_token, device
        )
        prompt_ids = tokenizer(wrapped_prompt, add_special_tokens=False).input_ids
        prompt_ids = maybe_prepend_bos_token_ids(prompt_ids, tokenizer.bos_token_id)
        diff_first_round_latency = measure_diffusion_first_round_latency(
            model=diffusion_model,
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
        _sync_if_cuda(device)
        diff_t0 = time.perf_counter()
        diff_text = diffusion_generate(
            model=diffusion_model,
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
        _sync_if_cuda(device)
        diff_total_time = time.perf_counter() - diff_t0
        diff_out_ids = tokenizer(diff_text, add_special_tokens=False).input_ids
        diff_new_tokens = max(0, len(diff_out_ids) - len(prompt_ids))
        diff_gen_ids = diff_out_ids[-diff_new_tokens:] if diff_new_tokens > 0 else []
        diff_effective_tokens = count_effective_text_tokens(diff_gen_ids, special_ids)
        print("\n=== Diffusion Output ===")
        print(diff_text)
        print_speed_stats("Diffusion", diff_total_time, diff_new_tokens, diff_effective_tokens, diff_first_round_latency)


if __name__ == "__main__":
    main()

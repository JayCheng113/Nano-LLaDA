import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from core.dpo_dataset import PreferenceConversationDataset, collate_preference_diffusion
from core.trainer_utils import auto_device, extract_state_dict, load_matching_weights, normalize_state_dict_keys
from scripts.eval.eval_diffusion import DiffusionModel

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA2.0-style diffusion DPO")
    parser.add_argument("--data", type=str, required=True, help="Preference jsonl path")
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Tokenizer directory")
    parser.add_argument("--save-dir", type=str, default="weights", help="Checkpoint directory")
    parser.add_argument("--run-name", type=str, default="diffusion_dpo", help="Checkpoint prefix")
    parser.add_argument("--load-from", type=str, required=True, help="Policy init checkpoint path (post-SFT)")
    parser.add_argument(
        "--reference-from",
        type=str,
        default=None,
        help="Optional reference checkpoint path (default: same as --load-from)",
    )
    parser.add_argument(
        "--resume-optimizer-state",
        action="store_true",
        help="When loading from full checkpoint, also resume optimizer/scaler/step",
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="DPO lr. Default: infer final SFT lr from --load-from args (learning_rate * final_lr_ratio), fallback 2.5e-6",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--final-decay-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-ratio", type=float, default=0.1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)

    parser.add_argument("--mask-token", type=str, default="<|mask|>")
    parser.add_argument("--iid-mask-eps", type=float, default=1e-3)
    parser.add_argument("--llada2-block-size", type=int, default=32)
    parser.add_argument("--llada2-alpha-min", type=float, default=0.05)
    parser.add_argument("--llada2-alpha-max", type=float, default=0.95)
    parser.add_argument("--llada2-quantize-effective-length", action="store_true")
    parser.add_argument("--dpo-beta", type=float, default=0.1)

    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu/mps; default auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def llada_sft_lr(step, total_steps, base_lr, warmup_steps, final_decay_ratio, final_lr_ratio):
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    decay_steps = max(1, int(total_steps * final_decay_ratio))
    decay_start = max(0, total_steps - decay_steps)
    if step < decay_start:
        return base_lr

    progress = float(step - decay_start) / float(max(total_steps - decay_start, 1))
    progress = min(max(progress, 0.0), 1.0)
    min_lr = base_lr * final_lr_ratio
    return base_lr - (base_lr - min_lr) * progress


def infer_hparams_from_state_dict(state_dict):
    vocab_size, n_embd = state_dict["token_emb.weight"].shape

    layers = []
    for k in state_dict:
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layers.append(int(parts[1]))
    n_layer = (max(layers) + 1) if layers else 8

    q_key = "blocks.0.attn.c_q.weight"
    if q_key not in state_dict:
        n_head = 8 if n_embd % 8 == 0 else 1
    else:
        n_head = 8 if n_embd % 8 == 0 else 1
        for cand in [16, 12, 10, 8, 6, 4, 2, 1, 32, 24]:
            if n_embd % cand == 0 and (n_embd // cand) in [32, 48, 64, 80, 96, 128]:
                n_head = cand
                break

    up_key = "blocks.0.mlp.up_proj.weight"
    intermediate_size = state_dict[up_key].shape[0] if up_key in state_dict else int(n_embd * 8 / 3)

    return {
        "vocab_size": vocab_size,
        "hidden_size": n_embd,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "intermediate_size": intermediate_size,
    }


def _extract_ckpt_args(meta):
    if not isinstance(meta, dict):
        return {}
    args_dict = meta.get("args")
    if isinstance(args_dict, dict):
        return args_dict
    return {}


def resolve_model_config(state_dict, meta):
    inferred = infer_hparams_from_state_dict(state_dict)
    ckpt_args = _extract_ckpt_args(meta)
    def _pick(name, fallback):
        value = ckpt_args.get(name, None)
        return fallback if value is None else value

    model_cfg = {
        "vocab_size": inferred["vocab_size"],
        "n_embd": int(_pick("hidden_size", inferred["hidden_size"])),
        "n_head": int(_pick("num_attention_heads", inferred["num_attention_heads"])),
        "n_layer": int(_pick("num_hidden_layers", inferred["num_hidden_layers"])),
        "intermediate_size": int(_pick("intermediate_size", inferred["intermediate_size"])),
        "dropout": float(_pick("dropout", 0.0)),
        "hidden_act": str(_pick("hidden_act", "silu")),
        "rms_norm_eps": float(_pick("rms_norm_eps", 1e-5)),
        "rope_base": float(_pick("rope_theta", 1e6)),
        "max_position_embeddings": int(_pick("max_position_embeddings", 32768)),
        "rope_scaling": None,
    }

    if bool(ckpt_args.get("inference_rope_scaling", False)):
        model_cfg["rope_scaling"] = {
            "original_max_position_embeddings": 2048,
            "factor": 16,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.0,
            "type": "yarn",
        }

    return model_cfg


def ensure_nonempty_mask(mask, candidate_mask):
    for b in range(mask.size(0)):
        if candidate_mask[b].any() and not mask[b].any():
            valid_pos = torch.nonzero(candidate_mask[b], as_tuple=False).view(-1)
            chosen = valid_pos[torch.randint(valid_pos.numel(), (1,), device=mask.device)]
            mask[b, chosen] = True
    return mask


def build_llada_block_diffusion_batch(
    input_ids,
    prompt_lengths,
    effective_seq_lengths,
    mask_token_id,
    block_size,
    alpha_min,
    alpha_max,
):
    bsz, max_len = input_ids.shape
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    token_pos = torch.arange(max_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
    base_noisy = input_ids.clone()
    sampled_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    t = torch.rand(bsz, device=input_ids.device) * (alpha_max - alpha_min) + alpha_min

    for b in range(bsz):
        prompt_len = int(prompt_lengths[b].item())
        seq_len = int(effective_seq_lengths[b].item())
        seq_len = max(min(seq_len, max_len), 0)
        response_len = max(seq_len - prompt_len, 0)
        if response_len <= 0:
            continue

        num_blocks = (response_len + block_size - 1) // block_size
        block_idx = int(torch.randint(num_blocks, (1,), device=input_ids.device).item())
        block_start = prompt_len + block_idx * block_size
        block_end = min(block_start + block_size, seq_len)

        current_block = (token_pos[b] >= block_start) & (token_pos[b] < block_end)
        future_block = (token_pos[b] >= block_end) & (token_pos[b] < seq_len)
        sampled = (torch.rand(max_len, device=input_ids.device) < t[b]) & current_block
        sampled = ensure_nonempty_mask(sampled.unsqueeze(0), current_block.unsqueeze(0)).squeeze(0)

        sampled_mask[b] = sampled
        base_noisy[b, future_block] = mask_token_id

    noisy = base_noisy.clone()
    noisy[sampled_mask] = mask_token_id
    return noisy, sampled_mask, t


def diffusion_time_weight_from_t(t, eps):
    alpha = torch.cos(0.5 * math.pi * t) ** 2
    alpha_prime = -0.5 * math.pi * torch.sin(math.pi * t)
    denom = torch.clamp(1.0 - alpha, min=eps)
    return torch.clamp((-alpha_prime / denom), min=eps)


def estimate_bbdlm(model, target_ids, noisy_ids, sampled_mask, t, eps):
    logits = model(noisy_ids)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    masked_token_logp = token_logp * sampled_mask.float()
    time_w = diffusion_time_weight_from_t(t, eps)
    return time_w * masked_token_logp.sum(dim=1)


def save_checkpoint(path, model, optimizer, scaler, epoch, global_step, args):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "step": global_step,
        "args": vars(args),
    }
    torch.save(payload, path)


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return []
    smoothed = []
    current_sum = sum(values[:window])
    smoothed.append(current_sum / window)
    for idx in range(window, len(values)):
        current_sum += values[idx] - values[idx - window]
        smoothed.append(current_sum / window)
    return smoothed


def save_loss_artifacts(save_dir, run_name, steps, losses, log_fn):
    if not steps:
        log_fn("No DPO loss records found, skip saving loss artifacts.")
        return

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{run_name}_dpo_loss.json")
    payload = [{"step": int(step), "loss": float(loss)} for step, loss in zip(steps, losses)]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    log_fn(f"Saved DPO loss series: {json_path}")

    if plt is None:
        log_fn("matplotlib is not installed, skip loss plotting.")
        return

    plot_path = os.path.join(save_dir, f"{run_name}_dpo_loss.png")
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="dpo loss", linewidth=1.2, alpha=0.8)
    window = min(100, max(5, len(losses) // 50))
    smoothed = moving_average(losses, window)
    if smoothed:
        smooth_steps = steps[window - 1 :]
        plt.plot(smooth_steps, smoothed, label=f"moving avg (window={window})", linewidth=2.0)
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Diffusion DPO Training Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    log_fn(f"Saved DPO loss plot: {plot_path}")


def infer_default_dpo_lr(meta):
    fallback_lr = 2.5e-6
    if not isinstance(meta, dict):
        return fallback_lr
    ckpt_args = meta.get("args") if isinstance(meta.get("args"), dict) else meta
    if not isinstance(ckpt_args, dict):
        return fallback_lr
    sft_lr = ckpt_args.get("learning_rate")
    final_lr_ratio = ckpt_args.get("final_lr_ratio")
    if isinstance(sft_lr, (int, float)) and isinstance(final_lr_ratio, (int, float)):
        if sft_lr > 0 and final_lr_ratio > 0:
            return float(sft_lr) * float(final_lr_ratio)
    return fallback_lr


def main():
    args = parse_args()
    set_seed(args.seed)
    device = auto_device(args.device)

    if not (0.0 < args.iid_mask_eps < 1.0):
        raise ValueError("--iid-mask-eps must be in (0, 1)")
    if not (0.0 < args.final_decay_ratio <= 1.0):
        raise ValueError("--final-decay-ratio must be in (0, 1]")
    if not (0.0 < args.final_lr_ratio <= 1.0):
        raise ValueError("--final-lr-ratio must be in (0, 1]")
    if args.llada2_block_size <= 0:
        raise ValueError("--llada2-block-size must be > 0")
    if not (0.0 < args.llada2_alpha_min < 1.0):
        raise ValueError("--llada2-alpha-min must be in (0, 1)")
    if not (0.0 < args.llada2_alpha_max < 1.0):
        raise ValueError("--llada2-alpha-max must be in (0, 1)")
    if args.llada2_alpha_min > args.llada2_alpha_max:
        raise ValueError("--llada2-alpha-min must be <= --llada2-alpha-max")
    if args.dpo_beta <= 0:
        raise ValueError("--dpo-beta must be > 0")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    if args.mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [args.mask_token]})
    mask_token_id = tokenizer.convert_tokens_to_ids(args.mask_token)

    print(f"Loading DPO dataset from: {args.data}")
    dataset = PreferenceConversationDataset(args.data, tokenizer, max_length=args.max_seq_len)
    print(f"DPO dataset loaded, pairs={len(dataset)}")

    collate_quant_block = args.llada2_block_size if args.llada2_quantize_effective_length else 0
    collate_fn = partial(
        collate_preference_diffusion,
        eos_token_id=tokenizer.eos_token_id,
        quantize_block_size=collate_quant_block,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    policy_raw = torch.load(args.load_from, map_location="cpu")
    policy_state, policy_meta = extract_state_dict(policy_raw)
    if not isinstance(policy_state, dict):
        raise ValueError(f"Unsupported policy checkpoint format: {args.load_from}")
    policy_state = normalize_state_dict_keys(policy_state)

    ref_path = args.reference_from if args.reference_from is not None else args.load_from
    ref_raw = torch.load(ref_path, map_location="cpu")
    ref_state, ref_meta = extract_state_dict(ref_raw)
    if not isinstance(ref_state, dict):
        raise ValueError(f"Unsupported reference checkpoint format: {ref_path}")
    ref_state = normalize_state_dict_keys(ref_state)

    policy_model_cfg = resolve_model_config(policy_state, policy_meta)
    ref_args = _extract_ckpt_args(ref_meta)
    if ref_args:
        ref_model_cfg = resolve_model_config(ref_state, ref_meta)
        if policy_model_cfg != ref_model_cfg:
            raise ValueError(
                "Policy and reference checkpoint model hyperparameters do not match. "
                "Please use checkpoints from the same SFT model configuration."
            )
        model_cfg = policy_model_cfg
    else:
        model_cfg = policy_model_cfg
    vocab_size = model_cfg["vocab_size"]

    if mask_token_id >= vocab_size:
        raise ValueError(
            f"mask_token_id ({mask_token_id}) >= model vocab size ({vocab_size}). Use the tokenizer used by checkpoint."
        )

    model = DiffusionModel(**model_cfg).to(device)
    ref_model = DiffusionModel(**model_cfg).to(device)

    m1, s1, n1 = load_matching_weights(model, policy_state)
    m2, s2, n2 = load_matching_weights(ref_model, ref_state)
    print(f"Loaded policy {args.load_from}: matched={m1}, shape_mismatch={s1}, missing_name={n1}")
    print(f"Loaded reference {ref_path}: matched={m2}, shape_mismatch={s2}, missing_name={n2}")

    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    base_lr = args.learning_rate if args.learning_rate is not None else infer_default_dpo_lr(policy_meta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    use_amp = device.type == "cuda" and args.dtype in {"float16", "bfloat16"}
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.dtype == "float16"))

    start_epoch = 0
    global_step = 0
    if isinstance(policy_meta, dict) and "optimizer_state_dict" in policy_meta:
        if args.resume_optimizer_state:
            optimizer.load_state_dict(policy_meta["optimizer_state_dict"])
            start_epoch = int(policy_meta.get("epoch", 0))
            global_step = int(policy_meta.get("step", 0))
            if "scaler_state_dict" in policy_meta:
                scaler.load_state_dict(policy_meta["scaler_state_dict"])
            print(f"Resumed optimizer/scaler from epoch={start_epoch}, step={global_step}")
        else:
            print("Checkpoint includes optimizer state, but it was skipped. Pass --resume-optimizer-state to resume.")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.run_name}.pt")
    sd_path = os.path.join(args.save_dir, f"{args.run_name}_state_dict.pt")

    steps_per_epoch = len(loader)
    total_updates = args.epochs * ((steps_per_epoch + args.accumulation_steps - 1) // args.accumulation_steps)
    update_step = 0
    start_time = time.time()
    log_fn = tqdm.write if tqdm is not None else print

    print(
        f"device={device}, pairs={len(dataset)}, batch_size={args.batch_size}, max_seq_len={args.max_seq_len}, "
        f"mask_token_id={mask_token_id}, updates={total_updates}"
    )
    print(
        "Recovered model config from checkpoint args: "
        f"n_embd={model_cfg['n_embd']}, n_layer={model_cfg['n_layer']}, "
        f"n_head={model_cfg['n_head']}, intermediate_size={model_cfg['intermediate_size']}, "
        f"dropout={model_cfg['dropout']}, hidden_act={model_cfg['hidden_act']}, "
        f"rms_norm_eps={model_cfg['rms_norm_eps']}, rope_base={model_cfg['rope_base']}, "
        f"max_pos={model_cfg['max_position_embeddings']}, rope_scaling={model_cfg['rope_scaling'] is not None}"
    )
    print(
        f"DPO config: beta={args.dpo_beta}, lr={base_lr:.8g}, block_size={args.llada2_block_size}, "
        f"alpha_band=[{args.llada2_alpha_min:.2f},{args.llada2_alpha_max:.2f}], "
        f"quantize_len={args.llada2_quantize_effective_length}"
    )

    model.train()
    train_loss_steps = []
    train_loss_values = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        epoch_update_count = 0
        optimizer.zero_grad(set_to_none=True)

        epoch_iter = (
            tqdm(loader, total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
            if tqdm is not None
            else loader
        )

        for micro_step, batch in enumerate(epoch_iter, start=1):
            chosen_ids = batch["chosen_input_ids"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)
            chosen_eff = batch["chosen_effective_seq_lengths"].to(device)
            rejected_eff = batch["rejected_effective_seq_lengths"].to(device)

            lr = llada_sft_lr(
                update_step,
                total_updates,
                base_lr,
                args.warmup_steps,
                args.final_decay_ratio,
                args.final_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            noisy_w, mask_w, t_w = build_llada_block_diffusion_batch(
                input_ids=chosen_ids,
                prompt_lengths=prompt_lengths,
                effective_seq_lengths=chosen_eff,
                mask_token_id=mask_token_id,
                block_size=args.llada2_block_size,
                alpha_min=args.llada2_alpha_min,
                alpha_max=args.llada2_alpha_max,
            )
            noisy_l, mask_l, t_l = build_llada_block_diffusion_batch(
                input_ids=rejected_ids,
                prompt_lengths=prompt_lengths,
                effective_seq_lengths=rejected_eff,
                mask_token_id=mask_token_id,
                block_size=args.llada2_block_size,
                alpha_min=args.llada2_alpha_min,
                alpha_max=args.llada2_alpha_max,
            )

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                b_policy_w = estimate_bbdlm(model, chosen_ids, noisy_w, mask_w, t_w, args.iid_mask_eps)
                b_policy_l = estimate_bbdlm(model, rejected_ids, noisy_l, mask_l, t_l, args.iid_mask_eps)

                with torch.no_grad():
                    b_ref_w = estimate_bbdlm(ref_model, chosen_ids, noisy_w, mask_w, t_w, args.iid_mask_eps)
                    b_ref_l = estimate_bbdlm(ref_model, rejected_ids, noisy_l, mask_l, t_l, args.iid_mask_eps)

                delta_w = b_policy_w - b_ref_w
                delta_l = b_policy_l - b_ref_l
                margin = args.dpo_beta * (delta_w - delta_l)
                raw_loss = -F.logsigmoid(margin).mean()
                loss = raw_loss / args.accumulation_steps
                pair_acc = (margin > 0).float().mean()

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if micro_step % args.accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                update_step += 1
                global_step += 1
                epoch_loss_sum += float(raw_loss.item())
                epoch_acc_sum += float(pair_acc.item())
                epoch_update_count += 1
                train_loss_steps.append(global_step)
                train_loss_values.append(float(raw_loss.item()))

                if global_step % args.log_interval == 0:
                    elapsed = time.time() - start_time
                    msg = (
                        f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                        f"loss={raw_loss.item():.4f} pair_acc={pair_acc.item():.4f} "
                        f"lr={lr:.8f} elapsed={elapsed:.1f}s"
                    )
                    if tqdm is not None:
                        epoch_iter.set_postfix(
                            loss=f"{raw_loss.item():.4f}",
                            pair_acc=f"{pair_acc.item():.4f}",
                            lr=f"{lr:.2e}",
                            step=global_step,
                        )
                    log_fn(msg)

                if global_step % args.save_interval == 0:
                    save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, global_step, args)
                    torch.save(model.state_dict(), sd_path)
                    save_loss_artifacts(args.save_dir, args.run_name, train_loss_steps, train_loss_values, log_fn)
                    log_fn(f"Saved checkpoint: {ckpt_path}")

        epoch_elapsed = time.time() - epoch_start
        epoch_avg_loss = (epoch_loss_sum / epoch_update_count) if epoch_update_count > 0 else float("nan")
        epoch_avg_acc = (epoch_acc_sum / epoch_update_count) if epoch_update_count > 0 else float("nan")
        log_fn(
            f"epoch={epoch + 1}/{args.epochs} done avg_loss={epoch_avg_loss:.4f} "
            f"avg_pair_acc={epoch_avg_acc:.4f} updates={epoch_update_count} epoch_time={epoch_elapsed:.1f}s"
        )

    save_checkpoint(ckpt_path, model, optimizer, scaler, args.epochs, global_step, args)
    torch.save(model.state_dict(), sd_path)
    save_loss_artifacts(args.save_dir, args.run_name, train_loss_steps, train_loss_values, print)
    print(f"Saved final checkpoint: {ckpt_path}")
    print(f"Saved state_dict: {sd_path}")


if __name__ == "__main__":
    main()

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

from scripts.eval.eval_diffusion import DiffusionModel
from core.sft_dataset import SFTConversationDataset, StreamingSFTConversationDataset, collate_sft_diffusion
from core.trainer_utils import auto_device, extract_state_dict, load_matching_weights, normalize_state_dict_keys

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA-style diffusion SFT")
    parser.add_argument("--data", type=str, required=True, help="SFT jsonl path")
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Tokenizer directory")
    parser.add_argument("--save-dir", type=str, default="weights", help="Checkpoint directory")
    parser.add_argument("--run-name", type=str, default="diffusion_sft", help="Checkpoint prefix")
    parser.add_argument("--load-from", type=str, default=None, help="Init/resume checkpoint path")
    parser.add_argument(
        "--resume-optimizer-state",
        action="store_true",
        help="When loading from full checkpoint, also resume optimizer/scaler/step",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--streaming", action="store_true", help="Stream jsonl rows and train while loading")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Required when --streaming; number of dataloader steps per epoch",
    )
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--final-decay-ratio", type=float, default=0.1)
    parser.add_argument("--final-lr-ratio", type=float, default=0.1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--mask-token", type=str, default="<|mask|>")
    parser.add_argument("--iid-mask-eps", type=float, default=1e-3, help="t ~ U(eps, 1)")
    parser.add_argument(
        "--llada2-enable-block-diffusion",
        action="store_true",
        help="Enable LLaDA2.0 Eq.(5)-style block diffusion SFT objective (default off, keeps 1.0-style iid_t)",
    )
    parser.add_argument(
        "--llada2-block-size",
        type=int,
        default=32,
        help="Block size LB for LLaDA2.0 block diffusion SFT when enabled",
    )
    parser.add_argument(
        "--llada2-alpha-min",
        type=float,
        default=0.05,
        help="Lower bound of mask ratio alpha in LLaDA2.0 block diffusion SFT",
    )
    parser.add_argument(
        "--llada2-alpha-max",
        type=float,
        default=0.95,
        help="Upper bound of mask ratio alpha in LLaDA2.0 block diffusion SFT",
    )
    parser.add_argument(
        "--llada2-enable-cap",
        action="store_true",
        help="Enable CAP confidence auxiliary loss (default off, keeps 1.0 objective)",
    )
    parser.add_argument(
        "--llada2-cap-lambda",
        type=float,
        default=0.1,
        help="CAP auxiliary loss weight lambda in L = LSFT + lambda * Lconf",
    )
    parser.add_argument(
        "--llada2-enable-complementary-masking",
        action="store_true",
        help="Enable LLaDA2.0 complementary masking (requires --llada2-enable-block-diffusion)",
    )
    parser.add_argument(
        "--llada2-quantize-effective-length",
        action="store_true",
        help="Quantize each sample effective length to nearest block multiple in block diffusion SFT (default off)",
    )

    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden-act", choices=["relu", "gelu", "silu"], default="silu")
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=1e6)
    parser.add_argument("--max-position-embeddings", type=int, default=32768)
    parser.add_argument("--inference-rope-scaling", action="store_true")

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
        # c_q shape is [hidden, hidden], infer head count from common divisors.
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


def ensure_nonempty_mask(mask, candidate_mask):
    for b in range(mask.size(0)):
        if candidate_mask[b].any() and not mask[b].any():
            valid_pos = torch.nonzero(candidate_mask[b], as_tuple=False).view(-1)
            chosen = valid_pos[torch.randint(valid_pos.numel(), (1,), device=mask.device)]
            mask[b, chosen] = True
    return mask


def build_llada_masked_batch(input_ids, prompt_lengths, mask_token_id, eps):
    bsz, seq_len = input_ids.shape
    token_positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
    prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
    response_mask = ~prompt_mask

    t = torch.rand(bsz, device=input_ids.device) * (1.0 - eps) + eps
    sampled_mask = (torch.rand_like(input_ids.float()) < t.unsqueeze(1)) & response_mask
    sampled_mask = ensure_nonempty_mask(sampled_mask, response_mask)

    noisy = input_ids.clone()
    noisy[sampled_mask] = mask_token_id
    noisy[prompt_mask] = input_ids[prompt_mask]
    return noisy, sampled_mask, response_mask, t


def diffusion_time_weight_from_t(t, eps):
    alpha = torch.cos(0.5 * math.pi * t) ** 2
    alpha_prime = -0.5 * math.pi * torch.sin(math.pi * t)
    denom = torch.clamp(1.0 - alpha, min=eps)
    return torch.clamp((-alpha_prime / denom), min=eps)


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
    block_region_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_region_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    t = torch.rand(bsz, device=input_ids.device) * (alpha_max - alpha_min) + alpha_min

    for b in range(bsz):
        prompt_len = int(prompt_lengths[b].item())
        seq_len = int(effective_seq_lengths[b].item())
        seq_len = max(min(seq_len, max_len), 0)
        response_len = max(seq_len - prompt_len, 0)
        if response_len <= 0:
            continue
        response_region = (token_pos[b] >= prompt_len) & (token_pos[b] < seq_len)
        response_region_mask[b] = response_region

        num_blocks = (response_len + block_size - 1) // block_size
        block_idx = int(torch.randint(num_blocks, (1,), device=input_ids.device).item())
        block_start = prompt_len + block_idx * block_size
        block_end = min(block_start + block_size, seq_len)

        current_block = (token_pos[b] >= block_start) & (token_pos[b] < block_end)
        future_block = (token_pos[b] >= block_end) & (token_pos[b] < seq_len)
        sampled = (torch.rand(max_len, device=input_ids.device) < t[b]) & current_block
        sampled = ensure_nonempty_mask(sampled.unsqueeze(0), current_block.unsqueeze(0)).squeeze(0)

        block_region_mask[b] = current_block
        sampled_mask[b] = sampled
        base_noisy[b, future_block] = mask_token_id

    noisy = base_noisy.clone()
    noisy[sampled_mask] = mask_token_id
    return noisy, sampled_mask, t, base_noisy, block_region_mask, response_region_mask


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
        log_fn("No train loss records found, skip saving loss artifacts.")
        return

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{run_name}_train_loss.json")
    payload = [{"step": int(step), "loss": float(loss)} for step, loss in zip(steps, losses)]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    log_fn(f"Saved train loss series: {json_path}")

    if plt is None:
        log_fn("matplotlib is not installed, skip loss plotting.")
        return

    plot_path = os.path.join(save_dir, f"{run_name}_train_loss.png")
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="train total loss", linewidth=1.2, alpha=0.8)
    window = min(100, max(5, len(losses) // 50))
    smoothed = moving_average(losses, window)
    if smoothed:
        smooth_steps = steps[window - 1 :]
        plt.plot(smooth_steps, smoothed, label=f"moving avg (window={window})", linewidth=2.0)
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("SFT Total Training Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    log_fn(f"Saved train loss plot: {plot_path}")


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
    if args.llada2_cap_lambda < 0.0:
        raise ValueError("--llada2-cap-lambda must be >= 0")
    if not args.llada2_enable_block_diffusion:
        # LLaDA1.0-compatible SFT path: keep all LLaDA2.0 knobs disabled.
        if args.llada2_enable_cap:
            print("Warning: LLaDA1.0 SFT path disables --llada2-enable-cap.")
            args.llada2_enable_cap = False
        if args.llada2_enable_complementary_masking:
            print("Warning: LLaDA1.0 SFT path disables --llada2-enable-complementary-masking.")
            args.llada2_enable_complementary_masking = False
        if args.llada2_quantize_effective_length:
            print("Warning: LLaDA1.0 SFT path disables --llada2-quantize-effective-length.")
            args.llada2_quantize_effective_length = False
    if args.llada2_enable_complementary_masking and not args.llada2_enable_block_diffusion:
        print(
            "Warning: --llada2-enable-complementary-masking is a LLaDA2.0 path and requires "
            "--llada2-enable-block-diffusion. Disabling complementary masking."
        )
        args.llada2_enable_complementary_masking = False

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    if args.mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [args.mask_token]})
    mask_token_id = tokenizer.convert_tokens_to_ids(args.mask_token)

    print(f"Loading SFT dataset from: {args.data}")
    if args.streaming:
        if args.steps_per_epoch is None or args.steps_per_epoch <= 0:
            raise ValueError("--streaming requires --steps-per-epoch > 0")
        dataset = StreamingSFTConversationDataset(args.data, tokenizer, max_length=args.max_seq_len)
        dataset_size_text = "streaming"
        print("SFT dataset initialized in streaming mode.")
    else:
        dataset = SFTConversationDataset(args.data, tokenizer, max_length=args.max_seq_len)
        dataset_size_text = str(len(dataset))
        print(f"SFT dataset loaded, samples={len(dataset)}")
    collate_quant_block = args.llada2_block_size if args.llada2_quantize_effective_length else 0
    collate_fn = partial(
        collate_sft_diffusion,
        eos_token_id=tokenizer.eos_token_id,
        quantize_block_size=collate_quant_block,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(not args.streaming),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    state_dict = None
    meta = None
    if args.load_from:
        raw = torch.load(args.load_from, map_location="cpu")
        state_dict, meta = extract_state_dict(raw)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format: {args.load_from}")
        state_dict = normalize_state_dict_keys(state_dict)

    if state_dict is not None:
        inferred = infer_hparams_from_state_dict(state_dict)
        vocab_size = inferred["vocab_size"]
        hidden_size = inferred["hidden_size"]
        num_hidden_layers = inferred["num_hidden_layers"]
        num_attention_heads = inferred["num_attention_heads"]
        intermediate_size = inferred["intermediate_size"]
    else:
        vocab_size = len(tokenizer)
        hidden_size = args.hidden_size
        num_hidden_layers = args.num_hidden_layers
        num_attention_heads = args.num_attention_heads
        intermediate_size = args.intermediate_size

    if mask_token_id >= vocab_size:
        raise ValueError(
            f"mask_token_id ({mask_token_id}) >= model vocab size ({vocab_size}). "
            "Use the tokenizer used by this checkpoint."
        )

    rope_scaling = (
        {
            "original_max_position_embeddings": 2048,
            "factor": 16,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.0,
            "type": "yarn",
        }
        if args.inference_rope_scaling
        else None
    )

    model = DiffusionModel(
        vocab_size=vocab_size,
        n_embd=hidden_size,
        n_head=num_attention_heads,
        n_layer=num_hidden_layers,
        intermediate_size=intermediate_size if intermediate_size is not None else int(hidden_size * 8 / 3),
        dropout=args.dropout,
        hidden_act=args.hidden_act,
        rms_norm_eps=args.rms_norm_eps,
        rope_base=args.rope_theta,
        max_position_embeddings=args.max_position_embeddings,
        rope_scaling=rope_scaling,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and args.dtype in {"float16", "bfloat16"}
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.dtype == "float16"))

    start_epoch = 0
    global_step = 0
    if state_dict is not None:
        matched, shape_mismatch, missing_name = load_matching_weights(model, state_dict)
        print(
            f"Loaded {args.load_from}: matched={matched}, shape_mismatch={shape_mismatch}, missing_name={missing_name}"
        )
        if isinstance(meta, dict) and "optimizer_state_dict" in meta:
            if args.resume_optimizer_state:
                optimizer.load_state_dict(meta["optimizer_state_dict"])
                start_epoch = int(meta.get("epoch", 0))
                global_step = int(meta.get("step", 0))
                if "scaler_state_dict" in meta:
                    scaler.load_state_dict(meta["scaler_state_dict"])
                print(f"Resumed optimizer/scaler from epoch={start_epoch}, step={global_step}")
            else:
                print(
                    "Checkpoint includes optimizer state, but it was skipped. "
                    "Pass --resume-optimizer-state to resume full training state."
                )

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.run_name}.pt")
    sd_path = os.path.join(args.save_dir, f"{args.run_name}_state_dict.pt")

    steps_per_epoch = args.steps_per_epoch if args.streaming else len(loader)
    if steps_per_epoch is None or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0.")
    total_updates = args.epochs * ((steps_per_epoch + args.accumulation_steps - 1) // args.accumulation_steps)
    update_step = 0
    start_time = time.time()
    log_fn = tqdm.write if tqdm is not None else print

    print(
        f"device={device}, samples={dataset_size_text}, batch_size={args.batch_size}, max_seq_len={args.max_seq_len}, "
        f"mask_token_id={mask_token_id}, updates={total_updates}"
    )
    print(
        f"LLaDA2.0 switches: block_diffusion={args.llada2_enable_block_diffusion}, "
        f"LB={args.llada2_block_size}, alpha_band=[{args.llada2_alpha_min:.2f},{args.llada2_alpha_max:.2f}], "
        f"CAP={args.llada2_enable_cap}, cap_lambda={args.llada2_cap_lambda}, "
        f"complementary={args.llada2_enable_complementary_masking}, "
        f"quantize_len={args.llada2_quantize_effective_length}"
    )
    if tqdm is None:
        print("tqdm not installed; fallback to plain logs.")

    model.train()
    train_loss_steps = []
    train_loss_values = []
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_masked_sum = 0
        epoch_update_count = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_iter = (
            tqdm(
                loader,
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                dynamic_ncols=True,
            )
            if tqdm is not None
            else loader
        )
        for micro_step, batch in enumerate(epoch_iter, start=1):
            if args.streaming and micro_step > steps_per_epoch:
                break
            input_ids = batch["input_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)
            answer_lengths = batch["answer_lengths"].to(device).float().clamp(min=1.0)
            effective_seq_lengths = batch["effective_seq_lengths"].to(device)

            lr = llada_sft_lr(
                update_step,
                total_updates,
                args.learning_rate,
                args.warmup_steps,
                args.final_decay_ratio,
                args.final_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            if args.llada2_enable_block_diffusion:
                noisy_ids, masked_indices, t, base_noisy, block_region_mask, response_region_mask = build_llada_block_diffusion_batch(
                    input_ids=input_ids,
                    prompt_lengths=prompt_lengths,
                    effective_seq_lengths=effective_seq_lengths,
                    mask_token_id=mask_token_id,
                    block_size=args.llada2_block_size,
                    alpha_min=args.llada2_alpha_min,
                    alpha_max=args.llada2_alpha_max,
                )
                response_mask = response_region_mask
            else:
                noisy_ids, masked_indices, response_mask, t = build_llada_masked_batch(
                    input_ids=input_ids,
                    prompt_lengths=prompt_lengths,
                    mask_token_id=mask_token_id,
                    eps=args.iid_mask_eps,
                )
                base_noisy = None
                block_region_mask = None
                response_region_mask = None

            train_input_ids = input_ids
            train_noisy_ids = noisy_ids
            train_masked_indices = masked_indices
            train_t = t
            if args.llada2_enable_block_diffusion and args.llada2_enable_complementary_masking:
                complementary_mask = response_region_mask & (~masked_indices)
                complementary_noisy = input_ids.clone()
                complementary_noisy[complementary_mask] = mask_token_id
                train_input_ids = torch.cat([input_ids, input_ids], dim=0)
                train_noisy_ids = torch.cat([noisy_ids, complementary_noisy], dim=0)
                train_masked_indices = torch.cat([masked_indices, complementary_mask], dim=0)
                train_t = torch.cat([t, t], dim=0)

            with autocast_ctx:
                logits = model(train_noisy_ids)
                ce = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    train_input_ids.view(-1),
                    reduction="none",
                ).view_as(train_input_ids)

                masked_ce = ce * train_masked_indices.float()
                if args.llada2_enable_block_diffusion:
                    time_w = diffusion_time_weight_from_t(train_t, args.iid_mask_eps)
                    weighted_mask = train_masked_indices.float() * time_w.unsqueeze(1)
                    weighted_sum = (masked_ce * time_w.unsqueeze(1)).sum()
                    weight_denom = torch.clamp(weighted_mask.sum(), min=1.0)
                    lsft = weighted_sum / weight_denom
                else:
                    per_sample = masked_ce.sum(dim=1) / (t * answer_lengths)
                    lsft = per_sample.mean()

                lconf = logits.new_zeros(())
                if args.llada2_enable_cap:
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=-1)
                        correct_mask = (pred == train_input_ids) & train_masked_indices
                    if correct_mask.any():
                        log_probs = F.log_softmax(logits.float(), dim=-1)
                        probs = log_probs.exp()
                        entropy = -(probs * log_probs).sum(dim=-1)
                        lconf = entropy[correct_mask].mean().to(logits.dtype)

                raw_loss = lsft + args.llada2_cap_lambda * lconf
                loss = raw_loss / args.accumulation_steps

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
                masked_tokens = int(train_masked_indices.sum().item())
                epoch_loss_sum += float(raw_loss.item())
                epoch_masked_sum += masked_tokens
                epoch_update_count += 1
                train_loss_steps.append(global_step)
                train_loss_values.append(float(raw_loss.item()))

                if global_step % args.log_interval == 0:
                    elapsed = time.time() - start_time
                    msg = (
                        f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                        f"loss={raw_loss.item():.4f} lsft={lsft.item():.4f} "
                        f"lconf={lconf.item():.4f} lr={lr:.8f} masked={masked_tokens} elapsed={elapsed:.1f}s"
                    )
                    if tqdm is not None:
                        epoch_iter.set_postfix(
                            loss=f"{raw_loss.item():.4f}",
                            lsft=f"{lsft.item():.4f}",
                            lconf=f"{lconf.item():.4f}",
                            lr=f"{lr:.2e}",
                            masked=masked_tokens,
                            step=global_step,
                        )
                    log_fn(msg)

                if global_step % args.save_interval == 0:
                    save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, global_step, args)
                    torch.save(model.state_dict(), sd_path)
                    log_fn(f"Saved checkpoint: {ckpt_path}")

        epoch_elapsed = time.time() - epoch_start
        epoch_avg_loss = (epoch_loss_sum / epoch_update_count) if epoch_update_count > 0 else float("nan")
        epoch_avg_masked = (epoch_masked_sum / epoch_update_count) if epoch_update_count > 0 else 0.0
        log_fn(
            f"epoch={epoch + 1}/{args.epochs} done avg_loss={epoch_avg_loss:.4f} "
            f"avg_masked={epoch_avg_masked:.1f} updates={epoch_update_count} epoch_time={epoch_elapsed:.1f}s"
        )

    save_checkpoint(ckpt_path, model, optimizer, scaler, args.epochs, global_step, args)
    torch.save(model.state_dict(), sd_path)
    save_loss_artifacts(args.save_dir, args.run_name, train_loss_steps, train_loss_values, print)
    print(f"Saved final checkpoint: {ckpt_path}")
    print(f"Saved state_dict: {sd_path}")


if __name__ == "__main__":
    main()

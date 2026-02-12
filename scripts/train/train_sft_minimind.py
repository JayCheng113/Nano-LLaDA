import argparse
import json
import os
import time
from contextlib import nullcontext
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from core.model_minimind import MiniMindConfig, MiniMindForCausalLM
from core.sft_dataset import SFTConversationDataset, StreamingSFTConversationDataset, collate_sft_ar
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
    parser = argparse.ArgumentParser(description="MiniMind SFT (autoregressive)")
    parser.add_argument("--data", type=str, required=True, help="SFT jsonl path")
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Tokenizer directory")
    parser.add_argument("--save-dir", type=str, default="weights", help="Checkpoint directory")
    parser.add_argument("--run-name", type=str, default="minimind_sft", help="Checkpoint prefix")
    parser.add_argument("--load-from", type=str, default=None, help="Init/resume checkpoint path")

    parser.add_argument("--epochs", type=int, default=2)
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
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)

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


def cosine_lr_with_warmup(step, total_steps, base_lr, warmup_steps, min_lr_ratio):
    if total_steps <= 0:
        return base_lr
    min_lr = base_lr * min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
    return min_lr + (base_lr - min_lr) * cosine


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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")

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
    collate_fn = partial(collate_sft_ar, eos_token_id=tokenizer.eos_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(not args.streaming),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    config = MiniMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        hidden_act=args.hidden_act,
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=args.rope_theta,
        max_position_embeddings=args.max_position_embeddings,
        inference_rope_scaling=args.inference_rope_scaling,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attention_is_causal=True,
    )
    model = MiniMindForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    use_amp = device.type == "cuda" and args.dtype in {"float16", "bfloat16"}
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.dtype == "float16"))

    start_epoch = 0
    global_step = 0
    if args.load_from:
        raw = torch.load(args.load_from, map_location="cpu")
        state_dict, meta = extract_state_dict(raw)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format: {args.load_from}")
        state_dict = normalize_state_dict_keys(state_dict)
        matched, shape_mismatch, missing_name = load_matching_weights(model, state_dict)
        print(
            f"Loaded {args.load_from}: matched={matched}, shape_mismatch={shape_mismatch}, missing_name={missing_name}"
        )
        if isinstance(meta, dict) and "optimizer_state_dict" in meta:
            optimizer.load_state_dict(meta["optimizer_state_dict"])
            start_epoch = int(meta.get("epoch", 0))
            global_step = int(meta.get("step", 0))
            if "scaler_state_dict" in meta:
                scaler.load_state_dict(meta["scaler_state_dict"])
            print(f"Resumed optimizer/scaler from epoch={start_epoch}, step={global_step}")

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
        f"device={device}, samples={dataset_size_text}, batch_size={args.batch_size}, "
        f"max_seq_len={args.max_seq_len}, updates={total_updates}"
    )
    if tqdm is None:
        print("tqdm not installed; fallback to plain logs.")

    model.train()
    train_loss_steps = []
    train_loss_values = []
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_loss_sum = 0.0
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
            input_ids, labels, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            lr = cosine_lr_with_warmup(
                update_step,
                total_updates,
                args.learning_rate,
                args.warmup_steps,
                args.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            with autocast_ctx:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                raw_loss = out.loss
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
                epoch_loss_sum += float(raw_loss.item())
                epoch_update_count += 1
                train_loss_steps.append(global_step)
                train_loss_values.append(float(raw_loss.item()))

                if global_step % args.log_interval == 0:
                    elapsed = time.time() - start_time
                    msg = (
                        f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                        f"loss={raw_loss.item():.4f} lr={lr:.8f} elapsed={elapsed:.1f}s"
                    )
                    if tqdm is not None:
                        epoch_iter.set_postfix(loss=f"{raw_loss.item():.4f}", lr=f"{lr:.2e}", step=global_step)
                    log_fn(msg)

                if global_step % args.save_interval == 0:
                    save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, global_step, args)
                    torch.save(model.state_dict(), sd_path)
                    log_fn(f"Saved checkpoint: {ckpt_path}")

        epoch_elapsed = time.time() - epoch_start
        epoch_avg_loss = (epoch_loss_sum / epoch_update_count) if epoch_update_count > 0 else float("nan")
        log_fn(
            f"epoch={epoch + 1}/{args.epochs} done avg_loss={epoch_avg_loss:.4f} "
            f"updates={epoch_update_count} epoch_time={epoch_elapsed:.1f}s"
        )

    save_checkpoint(ckpt_path, model, optimizer, scaler, args.epochs, global_step, args)
    torch.save(model.state_dict(), sd_path)
    save_loss_artifacts(args.save_dir, args.run_name, train_loss_steps, train_loss_values, print)
    print(f"Saved final checkpoint: {ckpt_path}")
    print(f"Saved state_dict: {sd_path}")


if __name__ == "__main__":
    main()

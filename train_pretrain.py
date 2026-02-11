import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lm_dataset import PretrainDataset
from model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer_utils import auto_device, extract_state_dict

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind causal pretraining")
    parser.add_argument("--data", type=str, default="./dataset/pretrain_hq.jsonl", help="Path to .jsonl/.txt pretrain corpus")
    parser.add_argument("--jsonl-field", type=str, default="text", help="Field name for jsonl rows")
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Directory containing tokenizer files")
    parser.add_argument(
        "--tokenizer-json",
        type=str,
        default=None,
        help="Load tokenizer directly from a tokenizer JSON file (e.g. ./tokenizer_xxx.json)",
    )
    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="Train a new ByteLevel BPE tokenizer from --data before pretraining",
    )
    parser.add_argument(
        "--retrained-tokenizer-json-name",
        type=str,
        default="tokenizer_retrained.json",
        help="Output file name for retrained tokenizer JSON (must not be tokenizer.json)",
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=6400,
        help="Vocab size when --retrain-tokenizer is enabled",
    )
    parser.add_argument(
        "--tokenizer-train-max-lines",
        type=int,
        default=10000,
        help="Max rows/lines used to train tokenizer (-1 means no limit)",
    )
    parser.add_argument("--mask-token", type=str, default="<|mask|>", help="Mask token used by diffusion tokenizer flow")
    parser.add_argument("--target-vocab-size", type=int, default=6400, help="If > current vocab size, append extra tokens to reach this size")
    parser.add_argument("--save-dir", type=str, default="weights", help="Directory to save checkpoints")
    parser.add_argument("--run-name", type=str, default="minimind_pretrain", help="Checkpoint name prefix")
    parser.add_argument("--load-from", type=str, default=None, help="Optional checkpoint path for resume/init")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_texts(data_path: str, text_field: str, max_lines: int):
    lines_seen = 0
    is_jsonl = str(data_path).endswith(".jsonl")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_lines > 0 and lines_seen >= max_lines:
                break
            row = line.strip()
            if not row:
                continue
            if is_jsonl:
                obj = json.loads(row)
                value = obj.get(text_field, "")
                text = value if isinstance(value, str) else str(value)
            else:
                text = row
            if not text:
                continue
            lines_seen += 1
            yield text


def train_tokenizer_to_json(data_path: str, text_field: str, vocab_size: int, output_json_path: str, max_lines: int):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iter_texts(data_path, text_field, max_lines), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    tokenizer.save(output_json_path)
    return output_json_path


def load_fast_tokenizer_from_json(tokenizer_json_path: str):
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
    )


def load_model_weights(model, checkpoint_path, device):
    raw = torch.load(checkpoint_path, map_location=device)
    state_dict, meta = extract_state_dict(raw)

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    # Compatibility: checkpoints saved from wrappers may prefix keys with "model.".
    if "model.token_emb.weight" in state_dict and "token_emb.weight" not in state_dict:
        state_dict = {
            (k[len("model.") :] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }

    model_state = model.state_dict()
    matched = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value

    missing, unexpected = model.load_state_dict(matched, strict=False)
    return missing, unexpected, meta, len(matched)


def save_checkpoint(path, model, optimizer, step, epoch, args):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "args": vars(args),
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = auto_device()
    use_amp = device.type == "cuda" and args.dtype in {"float16", "bfloat16"}
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.dtype == "float16"))

    if args.retrain_tokenizer:
        if args.retrained_tokenizer_json_name == "tokenizer.json":
            raise ValueError("--retrained-tokenizer-json-name must be different from tokenizer.json")
        retrained_json_path = os.path.join(args.tokenizer_dir, args.retrained_tokenizer_json_name)
        train_tokenizer_to_json(
            data_path=args.data,
            text_field=args.jsonl_field,
            vocab_size=args.tokenizer_vocab_size,
            output_json_path=retrained_json_path,
            max_lines=args.tokenizer_train_max_lines,
        )
        print(f"Trained tokenizer saved to: {retrained_json_path}")
        tokenizer = load_fast_tokenizer_from_json(retrained_json_path)
    elif args.tokenizer_json:
        tokenizer = load_fast_tokenizer_from_json(args.tokenizer_json)
        print(f"Loaded tokenizer from JSON: {args.tokenizer_json}")
    else:
        default_tokenizer_json = os.path.join(args.tokenizer_dir, "tokenizer.json")
        if os.path.isfile(default_tokenizer_json):
            tokenizer = load_fast_tokenizer_from_json(default_tokenizer_json)
            print(f"Loaded default tokenizer JSON: {default_tokenizer_json}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [args.mask_token]})
    if args.target_vocab_size is not None and args.target_vocab_size > len(tokenizer):
        extra_count = args.target_vocab_size - len(tokenizer)
        tokenizer.add_tokens([f"<|extra_{i}|>" for i in range(extra_count)])

    dataset = PretrainDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        text_field=args.jsonl_field,
        max_length=args.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0

    if args.load_from:
        missing, unexpected, meta, matched = load_model_weights(model, args.load_from, device)
        print(f"Loaded init weights: {args.load_from}")
        print(f"  matched keys: {matched}, missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if matched == 0:
            raise ValueError("No compatible tensors loaded from --load-from checkpoint.")
        if meta and "optimizer_state_dict" in meta:
            optimizer.load_state_dict(meta["optimizer_state_dict"])
            start_epoch = int(meta.get("epoch", 0))
            global_step = int(meta.get("step", 0))
            print(f"  resumed optimizer from epoch={start_epoch}, step={global_step}")

    os.makedirs(args.save_dir, exist_ok=True)
    final_path = os.path.join(args.save_dir, f"{args.run_name}.pt")
    state_dict_path = os.path.join(args.save_dir, f"{args.run_name}_state_dict.pt")

    print(
        f"Device={device}, samples={len(dataset)}, batch_size={args.batch_size}, "
        f"seq_len={args.max_seq_len}, vocab={len(tokenizer)}, mask_token={args.mask_token}"
    )

    start_time = time.time()
    model.train()
    for epoch in range(start_epoch, args.epochs):
        for step, batch in enumerate(loader, start=1):
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            with autocast_ctx:
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            global_step += 1

            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                    f"loss={loss.item():.4f} elapsed={elapsed:.1f}s"
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(final_path, model, optimizer, global_step, epoch, args)
                torch.save(model.state_dict(), state_dict_path)
                print(f"Saved checkpoint: {final_path}")

    save_checkpoint(final_path, model, optimizer, global_step, args.epochs, args)
    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved final checkpoint: {final_path}")
    print(f"Saved state_dict for diffusion init: {state_dict_path}")


if __name__ == "__main__":
    main()

import argparse
import difflib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from core.trainer_utils import auto_device
from scripts.eval.eval_diffusion import (
    DiffusionModel,
    extract_model_state_dict,
    generate,
    infer_hparams_from_state_dict,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LLaDA1.0-style and LLaDA2.0-style diffusion decoding outputs"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion checkpoint")
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Tokenizer directory")
    parser.add_argument("--seq-len", type=int, default=256)

    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt text. Can be specified multiple times.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional prompt file (.txt: one prompt per line; .jsonl: field from --prompt-field)",
    )
    parser.add_argument("--prompt-field", type=str, default="prompt", help="Field used when --prompt-file is jsonl")
    parser.add_argument("--max-samples", type=int, default=0, help="Only keep first N prompts (0 means all)")

    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--gen-steps", type=int, default=64)
    parser.add_argument("--gen-cfg-scale", type=float, default=1.5)
    parser.add_argument("--gen-temp", type=float, default=0.8)
    parser.add_argument("--gen-top-k", type=int, default=1)
    parser.add_argument("--gen-repeat-penalty", type=float, default=0.0)
    parser.add_argument("--gen-repeat-window", type=int, default=128)

    parser.add_argument(
        "--gen-confidence-threshold",
        type=float,
        default=0.9,
        help="2.0 CAP confidence threshold",
    )
    parser.add_argument("--gen-cap-start-ratio", type=float, default=0.08)
    parser.add_argument("--gen-cap-end-ratio", type=float, default=0.5)
    parser.add_argument("--gen-max-decode-per-step", type=int, default=32)
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "pretrain", "sft"],
        help="Prompt handling mode: pretrain->plain text, sft->chat wrapper, auto->infer from prompt",
    )

    parser.add_argument(
        "--no-chat-wrap",
        action="store_true",
        help="Force disable chat wrapping (highest priority, mainly for pretrain checkpoints)",
    )
    parser.add_argument("--mask-token", type=str, default="<|mask|>")
    parser.add_argument("--output", type=str, default="outputs/llada_style_compare.jsonl")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def read_prompts(args):
    prompts = []
    if args.prompt:
        prompts.extend([p for p in args.prompt if p and p.strip()])

    if args.prompt_file:
        p = Path(args.prompt_file)
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = str(obj.get(args.prompt_field, "")).strip()
                    if text:
                        prompts.append(text)
        else:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        prompts.append(text)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in prompts:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    if args.max_samples > 0:
        uniq = uniq[: args.max_samples]

    if not uniq:
        raise ValueError("No prompt provided. Use --prompt and/or --prompt-file")
    return uniq


def load_diffusion_model(checkpoint, tokenizer, mask_token, device):
    ckpt_raw = torch.load(checkpoint, map_location="cpu")
    state_dict, ckpt_args = extract_model_state_dict(ckpt_raw)
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format")

    ckpt_vocab_size = state_dict["token_emb.weight"].shape[0]
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    if len(tokenizer) < ckpt_vocab_size:
        tokenizer.add_tokens([f"<|extra_eval_{i}|>" for i in range(ckpt_vocab_size - len(tokenizer))])
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    if mask_token_id >= ckpt_vocab_size:
        raise ValueError(
            f"mask_token_id ({mask_token_id}) >= checkpoint vocab size ({ckpt_vocab_size}). "
            "Please evaluate with the tokenizer used in training."
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
    if len(matched) == 0:
        raise ValueError("No tensors matched. Check checkpoint and tokenizer.")

    print("=== Load Report ===")
    print(f"checkpoint: {checkpoint}")
    print(f"matched_tensors: {len(matched)}")
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")
    print(f"hparams: hidden={h['n_embd']}, layers={h['n_layer']}, heads={h['n_head']}, vocab={h['vocab_size']}")
    return model.eval(), mask_token_id


def wrap_prompt(prompt, task="auto", no_chat_wrap=False):
    if no_chat_wrap:
        return prompt
    if task == "pretrain":
        return prompt
    if task == "sft":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if "<|im_start|>" in prompt:
        return prompt
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def run_generate(model, tokenizer, mask_token_id, prompt, args, device, llada2_enable_parallel_decoding=False):
    wrapped_prompt = wrap_prompt(prompt, task=args.task, no_chat_wrap=args.no_chat_wrap)
    prompt_ids = tokenizer(wrapped_prompt, add_special_tokens=False).input_ids
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
        llada2_enable_parallel_decoding=llada2_enable_parallel_decoding,
    )
    return wrapped_prompt, out_text


def make_diff(a, b):
    return "\n".join(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            fromfile="llada1.0_style",
            tofile="llada2.0_style",
            lineterm="",
        )
    )


def main():
    args = parse_args()
    if args.gen_steps <= 0:
        raise ValueError("--gen-steps must be > 0")
    if args.gen_cfg_scale < 0:
        raise ValueError("--gen-cfg-scale must be >= 0")

    prompts = read_prompts(args)
    device = auto_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, mask_token_id = load_diffusion_model(args.checkpoint, tokenizer, args.mask_token, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, prompt in enumerate(prompts):
            used_prompt, out_v1 = run_generate(
                model,
                tokenizer,
                mask_token_id,
                prompt,
                args,
                device,
                llada2_enable_parallel_decoding=False,
            )
            _, out_v2 = run_generate(
                model,
                tokenizer,
                mask_token_id,
                prompt,
                args,
                device,
                llada2_enable_parallel_decoding=True,
            )
            diff_text = make_diff(out_v1, out_v2)

            record = {
                "index": idx,
                "prompt": prompt,
                "prompt_used": used_prompt,
                "llada1_output": out_v1,
                "llada2_output": out_v2,
                "same_output": out_v1 == out_v2,
                "diff": diff_text,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"\n===== Sample {idx} =====")
            print(f"Prompt: {prompt}")
            print("[LLaDA1.0 style]")
            print(out_v1)
            print("[LLaDA2.0 style]")
            print(out_v2)
            if diff_text:
                print("[Diff]")
                print(diff_text)
            else:
                print("[Diff] identical")

    print(f"\nSaved comparison results to: {output_path}")


if __name__ == "__main__":
    main()

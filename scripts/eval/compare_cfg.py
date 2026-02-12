import argparse

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
    parser = argparse.ArgumentParser(description="Compare diffusion outputs with and without CFG")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion checkpoint")
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Tokenizer directory")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--gen-steps", type=int, default=64)
    parser.add_argument("--cfg-on-scale", type=float, default=1.5)
    parser.add_argument("--cfg-off-scale", type=float, default=0.0)
    parser.add_argument("--gen-repeat-penalty", type=float, default=0.0)
    parser.add_argument("--gen-repeat-window", type=int, default=128)
    parser.add_argument("--mask-token", type=str, default="<|mask|>")
    parser.add_argument(
        "--no-chat-wrap",
        action="store_true",
        help="Disable automatic <|im_start|>user/.../<|im_start|>assistant wrapping",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


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
    print(f"matched_tensors: {len(matched)}")
    print(f"missing_keys: {len(missing)}")
    print(f"unexpected_keys: {len(unexpected)}")
    return model.eval(), mask_token_id


@torch.no_grad()
def run_once(model, tokenizer, prompt, mask_token_id, device, args, cfg_scale):
    wrapped_prompt = prompt
    if not args.no_chat_wrap and "<|im_start|>" not in prompt:
        wrapped_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer(wrapped_prompt, add_special_tokens=False).input_ids
    if tokenizer.bos_token_id is not None:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_ids,
        mask_token_id=mask_token_id,
        device=device,
        block_size=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        temp=1.0,
        confidence_threshold=0.0,
        top_k=1,
        repeat_penalty=args.gen_repeat_penalty,
        repeat_window=args.gen_repeat_window,
        cap_start_ratio=1.0,
        cap_end_ratio=1.0,
        max_decode_per_step=0,
        gen_steps=args.gen_steps,
        cfg_scale=cfg_scale,
    ), wrapped_prompt


def main():
    args = parse_args()
    if args.gen_steps <= 0:
        raise ValueError("--gen-steps must be > 0")
    if args.cfg_on_scale < 0 or args.cfg_off_scale < 0:
        raise ValueError("--cfg-on-scale/--cfg-off-scale must be >= 0")

    device = auto_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, mask_token_id = load_diffusion_model(args.checkpoint, tokenizer, args.mask_token, device)
    out_no_cfg, wrapped_prompt = run_once(
        model, tokenizer, args.prompt, mask_token_id, device, args, args.cfg_off_scale
    )
    out_cfg, _ = run_once(model, tokenizer, args.prompt, mask_token_id, device, args, args.cfg_on_scale)

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Prompt Used For Generation ===")
    print(wrapped_prompt)
    print("\n=== Output (CFG OFF) ===")
    print(f"cfg_scale={args.cfg_off_scale}")
    print(out_no_cfg)
    print("\n=== Output (CFG ON) ===")
    print(f"cfg_scale={args.cfg_on_scale}")
    print(out_cfg)


if __name__ == "__main__":
    main()

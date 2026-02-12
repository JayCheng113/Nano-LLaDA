import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from core.trainer_utils import auto_device
from scripts.eval.eval_diffusion import generate as diffusion_generate
from scripts.eval.eval_sft_one_prompt import (
    ar_generate_text,
    load_diffusion_model,
    load_minimind_model,
    maybe_prepend_bos_token_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate AR + diffusion SFT checkpoints on a HuggingFace Chinese QA dataset."
    )
    parser.add_argument("--tokenizer-dir", type=str, default=".", help="Tokenizer directory")
    parser.add_argument("--minimind-checkpoint", type=str, required=True, help="AR SFT checkpoint path")
    parser.add_argument("--diffusion-checkpoint", type=str, required=True, help="Diffusion SFT checkpoint path")

    parser.add_argument("--dataset", type=str, default="cmrc2018", help="HF dataset name")
    parser.add_argument("--dataset-config", type=str, default=None, help="HF dataset config/subset")
    parser.add_argument("--split", type=str, default="validation", help="HF split name")
    parser.add_argument("--question-field", type=str, default="question", help="Field name for question text")
    parser.add_argument(
        "--context-field",
        type=str,
        default="context",
        help="Optional context field name; empty string disables context",
    )
    parser.add_argument(
        "--answers-field",
        type=str,
        default="answers",
        help="Field name for reference answers (if present)",
    )
    parser.add_argument("--num-samples", type=int, default=50, help="Number of random examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--device", type=str, default=None, help="cuda/cpu/mps; default auto")
    parser.add_argument("--seq-len", type=int, default=256, help="Diffusion block size")
    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--ar-temperature", type=float, default=0.8)
    parser.add_argument("--ar-top-k", type=int, default=20)

    parser.add_argument("--gen-temp", type=float, default=0.8)
    parser.add_argument("--gen-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--gen-top-k", type=int, default=8)
    parser.add_argument("--gen-repeat-penalty", type=float, default=0.0)
    parser.add_argument("--gen-repeat-window", type=int, default=128)
    parser.add_argument("--gen-cap-start-ratio", type=float, default=0.08)
    parser.add_argument("--gen-cap-end-ratio", type=float, default=0.5)
    parser.add_argument("--gen-max-decode-per-step", type=int, default=32)
    parser.add_argument(
        "--llada2-enable-parallel-decoding",
        action="store_true",
        help="Enable LLaDA2.0 CAP-style confidence-aware parallel decoding (default off keeps 1.0-style s/t remask)",
    )
    parser.add_argument("--gen-steps", type=int, default=64)
    parser.add_argument("--gen-cfg-scale", type=float, default=1.5)
    parser.add_argument("--mask-token", type=str, default="<|mask|>")

    parser.add_argument("--include-context", action="store_true", help="Add context to the prompt if available")
    parser.add_argument("--output", type=str, default="outputs/sft_cmrc2018_eval.jsonl", help="Output .jsonl path")
    return parser.parse_args()


def build_user_content(question, context, include_context):
    if include_context and context:
        return f"请根据给定资料回答问题。\n资料：{context}\n问题：{question}"
    return question


def render_chat_messages(messages):
    chunks = []
    for m in messages:
        role = str(m.get("role", "")).strip()
        if not role:
            continue
        content = m.get("content", "")
        content = content if isinstance(content, str) else str(content)
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(chunks)


def build_chat_prompt_from_user(user_content):
    messages = [{"role": "user", "content": user_content}]
    return render_chat_messages(messages) + "<|im_start|>assistant\n"


def extract_assistant_answer_from_ids(full_text, tokenizer, prompt_ids):
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    answer_ids = []
    if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        answer_ids = full_ids[len(prompt_ids) :]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=False)
    else:
        # Fallback for decode/encode mismatch: recover by assistant marker in text.
        marker = "<|im_start|>assistant\n"
        if marker in full_text:
            answer = full_text.split(marker)[-1]
        else:
            answer = full_text
    if "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>", 1)[0]
    if "<|im_start|>" in answer:
        answer = answer.split("<|im_start|>", 1)[0]
    return answer.strip()


def build_diffusion_prompt_ids(tokenizer, wrapped_prompt, seq_len):
    prompt_ids = tokenizer(wrapped_prompt, add_special_tokens=False).input_ids
    prompt_ids = maybe_prepend_bos_token_ids(prompt_ids, tokenizer.bos_token_id)
    max_prompt_len = max(int(seq_len) - 1, 1)
    truncated = 0
    if len(prompt_ids) > max_prompt_len:
        truncated = len(prompt_ids) - max_prompt_len
        # Keep tail tokens to preserve the latest user turn and assistant prefix,
        # aligned with SFT dataset truncation behavior.
        prompt_ids = prompt_ids[-max_prompt_len:]
    return prompt_ids, truncated


def unpack_reference_answers(x):
    if isinstance(x, dict):
        texts = x.get("text")
        if isinstance(texts, list):
            return [str(t) for t in texts]
        if isinstance(texts, str):
            return [texts]
    if isinstance(x, list):
        return [str(t) for t in x]
    if isinstance(x, str):
        return [x]
    return []


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.gen_steps <= 0:
        raise ValueError("--gen-steps must be > 0")

    device = auto_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"device: {device}")
    print(f"loading dataset: {args.dataset}, split={args.split}")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)

    valid_indices = []
    for i, row in enumerate(dataset):
        q = row.get(args.question_field, None)
        if isinstance(q, str) and q.strip():
            valid_indices.append(i)
    if not valid_indices:
        raise ValueError(
            f"No valid question rows found. question_field={args.question_field}, dataset={args.dataset}, split={args.split}"
        )

    random.seed(args.seed)
    random.shuffle(valid_indices)
    picked = valid_indices[: min(args.num_samples, len(valid_indices))]
    print(f"valid_rows: {len(valid_indices)}, evaluating: {len(picked)}")

    print("loading AR model...")
    ar_model = load_minimind_model(args.minimind_checkpoint, tokenizer, args.seq_len, device)
    print("loading diffusion model...")
    diffusion_model, mask_token_id = load_diffusion_model(
        args.diffusion_checkpoint, tokenizer, args.mask_token, device
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = picked
    if tqdm is not None:
        iterator = tqdm(picked, desc="Evaluating", unit="sample", dynamic_ncols=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rank, idx in enumerate(iterator, start=1):
            row = dataset[int(idx)]
            question = str(row.get(args.question_field, "")).strip()
            context = ""
            if args.context_field:
                context = str(row.get(args.context_field, "") or "").strip()
            references = unpack_reference_answers(row.get(args.answers_field, None))

            user_content = build_user_content(question, context, args.include_context)
            wrapped_prompt = build_chat_prompt_from_user(user_content)

            ar_prompt_ids = tokenizer(wrapped_prompt, add_special_tokens=False).input_ids
            ar_prompt_ids = maybe_prepend_bos_token_ids(ar_prompt_ids, tokenizer.bos_token_id)
            ar_full = ar_generate_text(
                model=ar_model,
                tokenizer=tokenizer,
                prompt=wrapped_prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.ar_temperature,
                top_k=args.ar_top_k,
            )
            ar_answer = extract_assistant_answer_from_ids(ar_full, tokenizer, ar_prompt_ids)

            prompt_ids, diff_prompt_truncated_tokens = build_diffusion_prompt_ids(
                tokenizer, wrapped_prompt, args.seq_len
            )
            diff_full = diffusion_generate(
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
                llada2_enable_parallel_decoding=args.llada2_enable_parallel_decoding,
            )
            diff_answer = extract_assistant_answer_from_ids(diff_full, tokenizer, prompt_ids)

            rec = {
                "idx": int(idx),
                "question": question,
                "context": context if args.include_context else "",
                "references": references,
                "ar_answer": ar_answer,
                "diffusion_answer": diff_answer,
                "diffusion_prompt_truncated_tokens": int(diff_prompt_truncated_tokens),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if tqdm is None and (rank % 10 == 0 or rank == len(picked)):
                print(f"done {rank}/{len(picked)}")

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

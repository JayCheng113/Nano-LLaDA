import os
import time
import argparse
import json
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # default context length (can be overridden by --seq-len)
max_iters = 25000
eval_interval = 1000
learning_rate = 3e-4
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 8
head_dim = n_embd // n_head
ffn_intermediate_size = None
ffn_dropout = 0.0
ffn_hidden_act = "silu"
rope_base = 1e6
rms_norm_eps = 1e-5
max_position_embeddings = 32768
rope_scaling = {
    "original_max_position_embeddings": 2048,
    "factor": 16,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "attention_factor": 1.0,
}
# ------------
torch.manual_seed(1337)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run tiny diffusion model")
    parser.add_argument("--train", action="store_true", help="Train from scratch")
    parser.add_argument("--hidden-size", type=int, default=n_embd)
    parser.add_argument("--num-hidden-layers", type=int, default=n_layer)
    parser.add_argument("--num-attention-heads", type=int, default=n_head)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=ffn_dropout)
    parser.add_argument("--hidden-act", default=ffn_hidden_act, choices=["relu", "gelu", "silu"])
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument(
        "--lr-schedule",
        default="wsd",
        choices=["wsd", "cosine"],
        help="Learning rate schedule type",
    )
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument(
        "--lr-stable-ratio",
        type=float,
        default=0.4,
        help="For lr-schedule=wsd: ratio of total steps kept at max lr after warmup",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--rms-norm-eps", type=float, default=rms_norm_eps)
    parser.add_argument("--rope-theta", type=float, default=rope_base)
    parser.add_argument("--max-position-embeddings", type=int, default=max_position_embeddings)
    parser.add_argument("--inference-rope-scaling", action="store_true")
    parser.add_argument("--target-vocab-size", type=int, default=6400)
    parser.add_argument(
        "--use-tokenizer",
        action="store_true",
        help="Use tokenizer.json/tokenizer_config.json via HuggingFace tokenizer",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=".",
        help="Directory containing tokenizer files",
    )
    parser.add_argument(
        "--data",
        default="data.txt",
        help="Text corpus path. Supports .txt and .jsonl",
    )
    parser.add_argument(
        "--jsonl-field",
        default="text",
        help="Field name to read from each jsonl row",
    )
    parser.add_argument(
        "--jsonl-sep",
        default="\n",
        help="Separator used when concatenating jsonl rows",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=block_size,
        help="Sequence length used for training/generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Optional checkpoint path for loading/saving weights",
    )
    parser.add_argument(
        "--init-from-minimind",
        default=None,
        help="Optional MiniMind checkpoint/state_dict path used to initialize diffusion weights when no diffusion checkpoint exists",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=max_iters,
        help="Total training steps target (supports resume)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name used for checkpoint/loss filenames",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early-stop patience measured in eval intervals (<=0 disables early stop)",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum val loss improvement to reset early-stop patience",
    )
    parser.add_argument(
        "--repeat-penalty-weight",
        type=float,
        default=0.1,
        help="Auxiliary unlikelihood loss weight for consecutive repeated target tokens",
    )
    parser.add_argument(
        "--repeat-penalty-min-run",
        type=int,
        default=2,
        help="Only penalize positions whose consecutive repeat run length >= this value",
    )
    parser.add_argument(
        "--repeat-penalty-delay-steps",
        type=int,
        default=0,
        help="Disable repeat penalty for the first N training steps",
    )
    parser.add_argument(
        "--repeat-penalty-warmup-steps",
        type=int,
        default=0,
        help="After delay, linearly warm repeat penalty weight to target over M steps",
    )
    parser.add_argument(
        "--mask-schedule",
        default="wsd",
        choices=["random", "wsd", "iid_t"],
        help="Masking schedule for diffusion training",
    )
    parser.add_argument(
        "--iid-mask-eps",
        type=float,
        default=1e-3,
        help="Numerical epsilon when sampling t for iid_t masking",
    )
    parser.add_argument(
        "--variable-length-prob",
        type=float,
        default=0.01,
        help="Probability of using random variable-length training samples in iid_t mode",
    )
    parser.add_argument(
        "--wsd-min-mask-ratio",
        type=float,
        default=0.15,
        help="WSD: minimum contiguous mask ratio in warm-up/decay",
    )
    parser.add_argument(
        "--wsd-max-mask-ratio",
        type=float,
        default=1.0,
        help="WSD: maximum mask ratio in stable phase",
    )
    parser.add_argument(
        "--wsd-warmup-ratio",
        type=float,
        default=0.2,
        help="WSD: warm-up phase ratio over total steps",
    )
    parser.add_argument(
        "--wsd-stable-ratio",
        type=float,
        default=0.6,
        help="WSD: stable phase ratio over total steps",
    )
    parser.add_argument(
        "--use-block-curriculum",
        action="store_true",
        help="Enable block-size warmup/stable/decay curriculum for masking",
    )
    parser.add_argument(
        "--wsd-phase-ratios",
        default="0.3,0.5,0.2",
        help="Phase ratios for block curriculum as warmup,stable,decay",
    )
    parser.add_argument(
        "--wsd-block-sizes-up",
        default="1,4,16,64,256",
        help="Warmup block sizes progression, comma-separated",
    )
    parser.add_argument(
        "--wsd-block-sizes-down",
        default="256,128,64,32",
        help="Decay block sizes progression, comma-separated",
    )
    parser.add_argument(
        "--time-weighted-loss",
        action="store_true",
        help="Use diffusion-style time weighting alpha'(t)/(1-alpha(t)) on masked CE loss",
    )
    parser.add_argument(
        "--time-weight-eps",
        type=float,
        default=1e-3,
        help="Numerical stability epsilon for time sampling and weighting",
    )
    parser.add_argument(
        "--use-doc-attention-mask",
        action="store_true",
        help="Apply document-level attention mask in tokenizer mode",
    )
    parser.add_argument("--gen-temp", type=float, default=0.8, help="Generation temperature")
    parser.add_argument(
        "--gen-confidence-threshold",
        type=float,
        default=0.9,
        help="Decode positions whose confidence exceeds this threshold",
    )
    parser.add_argument(
        "--gen-top-k",
        type=int,
        default=8,
        help="Top-k sampling during generation",
    )
    parser.add_argument(
        "--gen-steps",
        type=int,
        default=64,
        help="Number of reverse diffusion sampling steps (uniformly discretized)",
    )
    parser.add_argument(
        "--gen-repeat-penalty",
        type=float,
        default=0.15,
        help="Inference-time logit penalty strength for repeated tokens (0 disables)",
    )
    parser.add_argument(
        "--gen-repeat-window",
        type=int,
        default=128,
        help="Only count repeats within the last N decoded tokens (<=0 means full context)",
    )
    parser.add_argument(
        "--gen-cap-start-ratio",
        type=float,
        default=0.08,
        help="CAP decoding budget ratio at early iterations",
    )
    parser.add_argument(
        "--gen-cap-end-ratio",
        type=float,
        default=0.5,
        help="CAP decoding budget ratio at late iterations",
    )
    parser.add_argument(
        "--gen-max-decode-per-step",
        type=int,
        default=32,
        help="Maximum number of positions to finalize per decoding step (<=0 means no cap)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay",
    )
    return parser.parse_args()


def load_text_corpus(path, jsonl_field, jsonl_sep):
    if path.endswith(".jsonl"):
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                value = row.get(jsonl_field, "")
                if isinstance(value, str) and value:
                    lines.append(value)
                elif value:
                    lines.append(str(value))
        if not lines:
            raise ValueError(
                f"No usable rows found in {path}. Check --jsonl-field {jsonl_field!r}."
            )
        return jsonl_sep.join(lines)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_text_samples(path, jsonl_field):
    if path.endswith(".jsonl"):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                value = row.get(jsonl_field, "")
                if isinstance(value, str) and value:
                    samples.append(value)
                elif value:
                    samples.append(str(value))
        if not samples:
            raise ValueError(
                f"No usable rows found in {path}. Check --jsonl-field {jsonl_field!r}."
            )
        return samples

    text = load_text_corpus(path, jsonl_field, "\n")
    if not text:
        raise ValueError(f"Empty text corpus in {path}.")
    return [text]


def find_unused_char(text):
    for code in [0, 1, 2, 3, 4, 5, 6, 7]:
        ch = chr(code)
        if ch not in text:
            return ch
    raise ValueError("Failed to find an unused mask token in corpus.")


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, text_field="text", max_length=512):
        super().__init__()
        self.data_path = data_path
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id.")
        if self.bos_token_id is None or self.eos_token_id is None:
            raise ValueError("Tokenizer must define bos_token_id and eos_token_id.")

        self.is_jsonl = data_path.endswith(".jsonl")
        if self.is_jsonl:
            self.offsets = []
            with open(data_path, "r", encoding="utf-8") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    value = row.get(text_field, "")
                    if isinstance(value, str) and value:
                        self.offsets.append(offset)
                    elif value:
                        self.offsets.append(offset)
            if not self.offsets:
                raise ValueError(
                    f"No usable rows found in {data_path}. Check --jsonl-field {text_field!r}."
                )
        else:
            self.samples = load_text_samples(data_path, text_field)

    def __len__(self):
        if self.is_jsonl:
            return len(self.offsets)
        return len(self.samples)

    def __getitem__(self, index):
        if self.is_jsonl:
            with open(self.data_path, "r", encoding="utf-8") as f:
                f.seek(self.offsets[index])
                row = json.loads(f.readline())
            value = row.get(self.text_field, "")
            sample = value if isinstance(value, str) else str(value)
        else:
            sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        input_ids = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
        return torch.tensor(input_ids, dtype=torch.long)


def ensure_nonempty_mask(mask, candidate_mask):
    # Ensure at least one position is masked in each sample with valid tokens.
    for b in range(mask.size(0)):
        if candidate_mask[b].any() and not mask[b].any():
            valid_pos = torch.nonzero(candidate_mask[b], as_tuple=False).view(-1)
            chosen = valid_pos[torch.randint(valid_pos.numel(), (1,))]
            mask[b, chosen] = True
    return mask


def extract_checkpoint_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
    return ckpt_obj


def normalize_transfer_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        normalized[new_key] = value
    return normalized


def load_matching_state_dict(model, state_dict):
    model_state = model.state_dict()
    matched = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append(
                (key, tuple(value.shape), tuple(model_state[key].shape))
            )
            continue
        matched[key] = value

    missing, unexpected = model.load_state_dict(matched, strict=False)
    return missing, unexpected, skipped_shape, skipped_missing, len(matched)


def parse_csv_float_list(value, expected_len=None, name="value"):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if expected_len is not None and len(vals) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} comma-separated values")
    return vals


def parse_csv_int_list(value, name="value"):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    vals = [int(p) for p in parts]
    if not vals:
        raise ValueError(f"{name} must contain at least one integer")
    if any(v <= 0 for v in vals):
        raise ValueError(f"{name} values must be > 0")
    return vals


args = parse_args()
block_size = args.seq_len
batch_size = args.batch_size
if block_size <= 0:
    raise ValueError("--seq-len must be > 0")
if batch_size <= 0:
    raise ValueError("--batch-size must be > 0")
if args.hidden_size <= 0 or args.num_hidden_layers <= 0 or args.num_attention_heads <= 0:
    raise ValueError("hidden-size/num-hidden-layers/num-attention-heads must be > 0")
if args.hidden_size % args.num_attention_heads != 0:
    raise ValueError("hidden-size must be divisible by num-attention-heads")

n_embd = args.hidden_size
n_head = args.num_attention_heads
n_layer = args.num_hidden_layers
head_dim = n_embd // n_head
ffn_intermediate_size = args.intermediate_size
ffn_dropout = args.dropout
ffn_hidden_act = args.hidden_act
learning_rate = args.learning_rate
warmup_steps = args.warmup_steps
min_lr_ratio = args.min_lr_ratio
rms_norm_eps = args.rms_norm_eps
rope_base = args.rope_theta
max_position_embeddings = args.max_position_embeddings
max_iters = args.max_iters
if max_iters <= 0:
    raise ValueError("--max-iters must be > 0")
if learning_rate <= 0:
    raise ValueError("--learning-rate must be > 0")
if warmup_steps < 0:
    raise ValueError("--warmup-steps must be >= 0")
if not (0.0 <= args.lr_stable_ratio <= 1.0):
    raise ValueError("--lr-stable-ratio must be in [0, 1]")
if not (0.0 <= min_lr_ratio <= 1.0):
    raise ValueError("--min-lr-ratio must be in [0, 1]")
if args.repeat_penalty_weight < 0.0:
    raise ValueError("--repeat-penalty-weight must be >= 0")
if args.repeat_penalty_min_run < 2:
    raise ValueError("--repeat-penalty-min-run must be >= 2")
if args.repeat_penalty_delay_steps < 0:
    raise ValueError("--repeat-penalty-delay-steps must be >= 0")
if args.repeat_penalty_warmup_steps < 0:
    raise ValueError("--repeat-penalty-warmup-steps must be >= 0")
if not (0.0 < args.wsd_min_mask_ratio <= 1.0):
    raise ValueError("--wsd-min-mask-ratio must be in (0, 1]")
if not (0.0 < args.wsd_max_mask_ratio <= 1.0):
    raise ValueError("--wsd-max-mask-ratio must be in (0, 1]")
if args.wsd_min_mask_ratio > args.wsd_max_mask_ratio:
    raise ValueError("--wsd-min-mask-ratio must be <= --wsd-max-mask-ratio")
if not (0.0 <= args.wsd_warmup_ratio <= 1.0):
    raise ValueError("--wsd-warmup-ratio must be in [0, 1]")
if not (0.0 <= args.wsd_stable_ratio <= 1.0):
    raise ValueError("--wsd-stable-ratio must be in [0, 1]")
if args.wsd_warmup_ratio + args.wsd_stable_ratio > 1.0:
    raise ValueError("--wsd-warmup-ratio + --wsd-stable-ratio must be <= 1")
if args.iid_mask_eps <= 0 or args.iid_mask_eps >= 0.5:
    raise ValueError("--iid-mask-eps must be in (0, 0.5)")
if not (0.0 <= args.variable_length_prob <= 1.0):
    raise ValueError("--variable-length-prob must be in [0, 1]")
if args.use_block_curriculum and args.mask_schedule != "wsd":
    raise ValueError("--use-block-curriculum requires --mask-schedule wsd")
if args.time_weight_eps <= 0 or args.time_weight_eps >= 0.5:
    raise ValueError("--time-weight-eps must be in (0, 0.5)")
if args.gen_temp <= 0:
    raise ValueError("--gen-temp must be > 0")
if not (0.0 <= args.gen_confidence_threshold <= 1.0):
    raise ValueError("--gen-confidence-threshold must be in [0, 1]")
if args.gen_top_k <= 0:
    raise ValueError("--gen-top-k must be > 0")
if args.gen_steps <= 0:
    raise ValueError("--gen-steps must be > 0")
if args.gen_repeat_penalty < 0:
    raise ValueError("--gen-repeat-penalty must be >= 0")
if not (0.0 < args.gen_cap_start_ratio <= 1.0):
    raise ValueError("--gen-cap-start-ratio must be in (0, 1]")
if not (0.0 < args.gen_cap_end_ratio <= 1.0):
    raise ValueError("--gen-cap-end-ratio must be in (0, 1]")
if args.gen_max_decode_per_step < 0:
    raise ValueError("--gen-max-decode-per-step must be >= 0")
if args.early_stop_patience < 0:
    raise ValueError("--early-stop-patience must be >= 0")
if args.early_stop_min_delta < 0:
    raise ValueError("--early-stop-min-delta must be >= 0")
if args.weight_decay < 0:
    raise ValueError("--weight-decay must be >= 0")
phase_ratios = parse_csv_float_list(args.wsd_phase_ratios, expected_len=3, name="--wsd-phase-ratios")
if any(r < 0 for r in phase_ratios):
    raise ValueError("--wsd-phase-ratios values must be >= 0")
phase_ratio_sum = sum(phase_ratios)
if phase_ratio_sum <= 0:
    raise ValueError("--wsd-phase-ratios sum must be > 0")
phase_ratios = [r / phase_ratio_sum for r in phase_ratios]
wsd_block_sizes_up = parse_csv_int_list(args.wsd_block_sizes_up, name="--wsd-block-sizes-up")
wsd_block_sizes_down = parse_csv_int_list(args.wsd_block_sizes_down, name="--wsd-block-sizes-down")
if args.use_block_curriculum:
    if wsd_block_sizes_up[-1] != block_size:
        raise ValueError(
            f"--wsd-block-sizes-up must end with seq-len ({block_size}) for MDLM stable phase"
        )
    if wsd_block_sizes_down[0] != block_size:
        raise ValueError(
            f"--wsd-block-sizes-down must start with seq-len ({block_size}) for decay phase"
        )
    all_blocks = set(wsd_block_sizes_up + wsd_block_sizes_down)
    non_divisors = [b for b in sorted(all_blocks) if block_size % b != 0]
    if non_divisors:
        raise ValueError(
            f"All curriculum block sizes must divide seq-len={block_size}, got invalid {non_divisors}"
        )
if args.use_doc_attention_mask and not args.use_tokenizer:
    print("Warning: --use-doc-attention-mask only applies in tokenizer mode; disabling it.")
    args.use_doc_attention_mask = False
if args.mask_schedule == "iid_t" and args.time_weighted_loss:
    print(
        "Warning: mask-schedule=iid_t uses Eq.3 inverse-t weighting by default; "
        "--time-weighted-loss will be ignored."
    )
    args.time_weighted_loss = False
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

if args.use_tokenizer:
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for --use-tokenizer. Install with: uv add transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    mask_token = "<|mask|>"
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    if args.target_vocab_size is not None and args.target_vocab_size > len(tokenizer):
        extra_count = args.target_vocab_size - len(tokenizer)
        tokenizer.add_tokens([f"<|extra_{i}|>" for i in range(extra_count)])
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    dataset = PretrainDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        text_field=args.jsonl_field,
        max_length=block_size,
    )
    if len(dataset) < 2:
        raise ValueError("Need at least 2 samples for train/val split in tokenizer mode.")
    train_size = int(0.9 * len(dataset))
    train_size = min(max(1, train_size), len(dataset) - 1)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1337),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    loader_iters = {"train": iter(train_loader), "val": iter(val_loader)}

    first_sample = train_dataset[0]
    prompt_tokens = first_sample[first_sample != pad_token_id].tolist()
    if len(prompt_tokens) < 2:
        raise ValueError("First training sample has too few non-pad tokens.")
    repeat_ignore_token_ids = [
        tid
        for tid in [pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
        if tid is not None
    ]

    def decode(l):
        return tokenizer.decode(l, skip_special_tokens=False)

else:
    text = load_text_corpus(args.data, args.jsonl_field, args.jsonl_sep)
    if len(text) <= block_size:
        raise ValueError(
            f"Corpus too short ({len(text)} chars). Need > block_size={block_size}."
        )

    # All the unique characters that occur in this text
    chars = sorted(list(set(text)))
    mask_char = find_unused_char(text)
    chars = [mask_char] + chars
    vocab_size = len(chars)
    # Create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    mask_token_id = stoi[mask_char]
    pad_token_id = None

    # encoder: take a string, output a list of integers
    def encode(s):
        return [stoi[ch] for ch in s]

    # decoder: take a list of integers, output a string
    def decode(l):
        return "".join([itos[n] for n in l])

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    prompt_tokens = data[:16].tolist()
    repeat_ignore_token_ids = []


def get_curriculum_phase(step):
    if step is None:
        return "stable", 1.0
    warmup_steps = int(max_iters * phase_ratios[0])
    stable_steps = int(max_iters * phase_ratios[1])
    decay_steps = max(max_iters - warmup_steps - stable_steps, 0)

    if warmup_steps > 0 and step < warmup_steps:
        return "warmup", (step + 1) / warmup_steps
    if step < warmup_steps + stable_steps:
        return "stable", 1.0
    if decay_steps <= 0:
        return "decay", 1.0
    decay_idx = step - warmup_steps - stable_steps
    return "decay", min(max((decay_idx + 1) / decay_steps, 0.0), 1.0)


def get_curriculum_block_size(step):
    if not args.use_block_curriculum:
        return block_size
    phase, progress = get_curriculum_phase(step)
    if phase == "stable":
        return wsd_block_sizes_up[-1]
    if phase == "warmup":
        idx = int(progress * len(wsd_block_sizes_up))
        idx = min(max(idx, 0), len(wsd_block_sizes_up) - 1)
        return wsd_block_sizes_up[idx]
    idx = int(progress * len(wsd_block_sizes_down))
    idx = min(max(idx, 0), len(wsd_block_sizes_down) - 1)
    return wsd_block_sizes_down[idx]


def get_wsd_mask_ratio(step):
    if step is None:
        return args.wsd_max_mask_ratio

    warmup_steps = int(max_iters * args.wsd_warmup_ratio)
    stable_steps = int(max_iters * args.wsd_stable_ratio)
    decay_steps = max(max_iters - warmup_steps - stable_steps, 0)
    min_ratio = args.wsd_min_mask_ratio
    max_ratio = args.wsd_max_mask_ratio

    if warmup_steps > 0 and step < warmup_steps:
        p = (step + 1) / warmup_steps
        return min_ratio + (max_ratio - min_ratio) * p
    if step < warmup_steps + stable_steps:
        return max_ratio
    if decay_steps <= 0:
        return max_ratio

    decay_idx = step - warmup_steps - stable_steps
    p = min(max((decay_idx + 1) / decay_steps, 0.0), 1.0)
    return max_ratio - (max_ratio - min_ratio) * p


def sample_contiguous_block_mask(candidate_mask, mask_ratio):
    bsz, seq_len = candidate_mask.size()
    mask = torch.zeros_like(candidate_mask)
    for b in range(bsz):
        valid_pos = torch.nonzero(candidate_mask[b], as_tuple=False).view(-1)
        if valid_pos.numel() == 0:
            continue
        span = max(1, int(round(valid_pos.numel() * mask_ratio)))
        span = min(span, valid_pos.numel())
        start = 0 if span == valid_pos.numel() else torch.randint(
            0, valid_pos.numel() - span + 1, (1,)
        ).item()
        chosen = valid_pos[start : start + span]
        mask[b, chosen] = True
    return mask


def sample_blockwise_mask(candidate_mask, mask_ratio, block_len):
    bsz, seq_len = candidate_mask.size()
    mask = torch.zeros_like(candidate_mask)
    block_len = max(1, min(block_len, seq_len))
    for b in range(bsz):
        block_ranges = []
        for start in range(0, seq_len, block_len):
            end = min(start + block_len, seq_len)
            if candidate_mask[b, start:end].any():
                block_ranges.append((start, end))
        if not block_ranges:
            continue
        select_n = max(1, int(round(len(block_ranges) * mask_ratio)))
        select_n = min(select_n, len(block_ranges))
        if select_n == len(block_ranges):
            chosen_idx = torch.arange(len(block_ranges))
        else:
            chosen_idx = torch.randperm(len(block_ranges))[:select_n]
        for idx in chosen_idx.tolist():
            s, e = block_ranges[idx]
            mask[b, s:e] = candidate_mask[b, s:e]
    return mask


def sample_iid_t_mask(candidate_mask, eps):
    bsz, seq_len = candidate_mask.size()
    t = torch.empty(bsz, 1).uniform_(eps, 1.0 - eps)
    mask = (torch.rand(bsz, seq_len) < t) & candidate_mask
    mask = ensure_nonempty_mask(mask, candidate_mask)
    return mask, t.squeeze(-1)


def iid_inv_t_weight(t, eps):
    return 1.0 / torch.clamp(t, min=eps)


def sample_diffusion_time_weight(batch_size):
    eps = args.time_weight_eps
    t = torch.empty(batch_size).uniform_(eps, 1.0 - eps)
    alpha = torch.cos(0.5 * math.pi * t) ** 2
    alpha_prime = -0.5 * math.pi * torch.sin(math.pi * t)
    denom = torch.clamp(1.0 - alpha, min=eps)
    weight = torch.clamp((-alpha_prime / denom), min=eps)
    return weight


def build_doc_ids_from_tokens(x):
    if not args.use_tokenizer:
        return torch.zeros_like(x, dtype=torch.long)
    if not args.use_doc_attention_mask:
        return torch.zeros_like(x, dtype=torch.long)

    doc_ids = torch.full_like(x, -1, dtype=torch.long)
    eos_id = tokenizer.eos_token_id
    for b in range(x.size(0)):
        doc_id = 0
        for t in range(x.size(1)):
            token = int(x[b, t].item())
            if token == pad_token_id:
                continue
            doc_ids[b, t] = doc_id
            if eos_id is not None and token == eos_id:
                doc_id += 1
    return doc_ids


def build_doc_attention_mask(doc_ids):
    if not args.use_doc_attention_mask:
        return None
    valid = doc_ids >= 0
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)
    return (same_doc & valid_pair).unsqueeze(1)


def maybe_apply_variable_length(x, y, prob):
    if prob <= 0.0 or not args.use_tokenizer:
        return x, y
    if pad_token_id is None:
        return x, y

    bsz, seq_len = x.size()
    for b in range(bsz):
        if torch.rand(1).item() >= prob:
            continue
        nonpad = torch.nonzero(x[b] != pad_token_id, as_tuple=False).view(-1)
        valid_len = int(nonpad.numel())
        if valid_len <= 0:
            continue

        # Keep a random prefix length to emulate variable-length pretraining.
        min_len = 1
        if valid_len >= 3:
            min_len = 3  # keep at least one trainable token besides BOS/EOS
        new_len = int(torch.randint(min_len, valid_len + 1, (1,)).item())
        if new_len < valid_len:
            x[b, new_len:] = pad_token_id
            y[b, new_len:] = pad_token_id
    return x, y


def get_batch(split, step=None):
    # generate a small batch of data of inputs x and targets y
    if args.use_tokenizer:
        loader = train_loader if split == "train" else val_loader
        try:
            x = next(loader_iters[split])
        except StopIteration:
            loader_iters[split] = iter(loader)
            x = next(loader_iters[split])
        x = x.clone()
        y = x.clone()  # original tokens
        if split == "train" and args.mask_schedule == "iid_t":
            x, y = maybe_apply_variable_length(x, y, args.variable_length_prob)
        candidate_mask = (
            (x != pad_token_id)
            & (x != tokenizer.bos_token_id)
            & (x != tokenizer.eos_token_id)
        )
    else:
        data = train_data if split == "train" else val_data
        idx = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in idx])
        y = x.clone()  # original tokens
        candidate_mask = torch.ones_like(x, dtype=torch.bool)

    iid_t = None
    if args.mask_schedule == "iid_t":
        block_len = block_size
        mask, iid_t = sample_iid_t_mask(candidate_mask, args.iid_mask_eps)
        seq_lengths = candidate_mask.sum(dim=1).clamp(min=1)
    elif split == "train" and args.mask_schedule == "wsd":
        mask_ratio = get_wsd_mask_ratio(step)
        if args.use_block_curriculum:
            block_len = get_curriculum_block_size(step)
            mask = sample_blockwise_mask(candidate_mask, mask_ratio, block_len)
        else:
            block_len = block_size
            mask = sample_contiguous_block_mask(candidate_mask, mask_ratio)
        mask = ensure_nonempty_mask(mask, candidate_mask)
        seq_lengths = candidate_mask.sum(dim=1).clamp(min=1)
    else:
        block_len = block_size
        seq_lengths = candidate_mask.sum(dim=1).clamp(min=1)
        # Baseline random masking for eval or when explicitly requested.
        bsz, seq_len = x.size()
        mask_probs = torch.rand(bsz, 1)
        mask = (torch.rand(bsz, seq_len) < mask_probs) & candidate_mask
        mask = ensure_nonempty_mask(mask, candidate_mask)
    x[mask] = mask_token_id

    doc_ids = build_doc_ids_from_tokens(y)
    attn_mask = build_doc_attention_mask(doc_ids)
    if args.mask_schedule == "iid_t":
        batch_time_weight = iid_inv_t_weight(iid_t, args.iid_mask_eps)
    elif args.time_weighted_loss:
        batch_time_weight = sample_diffusion_time_weight(x.size(0))
    else:
        batch_time_weight = None

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    doc_ids = doc_ids.to(device)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    if batch_time_weight is not None:
        batch_time_weight = batch_time_weight.to(device)
    if iid_t is not None:
        iid_t = iid_t.to(device)
    seq_lengths = seq_lengths.to(device)
    return x, y, mask, attn_mask, batch_time_weight, block_len, doc_ids, iid_t, seq_lengths


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),), eps=rms_norm_eps)


def get_activation(name):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation: {name}")


def get_lr(step):
    max_lr = learning_rate
    min_lr = learning_rate * min_lr_ratio
    if args.lr_schedule == "cosine":
        if warmup_steps > 0 and step < warmup_steps:
            return max_lr * float(step + 1) / float(warmup_steps)
        if max_iters <= warmup_steps:
            return min_lr
        progress = float(step - warmup_steps) / float(max_iters - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (max_lr - min_lr)

    # Warmup-Stable-Decay schedule:
    # 1) Linear warmup to max_lr
    # 2) Hold max_lr
    # 3) Linear decay to min_lr
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)

    stable_steps = int(max_iters * args.lr_stable_ratio)
    stable_end = warmup_steps + stable_steps
    if step < stable_end:
        return max_lr

    decay_steps = max(max_iters - stable_end, 1)
    decay_idx = min(max(step - stable_end, 0), decay_steps)
    progress = min(max(float(decay_idx) / float(decay_steps), 0.0), 1.0)
    return max_lr - progress * (max_lr - min_lr)


def get_repeat_penalty_weight(step=None):
    base = args.repeat_penalty_weight
    if base <= 0:
        return 0.0
    if step is None:
        return base

    delay = args.repeat_penalty_delay_steps
    warmup = args.repeat_penalty_warmup_steps

    if step < delay:
        return 0.0
    if warmup <= 0:
        return base

    warmup_progress = (step - delay + 1) / warmup
    warmup_progress = min(max(warmup_progress, 0.0), 1.0)
    return base * warmup_progress


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
            # YaRN: f'(i) = f(i)((1-gamma) + gamma/s), gamma is a linear ramp.
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
    assert x.ndim == 4  # multihead attention
    out = (x * cos) + (rotate_half(x) * sin)
    return out.to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin, attn_mask=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (B, T, H, D) -> (B, H, T, D)

        # [NEW]: Set to false for bidirectional instead of causal self-attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        # Re-assemble the heads and project back
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        intermediate_size = ffn_intermediate_size
        if intermediate_size is None:
            intermediate_size = int(n_embd * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=False)
        self.up_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.dropout = nn.Dropout(ffn_dropout)
        self.act_fn = get_activation(ffn_hidden_act)

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x, cos_sin, attn_mask=None):
        x = x + self.attn(norm(x), cos_sin, attn_mask=attn_mask)
        x = x + self.mlp(norm(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # Rotary embeddings
        self.rotary_seq_len = max(max_position_embeddings, block_size * 2)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])

        # Output head to predict denoised tokens
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        cos, sin = precompute_freqs_cis(
            dim=head_dim,
            end=seq_len,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
            device=device,
        )
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _consecutive_repeat_unlikelihood_loss(self, logits, targets, loss_mask):
        if args.repeat_penalty_weight <= 0:
            return logits.new_zeros(())
        _, T, _ = logits.shape
        if T < args.repeat_penalty_min_run:
            return logits.new_zeros(())

        # run_lengths[b, t] = length of consecutive same-token run ending at t.
        run_lengths = torch.ones_like(targets, dtype=torch.long)
        for t in range(1, T):
            same_as_prev = targets[:, t] == targets[:, t - 1]
            run_lengths[:, t] = torch.where(same_as_prev, run_lengths[:, t - 1] + 1, 1)

        repeat_pos = run_lengths >= args.repeat_penalty_min_run
        valid = repeat_pos & loss_mask

        for token_id in repeat_ignore_token_ids:
            valid = valid & (targets != token_id)

        if not valid.any():
            return logits.new_zeros(())

        log_probs = F.log_softmax(logits.float(), dim=-1)
        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        target_probs = target_log_probs.exp()
        target_probs = torch.clamp(target_probs, max=1.0 - 1e-6)
        unlikelihood = -torch.log1p(-target_probs)
        return unlikelihood[valid].mean().to(logits.dtype)

    def forward(
        self,
        idx,
        targets=None,
        mask=None,
        repeat_penalty_weight=None,
        attn_mask=None,
        time_weights=None,
        iid_t=None,
        seq_lengths=None,
    ):
        B, T = idx.size()

        # Get embeddings
        x = self.token_emb(idx)  # (B, T, n_embd)
        x = norm(x)

        # Get rotary embeddings
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, cos_sin, attn_mask=attn_mask)
        x = norm(x)

        # Predict denoised tokens
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # [NEW]: Only compute loss on masked tokens if mask is provided
            if mask is not None:
                ce_loss_per_token = F.cross_entropy(
                    logits.view(B * T, C), targets.view(B * T), reduction="none"
                ).view(B, T)
                if iid_t is not None and seq_lengths is not None:
                    # Eq.3: -(1/(t*L)) * sum_{masked i} log p(x0_i | xt)
                    per_sample_num = (ce_loss_per_token * mask.float()).sum(dim=1)
                    denom = torch.clamp(iid_t * seq_lengths.float(), min=1e-8)
                    ce_loss = (per_sample_num / denom).mean()
                    loss_mask = mask
                else:
                    weighted_mask = mask.float()
                    if time_weights is not None:
                        weighted_mask = weighted_mask * time_weights.view(B, 1)
                    mask_count = float(weighted_mask.sum().item())
                    if mask_count > 0:
                        ce_loss = (ce_loss_per_token * weighted_mask).sum() / mask_count
                        loss_mask = mask
                    else:
                        # Rare fallback for pathological batches with no valid masked positions.
                        ce_loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
                        loss_mask = torch.ones_like(targets, dtype=torch.bool)
                ul_loss = self._consecutive_repeat_unlikelihood_loss(logits, targets, loss_mask)
            else:
                ce_loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
                full_mask = torch.ones_like(targets, dtype=torch.bool)
                ul_loss = self._consecutive_repeat_unlikelihood_loss(
                    logits, targets, full_mask
                )

            if repeat_penalty_weight is None:
                repeat_penalty_weight = args.repeat_penalty_weight
            loss = ce_loss + repeat_penalty_weight * ul_loss

        return logits, loss


# LLaDA-style reverse diffusion decoding:
# at each t->s step, predict all masked tokens, then remask lowest-confidence
# tokens with expected ratio s/t.
@torch.no_grad()
def generate(
    model,
    max_new_tokens,
    prompt_len=16,
    temp=1.0,
    confidence_threshold=0.95,
    top_k=3,
    repeat_penalty=0.0,
    repeat_window=128,
    cap_start_ratio=0.08,
    cap_end_ratio=0.5,
    max_decode_per_step=0,
    gen_steps=64,
):
    effective_prompt_len = min(prompt_len, len(prompt_tokens))
    all_tokens = prompt_tokens[:effective_prompt_len]
    total_steps = 0

    _ = (confidence_threshold, cap_start_ratio, cap_end_ratio, max_decode_per_step)

    # Generate in chunks to keep memory bounded.
    while len(all_tokens) - effective_prompt_len < max_new_tokens:
        block_len = min(240, max_new_tokens - (len(all_tokens) - effective_prompt_len))
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :effective_prompt_len] = torch.tensor(
            all_tokens[-effective_prompt_len:], device=device
        )
        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, effective_prompt_len : effective_prompt_len + block_len] = True

        for step_idx in range(gen_steps):
            if not masked.any():
                break
            total_steps += 1
            logits, _ = model(x)
            logits[..., mask_token_id] = -float("inf")

            if repeat_penalty > 0:
                finalized = x[0][x[0] != mask_token_id]
                if repeat_window > 0:
                    finalized = finalized[-repeat_window:]
                if finalized.numel() > 0:
                    counts = torch.bincount(finalized, minlength=vocab_size).to(logits.dtype)
                    logits = logits - repeat_penalty * counts.view(1, 1, -1)

            k = min(top_k, vocab_size)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
            top_k_probs_norm = top_k_probs / torch.clamp(
                top_k_probs.sum(dim=-1, keepdim=True), min=1e-12
            )
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, k), 1).view(1, block_size)
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_k.unsqueeze(-1)).squeeze(-1)
            sampled_probs = torch.gather(probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

            # Fill all currently masked tokens with current predictions.
            x_filled = torch.where(masked, sampled_tokens, x)
            conf = torch.where(masked, sampled_probs, torch.tensor(float("inf"), device=device))

            if step_idx == gen_steps - 1:
                x = x_filled
                masked = torch.zeros_like(masked)
                break

            # Uniform reverse time discretization: t_k -> s_k.
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

        all_tokens.extend(
            x[0, effective_prompt_len : effective_prompt_len + block_len].tolist()
        )

        if args.use_tokenizer and tokenizer.eos_token_id is not None:
            generated_only = all_tokens[effective_prompt_len:]
            if tokenizer.eos_token_id in generated_only:
                first_eos = generated_only.index(tokenizer.eos_token_id)
                all_tokens = all_tokens[: effective_prompt_len + first_eos + 1]
                break

    tokens_generated = len(all_tokens) - effective_prompt_len
    print(f"Total steps: {total_steps} for {tokens_generated} tokens")
    print(f"Avg decoded per step: {tokens_generated / max(total_steps, 1):.2f}")
    return decode(all_tokens)


@torch.no_grad()
def estimate_loss(current_step=None):
    out = {}
    model.eval()
    repeat_penalty_weight = get_repeat_penalty_weight(current_step)
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M, AM, TW, _, _, IT, SL = get_batch(split, current_step)
            _, loss = model(
                X,
                Y,
                M,
                repeat_penalty_weight=repeat_penalty_weight,
                attn_mask=AM,
                time_weights=TW,
                iid_t=IT,
                seq_lengths=SL,
            )
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    train_flag = args.train
    if args.run_name:
        default_weights_path = os.path.join("weights", f"{args.run_name}.pt")
    else:
        default_weights_path = (
            "weights/diffusion_tokenizer.pt" if args.use_tokenizer else "weights/diffusion.pt"
        )
    weights_path = args.weights_path or default_weights_path
    best_weights_path = (
        os.path.join("weights", f"{args.run_name}_best.pt")
        if args.run_name
        else (
            "weights/diffusion_tokenizer_best.pt"
            if args.use_tokenizer
            else "weights/diffusion_best.pt"
        )
    )
    weights_dir = os.path.dirname(weights_path)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
    best_weights_dir = os.path.dirname(best_weights_path)
    if best_weights_dir:
        os.makedirs(best_weights_dir, exist_ok=True)

    model = Model()
    m = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_train_losses = []
    eval_val_losses = []
    best_val_loss = float("inf")
    best_step = -1
    bad_eval_count = 0
    checkpoint_loaded = False
    pretrained_init_loaded = False

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
    if args.use_tokenizer:
        print(
            f"Corpus: {args.data}, samples: {len(dataset)}, seq_len: {block_size}, vocab: {vocab_size}"
        )
    else:
        print(f"Corpus: {args.data}, chars: {len(text)}, vocab: {vocab_size}")
    print(
        f"LR schedule: {args.lr_schedule}, max_lr={learning_rate:.2e}, "
        f"min_lr={learning_rate * min_lr_ratio:.2e}, warmup_steps={warmup_steps}, "
        f"stable_ratio={args.lr_stable_ratio}, weight_decay={args.weight_decay}"
    )
    print(
        f"Repeat penalty: weight={args.repeat_penalty_weight}, "
        f"min_run={args.repeat_penalty_min_run}, "
        f"delay_steps={args.repeat_penalty_delay_steps}, "
        f"warmup_steps={args.repeat_penalty_warmup_steps}"
    )
    print(
        f"WSD mask schedule: mode={args.mask_schedule}, min_ratio={args.wsd_min_mask_ratio}, "
        f"max_ratio={args.wsd_max_mask_ratio}, warmup_ratio={args.wsd_warmup_ratio}, "
        f"stable_ratio={args.wsd_stable_ratio}, iid_eps={args.iid_mask_eps}"
    )
    if args.use_block_curriculum:
        print(
            f"Block curriculum: phase_ratios={phase_ratios}, "
            f"up={wsd_block_sizes_up}, down={wsd_block_sizes_down}"
        )
    print(
        f"V1: block_curriculum={args.use_block_curriculum}, "
        f"time_weighted_loss={args.time_weighted_loss}, "
        f"doc_attention_mask={args.use_doc_attention_mask and args.use_tokenizer}"
    )
    if args.mask_schedule == "iid_t":
        print("Loss: Eq.3 Monte Carlo masked CE with per-sample inverse-t weighting (1/t).")
    elif args.time_weighted_loss:
        print("Loss: masked CE with diffusion time-weight alpha'(t)/(1-alpha(t)).")
    else:
        print("Loss: masked CE on masked positions (uniform weight).")
    print(
        f"Generation (LLaDA-style remask): temp={args.gen_temp}, conf_thr={args.gen_confidence_threshold}, "
        f"top_k={args.gen_top_k}, rep_penalty={args.gen_repeat_penalty}, "
        f"rep_window={args.gen_repeat_window}, gen_steps={args.gen_steps}, "
        f"cap_start={args.gen_cap_start_ratio}, cap_end={args.gen_cap_end_ratio}, "
        f"max_decode_per_step={args.gen_max_decode_per_step}"
    )
    print(
        f"Validation checkpointing: best_path={best_weights_path}, "
        f"early_stop_patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}"
    )

    if os.path.exists(weights_path):
        print(f"Loading checkpoint from {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            m.load_state_dict(ckpt["model_state_dict"])
            checkpoint_loaded = True
            if train_flag and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = int(ckpt.get("step", -1)) + 1 if train_flag else 0
            train_steps = ckpt.get("train_steps", train_steps)
            train_losses = ckpt.get("train_losses", train_losses)
            eval_steps = ckpt.get("eval_steps", eval_steps)
            eval_train_losses = ckpt.get("eval_train_losses", eval_train_losses)
            eval_val_losses = ckpt.get("eval_val_losses", eval_val_losses)
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
            best_step = int(ckpt.get("best_step", best_step))
            bad_eval_count = int(ckpt.get("bad_eval_count", 0))
            if train_flag:
                print(f"Resuming training from step {start_step}")
        else:
            # Backward compatibility: old checkpoints saved as raw state_dict.
            m.load_state_dict(ckpt)
            checkpoint_loaded = True
            if train_flag:
                print("Loaded model weights (no optimizer/step state, resume from step 0)")
    elif train_flag and args.init_from_minimind:
        if not os.path.exists(args.init_from_minimind):
            raise FileNotFoundError(
                f"--init-from-minimind path not found: {args.init_from_minimind}"
            )
        print(f"Initializing diffusion weights from MiniMind: {args.init_from_minimind}")
        pretrain_ckpt = torch.load(args.init_from_minimind, map_location=device)
        pretrain_sd = extract_checkpoint_state_dict(pretrain_ckpt)
        if not isinstance(pretrain_sd, dict):
            raise ValueError(
                f"Unsupported checkpoint format at {args.init_from_minimind}"
            )
        pretrain_sd = normalize_transfer_state_dict_keys(pretrain_sd)
        (
            missing,
            unexpected,
            skipped_shape,
            skipped_missing,
            matched_count,
        ) = load_matching_state_dict(m, pretrain_sd)
        pretrained_init_loaded = True
        print(
            f"Loaded {matched_count} parameter tensors from MiniMind. "
            f"model_missing={len(missing)}, model_unexpected={len(unexpected)}, "
            f"skipped_shape={len(skipped_shape)}, skipped_missing={len(skipped_missing)}"
        )
        if matched_count == 0:
            raise ValueError(
                "No compatible parameter tensors were loaded from --init-from-minimind. "
                "Please check that model hyperparameters and tokenizer vocab size match."
            )
        if skipped_shape:
            ex = skipped_shape[0]
            print(
                "Example shape mismatch: "
                f"{ex[0]} pretrain{ex[1]} vs diffusion{ex[2]}"
            )
    elif not train_flag:
        raise FileNotFoundError(
            f"No checkpoint found at {weights_path}. Use --train to train from scratch."
        )

    if train_flag:
        if not checkpoint_loaded:
            if pretrained_init_loaded:
                print("Training starts from MiniMind-initialized weights")
            else:
                print("Training from scratch")
        elif start_step >= max_iters:
            print(
                f"Checkpoint already at/after max_iters ({start_step} >= {max_iters}). Skipping training."
            )

        start = time.time()
        total_train_steps = max(max_iters - start_step, 0)
        pbar = (
            tqdm(total=total_train_steps, desc="Training", dynamic_ncols=True)
            if tqdm and total_train_steps > 0
            else None
        )
        last_step = start_step - 1
        for step in range(start_step, max_iters):
            last_step = step
            current_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # every once in a while evaluate the loss on train and val sets
            if step % eval_interval == 0 or step == max_iters - 1:
                active_repeat_penalty = get_repeat_penalty_weight(step)
                active_mask_ratio = (
                    get_wsd_mask_ratio(step) if args.mask_schedule == "wsd" else float("nan")
                )
                phase, _ = get_curriculum_phase(step)
                active_block = get_curriculum_block_size(step)
                losses = estimate_loss(step)
                eval_steps.append(step)
                eval_train_losses.append(losses["train"].item())
                eval_val_losses.append(losses["val"].item())
                current_val = losses["val"].item()
                improved = current_val < (best_val_loss - args.early_stop_min_delta)
                if improved:
                    best_val_loss = current_val
                    best_step = step
                    bad_eval_count = 0
                    torch.save(
                        {
                            "model_state_dict": m.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": step,
                            "best_val_loss": best_val_loss,
                            "best_step": best_step,
                            "args": vars(args),
                        },
                        best_weights_path,
                    )
                    print(
                        f"New best val loss {best_val_loss:.4f} at step {best_step}. "
                        f"Saved to {best_weights_path}"
                    )
                else:
                    bad_eval_count += 1
                print(
                    f"step {step}: train loss {losses['train']:.4f},"
                    f"val loss {losses['val']:.4f}, lr {current_lr:.2e}, "
                    f"repeat_w {active_repeat_penalty:.4f}, "
                    f"mask_r {active_mask_ratio:.3f}, "
                    f"phase {phase}, block {active_block}, "
                    f"best_val {best_val_loss:.4f}@{best_step}, "
                    f"time {time.time() - start:.2f} seconds"
                )
                # Generate a sample
                sample = generate(
                    m,
                    max_new_tokens=240,
                    temp=args.gen_temp,
                    confidence_threshold=args.gen_confidence_threshold,
                    top_k=args.gen_top_k,
                    repeat_penalty=args.gen_repeat_penalty,
                    repeat_window=args.gen_repeat_window,
                    cap_start_ratio=args.gen_cap_start_ratio,
                    cap_end_ratio=args.gen_cap_end_ratio,
                    max_decode_per_step=args.gen_max_decode_per_step,
                    gen_steps=args.gen_steps,
                )
                print(f"Sample:\n{sample}\n")
                if args.early_stop_patience > 0 and bad_eval_count >= args.early_stop_patience:
                    print(
                        f"Early stopping at step {step}: no val improvement for "
                        f"{bad_eval_count} eval intervals."
                    )
                    break

            # sample a batch of data
            xb, yb, mb, am, tw, current_block_len, _, it, sl = get_batch("train", step)

            # evaluate the loss
            active_repeat_penalty = get_repeat_penalty_weight(step)
            logits, loss = model(
                xb,
                yb,
                mb,
                repeat_penalty_weight=active_repeat_penalty,
                attn_mask=am,
                time_weights=tw,
                iid_t=it,
                seq_lengths=sl,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_steps.append(step)
            train_losses.append(loss.item())

            if pbar is not None:
                pbar.update(1)
                if step % 10 == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{current_lr:.2e}",
                        blk=current_block_len,
                    )
            elif step % 100 == 0:
                elapsed = time.time() - start
                done = step - start_step + 1
                speed = done / max(elapsed, 1e-6)
                eta = (max_iters - step - 1) / max(speed, 1e-6)
                print(
                    f"progress {step + 1}/{max_iters}, "
                    f"loss {loss.item():.4f}, lr {current_lr:.2e}, eta {eta:.1f}s"
                )

        if pbar is not None:
            pbar.close()

        # Save the model weights
        print(f"Total training time: {time.time() - start:.2f} seconds")
        print(f"Saving checkpoint to {weights_path}")
        if best_step >= 0:
            print(f"Best val checkpoint: step {best_step}, val {best_val_loss:.4f}, path {best_weights_path}")
        torch.save(
            {
                "model_state_dict": m.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": last_step,
                "train_steps": train_steps,
                "train_losses": train_losses,
                "eval_steps": eval_steps,
                "eval_train_losses": eval_train_losses,
                "eval_val_losses": eval_val_losses,
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "bad_eval_count": bad_eval_count,
                "args": vars(args),
            },
            weights_path,
        )

        # Save loss curve
        if args.run_name:
            plot_path = os.path.join("weights", f"{args.run_name}_loss.png")
        else:
            plot_path = (
                "weights/diffusion_tokenizer_loss.png"
                if args.use_tokenizer
                else "weights/diffusion_loss.png"
            )
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_losses, label="train (per step)", alpha=0.35)
        plt.plot(eval_steps, eval_train_losses, label="train (eval avg)", linewidth=2)
        plt.plot(eval_steps, eval_val_losses, label="val (eval avg)", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved loss plot to {plot_path}")

    # generate from the model
    start = time.time()
    output = generate(
        m,
        max_new_tokens=2000,
        temp=args.gen_temp,
        confidence_threshold=args.gen_confidence_threshold,
        top_k=args.gen_top_k,
        repeat_penalty=args.gen_repeat_penalty,
        repeat_window=args.gen_repeat_window,
        cap_start_ratio=args.gen_cap_start_ratio,
        cap_end_ratio=args.gen_cap_end_ratio,
        max_decode_per_step=args.gen_max_decode_per_step,
        gen_steps=args.gen_steps,
    )
    print(f"Total generation time: {time.time() - start:.2f} seconds")
    print(f"\nOutput:\n{output}")

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def _render_chat_messages(messages: Sequence[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for m in messages:
        role = str(m.get("role", "")).strip()
        if not role:
            continue
        content = m.get("content", "")
        content = content if isinstance(content, str) else str(content)
        chunks.append(f"<|im_start|>{role}\\n{content}<|im_end|>\\n")
    return "".join(chunks)


def _as_text(value) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, dict):
        if "content" in value:
            return _as_text(value["content"])
        return str(value)
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(_as_text(item.get("text", "")))
            else:
                parts.append(_as_text(item))
        return "".join(parts)
    return str(value)


def _normalize_messages(raw) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        if not role:
            continue
        content = _as_text(m.get("content", ""))
        out.append({"role": role, "content": content})
    return out


def _extract_from_paired_conversations(row: Dict) -> Tuple[str, str, str]:
    chosen = _normalize_messages(row.get("chosen"))
    rejected = _normalize_messages(row.get("rejected"))
    if not chosen or not rejected:
        return "", "", ""
    if chosen[-1].get("role") != "assistant" or rejected[-1].get("role") != "assistant":
        return "", "", ""

    prompt_msgs = []
    if len(chosen) >= 2 and len(rejected) >= 2 and chosen[:-1] == rejected[:-1]:
        prompt_msgs = chosen[:-1]
    else:
        common = 0
        max_common = min(len(chosen), len(rejected)) - 1
        while common < max_common and chosen[common] == rejected[common]:
            common += 1
        prompt_msgs = chosen[:common]

    if not prompt_msgs:
        return "", "", ""

    prompt_text = _render_chat_messages(prompt_msgs) + "<|im_start|>assistant\\n"
    chosen_resp = _as_text(chosen[-1].get("content", ""))
    rejected_resp = _as_text(rejected[-1].get("content", ""))
    if not chosen_resp or not rejected_resp:
        return "", "", ""
    return prompt_text, chosen_resp + "<|im_end|>", rejected_resp + "<|im_end|>"


def _extract_from_prompt_style(row: Dict) -> Tuple[str, str, str]:
    chosen = _as_text(row.get("chosen", ""))
    rejected = _as_text(row.get("rejected", ""))
    if not chosen or not rejected:
        return "", "", ""

    prompt = (
        row.get("prompt")
        or row.get("instruction")
        or row.get("question")
        or row.get("input")
        or row.get("context")
        or ""
    )
    prompt = _as_text(prompt)
    if not prompt:
        return "", "", ""

    system = _as_text(row.get("system", ""))
    prompt_msgs: List[Dict[str, str]] = []
    if system:
        prompt_msgs.append({"role": "system", "content": system})
    prompt_msgs.append({"role": "user", "content": prompt})
    prompt_text = _render_chat_messages(prompt_msgs) + "<|im_start|>assistant\\n"
    return prompt_text, chosen + "<|im_end|>", rejected + "<|im_end|>"


def _extract_from_messages_plus_pair(row: Dict) -> Tuple[str, str, str]:
    messages = _normalize_messages(row.get("messages") or row.get("conversations"))
    chosen = _as_text(row.get("chosen", ""))
    rejected = _as_text(row.get("rejected", ""))
    if not messages or not chosen or not rejected:
        return "", "", ""

    prompt_text = _render_chat_messages(messages) + "<|im_start|>assistant\\n"
    return prompt_text, chosen + "<|im_end|>", rejected + "<|im_end|>"


def _extract_preference_triplet(row: Dict) -> Tuple[str, str, str]:
    prompt_text, chosen_text, rejected_text = _extract_from_paired_conversations(row)
    if prompt_text:
        return prompt_text, chosen_text, rejected_text

    prompt_text, chosen_text, rejected_text = _extract_from_messages_plus_pair(row)
    if prompt_text:
        return prompt_text, chosen_text, rejected_text

    return _extract_from_prompt_style(row)


class PreferenceConversationDataset(Dataset):
    """
    Preference-pair dataset for diffusion DPO.

    Expected per-row jsonl formats include:
    1) {"prompt": ..., "chosen": ..., "rejected": ...}
    2) {"system": ..., "instruction": ..., "chosen": ..., "rejected": ...}
    3) {"messages"|"conversations": [...], "chosen": ..., "rejected": ...}
    4) {"chosen": [...], "rejected": [...]}  # paired conversations ending in assistant
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        self.path = Path(str(data_path))
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for dynamic EOS padding.")
        if self.path.suffix != ".jsonl":
            raise ValueError("PreferenceConversationDataset expects a .jsonl file.")

        self.offsets = self._build_offsets()
        if not self.offsets:
            raise ValueError(f"No valid preference pairs found in {self.path}")

    def _build_offsets(self) -> List[int]:
        offsets: List[int] = []
        with self.path.open("r", encoding="utf-8") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                row = line.strip()
                if not row:
                    continue
                obj = json.loads(row)
                prompt_text, chosen_text, rejected_text = _extract_preference_triplet(obj)
                if prompt_text and chosen_text and rejected_text:
                    offsets.append(pos)
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_row(self, idx: int) -> Dict:
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            return json.loads(f.readline())

    def _truncate_pair(
        self,
        prompt_ids: List[int],
        chosen_ids: List[int],
        rejected_ids: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        bos_budget = 1 if self.bos_token_id is not None else 0
        max_body_len = self.max_length - bos_budget - 1
        if max_body_len <= 1:
            raise ValueError(f"max_length={self.max_length} is too small for DPO.")

        max_prompt_len = max_body_len - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        remain = max_body_len - len(prompt_ids)
        chosen_ids = chosen_ids[:remain]
        rejected_ids = rejected_ids[:remain]
        if not chosen_ids:
            chosen_ids = [self.eos_token_id]
        if not rejected_ids:
            rejected_ids = [self.eos_token_id]
        return prompt_ids, chosen_ids, rejected_ids

    def __getitem__(self, idx: int) -> Dict:
        row = self._read_row(idx)
        prompt_text, chosen_text, rejected_text = _extract_preference_triplet(row)
        if not prompt_text:
            raise ValueError(f"Invalid preference row at index {idx}")

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        chosen_ids = self.tokenizer(chosen_text, add_special_tokens=False).input_ids
        rejected_ids = self.tokenizer(rejected_text, add_special_tokens=False).input_ids

        prompt_ids, chosen_ids, rejected_ids = self._truncate_pair(prompt_ids, chosen_ids, rejected_ids)

        chosen_seq: List[int] = []
        rejected_seq: List[int] = []
        if self.bos_token_id is not None:
            chosen_seq.append(self.bos_token_id)
            rejected_seq.append(self.bos_token_id)

        chosen_seq.extend(prompt_ids)
        rejected_seq.extend(prompt_ids)
        prompt_len = len(chosen_seq)

        chosen_seq.extend(chosen_ids)
        rejected_seq.extend(rejected_ids)
        chosen_seq.append(self.eos_token_id)
        rejected_seq.append(self.eos_token_id)

        return {
            "chosen_input_ids": torch.tensor(chosen_seq, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_seq, dtype=torch.long),
            "prompt_len": prompt_len,
        }


def collate_preference_diffusion(
    batch: Sequence[Dict],
    eos_token_id: int,
    quantize_block_size: int = 0,
):
    max_len = 0
    for sample in batch:
        max_len = max(max_len, int(sample["chosen_input_ids"].numel()), int(sample["rejected_input_ids"].numel()))
    if quantize_block_size and quantize_block_size > 0:
        max_len = ((max_len + quantize_block_size - 1) // quantize_block_size) * quantize_block_size

    chosen_input_ids = []
    rejected_input_ids = []
    prompt_lengths = []
    chosen_answer_lengths = []
    rejected_answer_lengths = []
    chosen_effective_seq_lengths = []
    rejected_effective_seq_lengths = []

    for sample in batch:
        prompt_len = int(sample["prompt_len"])

        chosen_seq = sample["chosen_input_ids"].tolist()
        rejected_seq = sample["rejected_input_ids"].tolist()
        chosen_len = len(chosen_seq)
        rejected_len = len(rejected_seq)

        chosen_eff_len = chosen_len
        rejected_eff_len = rejected_len
        if quantize_block_size and quantize_block_size > 0:
            chosen_eff_len = ((chosen_len + quantize_block_size - 1) // quantize_block_size) * quantize_block_size
            rejected_eff_len = ((rejected_len + quantize_block_size - 1) // quantize_block_size) * quantize_block_size
            chosen_eff_len = min(chosen_eff_len, max_len)
            rejected_eff_len = min(rejected_eff_len, max_len)

        chosen_seq = chosen_seq + [eos_token_id] * (max_len - chosen_len)
        rejected_seq = rejected_seq + [eos_token_id] * (max_len - rejected_len)

        chosen_input_ids.append(chosen_seq)
        rejected_input_ids.append(rejected_seq)
        prompt_lengths.append(prompt_len)
        chosen_answer_lengths.append(max(chosen_len - prompt_len, 1))
        rejected_answer_lengths.append(max(rejected_len - prompt_len, 1))
        chosen_effective_seq_lengths.append(chosen_eff_len)
        rejected_effective_seq_lengths.append(rejected_eff_len)

    return {
        "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
        "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "chosen_answer_lengths": torch.tensor(chosen_answer_lengths, dtype=torch.long),
        "rejected_answer_lengths": torch.tensor(rejected_answer_lengths, dtype=torch.long),
        "chosen_effective_seq_lengths": torch.tensor(chosen_effective_seq_lengths, dtype=torch.long),
        "rejected_effective_seq_lengths": torch.tensor(rejected_effective_seq_lengths, dtype=torch.long),
    }

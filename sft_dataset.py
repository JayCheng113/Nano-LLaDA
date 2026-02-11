import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

OFFSET_CACHE_VERSION = 1


def _render_chat_messages(messages: Sequence[Dict[str, str]]) -> str:
    chunks = []
    for m in messages:
        role = str(m.get("role", "")).strip()
        if not role:
            continue
        content = m.get("content", "")
        content = content if isinstance(content, str) else str(content)
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(chunks)


def _build_turn_pairs_from_messages(messages: Sequence[Dict[str, str]]) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []
    for idx, m in enumerate(messages):
        role = str(m.get("role", "")).strip()
        if role != "assistant":
            continue
        content = m.get("content", "")
        content = content if isinstance(content, str) else str(content)
        if not content:
            continue
        context = messages[:idx]
        if not context:
            continue
        prompt_text = _render_chat_messages(context) + "<|im_start|>assistant\n"
        response_text = content + "<|im_end|>"
        turns.append((prompt_text, response_text))
    return turns


def _build_turn_pairs_from_flat_row(row: Dict) -> List[Tuple[str, str]]:
    prompt = (
        row.get("prompt")
        or row.get("instruction")
        or row.get("question")
        or row.get("input")
        or ""
    )
    response = row.get("response") or row.get("output") or row.get("answer") or row.get("target") or ""
    if not isinstance(prompt, str):
        prompt = str(prompt)
    if not isinstance(response, str):
        response = str(response)
    if not prompt or not response:
        return []

    system_text = row.get("system")
    messages = []
    if system_text:
        system_text = system_text if isinstance(system_text, str) else str(system_text)
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt})
    prompt_text = _render_chat_messages(messages) + "<|im_start|>assistant\n"
    response_text = response + "<|im_end|>"
    return [(prompt_text, response_text)]


def _extract_turn_pairs(row: Dict) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if isinstance(row.get("conversations"), list):
        pairs = _build_turn_pairs_from_messages(row["conversations"])
    elif isinstance(row.get("messages"), list):
        # Backward compatibility for older data files.
        pairs = _build_turn_pairs_from_messages(row["messages"])
    if not pairs:
        pairs = _build_turn_pairs_from_flat_row(row)
    return pairs


class SFTConversationDataset(Dataset):
    """
    Multi-turn dialogue dataset for SFT.

    For each dialogue with n assistant replies, we store n candidate (prompt, response) pairs:
    (p0, r0), (p0 r0 p1, r1), ...
    During training, __getitem__ randomly selects one candidate turn.
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

        self.offsets = self._build_offsets()
        if not self.offsets:
            raise ValueError(f"No valid SFT samples found in {self.path}")

    def _offset_cache_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".sft_offsets.pt")

    def _load_offsets_cache(self):
        cache_path = self._offset_cache_path()
        if not cache_path.exists():
            return None
        try:
            payload = torch.load(cache_path, map_location="cpu")
            if not isinstance(payload, dict):
                return None
            if payload.get("version") != OFFSET_CACHE_VERSION:
                return None
            stat = self.path.stat()
            if payload.get("source_size") != stat.st_size:
                return None
            if payload.get("source_mtime_ns") != stat.st_mtime_ns:
                return None
            offsets = payload.get("offsets")
            if not isinstance(offsets, list) or not offsets:
                return None
            if not all(isinstance(x, int) and x >= 0 for x in offsets):
                return None
            return offsets
        except Exception:
            return None

    def _save_offsets_cache(self, offsets: List[int]):
        if not offsets:
            return
        cache_path = self._offset_cache_path()
        stat = self.path.stat()
        payload = {
            "version": OFFSET_CACHE_VERSION,
            "source_path": str(self.path),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "offsets": offsets,
        }
        try:
            torch.save(payload, cache_path)
        except Exception:
            pass

    def _build_offsets(self):
        is_jsonl = self.path.suffix == ".jsonl"
        if not is_jsonl:
            raise ValueError("SFT dataset currently expects .jsonl input.")

        cached_offsets = self._load_offsets_cache()
        if cached_offsets is not None:
            return cached_offsets

        offsets = []
        with self.path.open("r", encoding="utf-8") as f:
            pbar = tqdm(desc="Loading SFT dataset", unit="lines", dynamic_ncols=True) if tqdm is not None else None
            while True:
                cur_pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if pbar is not None:
                    pbar.update(1)
                row = line.strip()
                if not row:
                    continue
                obj = json.loads(row)
                if _extract_turn_pairs(obj):
                    offsets.append(cur_pos)
            if pbar is not None:
                pbar.close()
        self._save_offsets_cache(offsets)
        return offsets

    def __len__(self):
        return len(self.offsets)

    def _read_row(self, idx: int) -> Dict:
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        return json.loads(line)

    def _truncate_pair(self, prompt_ids: List[int], response_ids: List[int]) -> Tuple[List[int], List[int]]:
        # Keep at least one response token.
        bos_budget = 1 if self.bos_token_id is not None else 0
        max_body_len = self.max_length - bos_budget - 1  # Reserve final EOS.
        if max_body_len <= 1:
            raise ValueError(f"max_length={self.max_length} is too small for SFT.")

        max_prompt_len = max_body_len - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        remain = max_body_len - len(prompt_ids)
        response_ids = response_ids[:remain]
        if not response_ids:
            response_ids = [self.eos_token_id]
        return prompt_ids, response_ids

    def __getitem__(self, idx):
        row = self._read_row(idx)
        turns = _extract_turn_pairs(row)
        if not turns:
            raise ValueError(f"Invalid SFT row at index {idx}: no usable turns")
        turn_idx = int(torch.randint(len(turns), (1,)).item())
        prompt_text, response_text = turns[turn_idx]
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(response_text, add_special_tokens=False).input_ids
        prompt_ids, response_ids = self._truncate_pair(prompt_ids, response_ids)

        seq: List[int] = []
        if self.bos_token_id is not None:
            seq.append(self.bos_token_id)
        seq.extend(prompt_ids)
        prompt_len = len(seq)
        seq.extend(response_ids)
        seq.append(self.eos_token_id)

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "prompt_len": prompt_len,
        }


class StreamingSFTConversationDataset(IterableDataset):
    """Line-by-line streaming dataset for SFT: tokenize while training."""

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
            raise ValueError("Streaming SFT dataset currently expects .jsonl input.")

    def _truncate_pair(self, prompt_ids: List[int], response_ids: List[int]) -> Tuple[List[int], List[int]]:
        bos_budget = 1 if self.bos_token_id is not None else 0
        max_body_len = self.max_length - bos_budget - 1
        if max_body_len <= 1:
            raise ValueError(f"max_length={self.max_length} is too small for SFT.")

        max_prompt_len = max_body_len - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        remain = max_body_len - len(prompt_ids)
        response_ids = response_ids[:remain]
        if not response_ids:
            response_ids = [self.eos_token_id]
        return prompt_ids, response_ids

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        with self.path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if (line_idx % num_workers) != worker_id:
                    continue
                row = line.strip()
                if not row:
                    continue
                obj = json.loads(row)
                turns = _extract_turn_pairs(obj)
                if not turns:
                    continue

                turn_idx = int(torch.randint(len(turns), (1,)).item())
                prompt_text, response_text = turns[turn_idx]
                prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
                response_ids = self.tokenizer(response_text, add_special_tokens=False).input_ids
                if not prompt_ids or not response_ids:
                    continue
                prompt_ids, response_ids = self._truncate_pair(prompt_ids, response_ids)

                seq: List[int] = []
                if self.bos_token_id is not None:
                    seq.append(self.bos_token_id)
                seq.extend(prompt_ids)
                prompt_len = len(seq)
                seq.extend(response_ids)
                seq.append(self.eos_token_id)

                yield {
                    "input_ids": torch.tensor(seq, dtype=torch.long),
                    "prompt_len": prompt_len,
                }


def collate_sft_ar(batch: Sequence[Dict], eos_token_id: int):
    max_len = max(int(sample["input_ids"].numel()) for sample in batch)
    input_ids = []
    labels = []
    prompt_lengths = []
    for sample in batch:
        seq = sample["input_ids"].tolist()
        prompt_len = int(sample["prompt_len"])
        pad_n = max_len - len(seq)
        seq = seq + [eos_token_id] * pad_n

        label = [-100] * prompt_len + seq[prompt_len:]
        input_ids.append(seq)
        labels.append(label)
        prompt_lengths.append(prompt_len)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long)
    return input_ids, labels, attention_mask, prompt_lengths


def collate_sft_diffusion(batch: Sequence[Dict], eos_token_id: int):
    max_len = max(int(sample["input_ids"].numel()) for sample in batch)
    input_ids = []
    prompt_lengths = []
    answer_lengths = []
    for sample in batch:
        seq = sample["input_ids"].tolist()
        prompt_len = int(sample["prompt_len"])
        pad_n = max_len - len(seq)
        seq = seq + [eos_token_id] * pad_n

        input_ids.append(seq)
        prompt_lengths.append(prompt_len)
        answer_lengths.append(max_len - prompt_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "answer_lengths": torch.tensor(answer_lengths, dtype=torch.long),
    }

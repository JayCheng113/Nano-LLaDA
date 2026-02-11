import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        vocab_size: int = 6400,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1e6,
        max_position_embeddings: int = 32768,
        inference_rope_scaling: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        attention_is_causal: bool = True,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.inference_rope_scaling = inference_rope_scaling
        self.attention_is_causal = attention_is_causal
        self.rope_scaling = (
            {
                "original_max_position_embeddings": 2048,
                "factor": 16,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if inference_rope_scaling
            else None
        )


def norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def get_activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation: {name}")


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
    assert x.ndim == 4
    out = (x * cos) + (rotate_half(x) * sin)
    return out.to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_embd = config.hidden_size
        self.n_head = config.num_attention_heads
        if self.n_embd % self.n_head != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.head_dim = self.n_embd // self.n_head
        self.dropout = config.dropout
        self.rms_norm_eps = config.rms_norm_eps
        self.is_causal = config.attention_is_causal

        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, attention_mask=None):
        bsz, seq_len, _ = x.size()

        q = self.c_q(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seq_len, self.n_head, self.head_dim)
        v = self.c_v(x).view(bsz, seq_len, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = norm(q, self.rms_norm_eps)
        k = norm(k, self.rms_norm_eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attention_mask is not None:
            # [B, T] -> [B, 1, T, T]
            am = attention_mask[:, None, None, :].to(torch.bool)
            am = am.expand(-1, 1, seq_len, -1)
        else:
            am = None

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=am,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        intermediate_size = config.intermediate_size
        if intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.rms_norm_eps = config.rms_norm_eps

    def forward(self, x, cos_sin, attention_mask=None):
        x = x + self.attn(norm(x, self.rms_norm_eps), cos_sin, attention_mask=attention_mask)
        x = x + self.mlp(norm(x, self.rms_norm_eps))
        return x


class MiniMindForCausalLM(PreTrainedModel):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig):
        super().__init__(config)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rms_norm_eps = config.rms_norm_eps

        # Names intentionally aligned with diffusion.Model for weight reuse.
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rotary_seq_len = max(config.max_position_embeddings, 512)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, device=None):
        if device is None:
            device = self.token_emb.weight.device
        cos, sin = precompute_freqs_cis(
            dim=self.head_dim,
            end=seq_len,
            rope_base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling,
            device=device,
        )
        return cos[None, :, None, :], sin[None, :, None, :]

    def _ensure_rope_length(self, seq_len: int, device: torch.device):
        if seq_len <= self.cos.size(1) and self.cos.device == device:
            return
        self.rotary_seq_len = max(seq_len, self.rotary_seq_len)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, device=device)
        self.cos = cos
        self.sin = sin

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        _ = kwargs
        bsz, seq_len = input_ids.size()
        self._ensure_rope_length(seq_len, input_ids.device)

        x = self.token_emb(input_ids)
        x = norm(x, self.rms_norm_eps)

        cos_sin = (self.cos[:, :seq_len], self.sin[:, :seq_len])
        for block in self.blocks:
            x = block(x, cos_sin, attention_mask=attention_mask)
        x = norm(x, self.rms_norm_eps)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

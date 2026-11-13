from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    d_ff: int = 4096
    context_length: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "context_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads != 0 or self.head_dim % 2 != 0:
            raise ValueError("d_model must be divisible by n_heads and head_dim must be even")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta and norm_eps must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate the mean square in float32, including during mixed precision.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [batch, heads, length, head_dim]. Rotate adjacent coordinate pairs.
        positions = torch.arange(q.size(-2), device=q.device, dtype=torch.float32)
        dimensions = torch.arange(0, self.head_dim, 2, device=q.device, dtype=torch.float32)
        frequencies = self.theta ** (-dimensions / self.head_dim)
        angles = positions[:, None] * frequencies[None, :]
        cos = angles.cos().to(dtype=q.dtype)
        sin = angles.sin().to(dtype=q.dtype)

        def rotate(x):
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v)]
        q, k = self.rope(q, k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # True above the diagonal: position i must not attend to positions j > i.
        if attention_mask is None:
            attention_mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(attention_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=v.dtype)
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])
        self.final_norm = RMSNorm(self.config.d_model, self.config.norm_eps)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Each block adds two residual branches; reduce their initial output scale.
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=residual_std)
            nn.init.normal_(block.ffn.down_proj.weight, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, document_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits [batch, length, vocab_size]; position t predicts token t+1."""
        if input_ids.ndim != 2 or not 1 <= input_ids.size(1) <= self.config.context_length:
            raise ValueError(f"input_ids must have shape [batch, length] with 1 <= length <= {self.config.context_length}")
        length = input_ids.size(1)
        attention_mask = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).triu(1)
        if document_ids is not None:
            if document_ids.shape != input_ids.shape:
                raise ValueError("document_ids must have the same shape as input_ids")
            # A position can attend only to earlier positions in the same document.
            different_document = document_ids[:, None, :, None] != document_ids[:, None, None, :]
            attention_mask = attention_mask[None, None] | different_document
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.lm_head(self.final_norm(x))

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """Return prompt plus sampled tokens; temperature=0 selects greedy decoding."""
        if input_ids.ndim != 2 or input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("input_ids must be a nonempty [batch, length] tensor")
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens and temperature must be nonnegative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id must be inside the vocabulary")
        if pad_token_id is not None and not 0 <= pad_token_id < self.config.vocab_size:
            raise ValueError("pad_token_id must be inside the vocabulary")
        if eos_token_id is not None and (pad_token_id is None or pad_token_id == eos_token_id):
            raise ValueError("EOS stopping requires a separate pad_token_id")

        was_training = self.training
        self.eval()
        try:
            finished = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.bool)
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.context_length:]
                logits = self(context)[:, -1, :].float()
                if pad_token_id is not None:
                    logits[:, pad_token_id] = float("-inf")
                if temperature == 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                        logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, values)
                    next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                if eos_token_id is not None:
                    # Preserve the first EOS; pad already finished rows to keep the batch rectangular.
                    next_token = next_token.masked_fill(finished[:, None], pad_token_id)
                    finished = finished | (next_token[:, 0] == eos_token_id)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                if eos_token_id is not None and finished.all():
                    break
            return input_ids
        finally:
            self.train(was_training)

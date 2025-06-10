"""
GPT-2 inference using Magnetron. (WORK IN PROGRESS NOT YET DONE)
This file will read (or download) the pretrained weights, which are stored in Magnetron's custom file format.
Then interactive inference is performed in the command line window.

References:
1) Andrej Karpathy's llm.c: https://github.com/karpathy/llm.c
2) The official GPT-2 TensorFlow implementation released by OpenAI: https://github.com/openai/gpt-2/blob/master/src/model.py
3) Huggingface/transformers PyTorch implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass

import magnetron as mag
import magnetron.nn as nn


@dataclass
class GPTConfig:
    use_flash_attn: bool = False  # Use flash attention
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        return 0.5 * x * (1.0 + (math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)).tanh())


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        self.cfg = cfg
        assert cfg.n_embed % cfg.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(cfg.n_embed, 3 * cfg.n_embed)
        # output projection
        self.c_proj = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.c_proj.RESIDUAL_SCALE_FLAG = True
        # regularization
        self.n_head = cfg.n_head
        self.n_embed = cfg.n_embed
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            'bias',
            torch.tril(mag.Tensor.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embed)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        if self.cfg.use_flash_attn:
            # flashattention
            # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            raise NotImplementedError()
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = att.softmax(dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        self.c_fc = nn.Linear(cfg.n_embed, 4 * cfg.n_embed)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * cfg.n_embed, cfg.n_embed)
        self.c_proj.RESIDUAL_SCALE_FLAG = True

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        self.ln_1 = nn.LayerNorm(cfg.n_embed)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = MLP(cfg)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.lm_head.SKIP_INIT = True
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'RESIDUAL_SCALE_FLAG') else 0.02 / math.sqrt(2 * self.cfg.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'SKIP_INIT'):
                module.weight.fill_random_normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.zeros_()
        elif isinstance(module, nn.Embedding):
            module.weight.fill_random_normal_(mean=0.0, std=0.02)

    @mag.no_grad()
    def generate(
        self,
        indices: mag.Tensor,
        max_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 40,
    ) -> mag.Tensor:
        pass

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        pass


if __name__ == '__main__':
    config = GPTConfig()
    gpt = GPT(config)

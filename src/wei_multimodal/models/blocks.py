from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from wei_multimodal.data.dataset import MultimodalBatch


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(features))


class ClinicalEncoder(nn.Module):
    def __init__(
        self,
        type_vocab_size: int,
        t_stage_vocab_size: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if type_vocab_size < 1 or t_stage_vocab_size < 1:
            raise ValueError("clinical vocabulary sizes must include unknown index 0")
        self.type_embedding = nn.Embedding(type_vocab_size, 4)
        self.t_stage_embedding = nn.Embedding(t_stage_vocab_size, 4)
        self.continuous_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.LayerNorm(16),
        )
        self.projection = nn.Sequential(
            nn.Linear(24, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, batch: MultimodalBatch) -> torch.Tensor:
        continuous = self.continuous_encoder(torch.cat([batch.age, batch.male], dim=1))
        type_embedding = self.type_embedding(batch.type_index)
        t_stage_embedding = self.t_stage_embedding(batch.t_stage_index)
        return cast(
            torch.Tensor,
            self.projection(torch.cat([continuous, type_embedding, t_stage_embedding], dim=1)),
        )


@dataclass(frozen=True)
class AttentionOutput:
    enhanced_query: torch.Tensor
    weights: torch.Tensor


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> AttentionOutput:
        attended, weights = self.attention(
            query=query,
            key=context,
            value=context,
            need_weights=True,
            average_attn_weights=True,
        )
        return AttentionOutput(
            enhanced_query=self.normalization(query + attended),
            weights=weights,
        )

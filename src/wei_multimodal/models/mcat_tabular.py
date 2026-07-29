from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from wei_multimodal.data.dataset import MultimodalBatch
from wei_multimodal.models.blocks import (
    ClinicalEncoder,
    CrossModalAttention,
    FeatureEncoder,
)


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    attention_weights: torch.Tensor


class MCATTabular(nn.Module):
    def __init__(
        self,
        *,
        type_vocab_size: int,
        t_stage_vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        include_clinical: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.type_vocab_size = type_vocab_size
        self.t_stage_vocab_size = t_stage_vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.include_clinical = include_clinical

        self.path_encoder = FeatureEncoder(768, 256, hidden_dim, dropout)
        self.ct_shape_encoder = FeatureEncoder(14, 64, hidden_dim, dropout)
        self.ct_original_encoder = FeatureEncoder(93, 128, hidden_dim, dropout)
        self.ct_wavelet_encoder = FeatureEncoder(744, 256, hidden_dim, dropout)
        self.ct_transformed_encoder = FeatureEncoder(558, 256, hidden_dim, dropout)
        self.clinical_encoder = ClinicalEncoder(
            type_vocab_size,
            t_stage_vocab_size,
            hidden_dim,
            dropout,
        )
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    @property
    def context_count(self) -> int:
        return 5 if self.include_clinical else 4

    def forward(self, batch: MultimodalBatch) -> ModelOutput:
        pathology = self.path_encoder(batch.pathology)
        query = pathology.unsqueeze(1)
        ct_tokens = torch.stack(
            [
                self.ct_shape_encoder(batch.ct_shape),
                self.ct_original_encoder(batch.ct_original),
                self.ct_wavelet_encoder(batch.ct_wavelet),
                self.ct_transformed_encoder(batch.ct_transformed),
            ],
            dim=1,
        )
        context = ct_tokens
        if self.include_clinical:
            clinical = self.clinical_encoder(batch).unsqueeze(1)
            context = torch.cat([ct_tokens, clinical], dim=1)

        attention = self.cross_attention(query, context)
        fusion = torch.cat([query[:, 0], attention.enhanced_query[:, 0]], dim=1)
        return ModelOutput(
            logits=self.classifier(fusion),
            attention_weights=attention.weights,
        )

    def artifact_config(self) -> dict[str, Any]:
        return {
            "model_name": "MCATTabular",
            "type_vocab_size": self.type_vocab_size,
            "t_stage_vocab_size": self.t_stage_vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "include_clinical": self.include_clinical,
            "clinical_policy": "aggressive",
            "clinical_features": ["age", "male", "Type", "T"],
            "context_count": self.context_count,
        }

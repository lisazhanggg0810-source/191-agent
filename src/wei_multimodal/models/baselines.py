from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from imblearn.ensemble import BalancedRandomForestClassifier
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from torch import nn

from wei_multimodal.data.dataset import MultimodalBatch
from wei_multimodal.models.blocks import ClinicalEncoder, FeatureEncoder
from wei_multimodal.models.mcat_tabular import MCATTabular, ModelOutput

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


def _classifier(input_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.GELU(),
        nn.LayerNorm(64),
        nn.Dropout(dropout),
        nn.Linear(64, 2),
    )


def _without_attention(logits: torch.Tensor) -> ModelOutput:
    empty = logits.new_empty((logits.shape[0], 0, 0))
    return ModelOutput(logits=logits, attention_weights=empty)


class PathologyOnly(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(768, 256, hidden_dim, dropout)
        self.classifier = _classifier(hidden_dim, dropout)

    def forward(self, batch: MultimodalBatch) -> ModelOutput:
        return _without_attention(self.classifier(self.encoder(batch.pathology)))


class CTOnly(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shape_encoder = FeatureEncoder(14, 64, hidden_dim, dropout)
        self.original_encoder = FeatureEncoder(93, 128, hidden_dim, dropout)
        self.wavelet_encoder = FeatureEncoder(744, 256, hidden_dim, dropout)
        self.transformed_encoder = FeatureEncoder(558, 256, hidden_dim, dropout)
        self.classifier = _classifier(hidden_dim * 4, dropout)

    def encode(self, batch: MultimodalBatch) -> torch.Tensor:
        return torch.cat(
            [
                self.shape_encoder(batch.ct_shape),
                self.original_encoder(batch.ct_original),
                self.wavelet_encoder(batch.ct_wavelet),
                self.transformed_encoder(batch.ct_transformed),
            ],
            dim=1,
        )

    def forward(self, batch: MultimodalBatch) -> ModelOutput:
        return _without_attention(self.classifier(self.encode(batch)))


class ClinicalOnly(nn.Module):
    def __init__(
        self,
        type_vocab_size: int,
        t_stage_vocab_size: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = ClinicalEncoder(
            type_vocab_size,
            t_stage_vocab_size,
            hidden_dim,
            dropout,
        )
        self.classifier = _classifier(hidden_dim, dropout)

    def forward(self, batch: MultimodalBatch) -> ModelOutput:
        return _without_attention(self.classifier(self.encoder(batch)))


class ConcatPathCT(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.path_encoder = FeatureEncoder(768, 256, hidden_dim, dropout)
        self.ct_encoder = CTOnly(hidden_dim, dropout)
        self.classifier = _classifier(hidden_dim * 5, dropout)

    def forward(self, batch: MultimodalBatch) -> ModelOutput:
        pathology = self.path_encoder(batch.pathology)
        ct = self.ct_encoder.encode(batch)
        return _without_attention(self.classifier(torch.cat([pathology, ct], dim=1)))


def build_neural_model(
    model_name: str,
    *,
    type_vocab_size: int,
    t_stage_vocab_size: int,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
) -> nn.Module:
    builders: dict[str, Callable[[], nn.Module]] = {
        "pathology_only": lambda: PathologyOnly(hidden_dim, dropout),
        "ct_only": lambda: CTOnly(hidden_dim, dropout),
        "clinical_only": lambda: ClinicalOnly(
            type_vocab_size,
            t_stage_vocab_size,
            hidden_dim,
            dropout,
        ),
        "concat_path_ct": lambda: ConcatPathCT(hidden_dim, dropout),
        "attention_path_ct": lambda: MCATTabular(
            type_vocab_size=type_vocab_size,
            t_stage_vocab_size=t_stage_vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            include_clinical=False,
        ),
        "attention_path_ct_clinical": lambda: MCATTabular(
            type_vocab_size=type_vocab_size,
            t_stage_vocab_size=t_stage_vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            include_clinical=True,
        ),
    }
    try:
        return builders[model_name]()
    except KeyError as error:
        raise ValueError(f"unknown neural model: {model_name}") from error


def build_classical_matrix(
    *,
    pathology: FloatArray,
    ct_shape: FloatArray,
    ct_original: FloatArray,
    ct_wavelet: FloatArray,
    ct_transformed: FloatArray,
    age: FloatArray,
    male: FloatArray,
    type_index: IntArray,
    t_stage_index: IntArray,
    type_vocab_size: int,
    t_stage_vocab_size: int,
) -> FloatArray:
    row_count = pathology.shape[0]
    arrays = (ct_shape, ct_original, ct_wavelet, ct_transformed, age, male)
    if any(array.shape[0] != row_count for array in arrays):
        raise ValueError("classical feature groups have inconsistent row counts")
    if type_index.shape != (row_count,) or t_stage_index.shape != (row_count,):
        raise ValueError("categorical indices must be one-dimensional and row aligned")
    if np.any(type_index < 0) or np.any(type_index >= type_vocab_size):
        raise ValueError("Type index is outside the configured vocabulary")
    if np.any(t_stage_index < 0) or np.any(t_stage_index >= t_stage_vocab_size):
        raise ValueError("T index is outside the configured vocabulary")

    type_one_hot = np.eye(type_vocab_size, dtype=np.float32)[type_index]
    t_stage_one_hot = np.eye(t_stage_vocab_size, dtype=np.float32)[t_stage_index]
    return cast(
        FloatArray,
        np.concatenate(
            [
                pathology,
                ct_shape,
                ct_original,
                ct_wavelet,
                ct_transformed,
                age,
                male,
                type_one_hot,
                t_stage_one_hot,
            ],
            axis=1,
        ).astype(np.float32, copy=False),
    )


def make_elastic_net_logistic(*, seed: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )


def make_balanced_random_forest(
    *,
    seed: int,
    n_estimators: int = 300,
) -> BalancedRandomForestClassifier:
    return BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=1,
        class_weight=None,
        replacement=True,
    )

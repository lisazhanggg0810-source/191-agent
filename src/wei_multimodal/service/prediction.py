from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from wei_multimodal.artifacts.bundle import feature_order_sha256
from wei_multimodal.artifacts.deployment import DeploymentBundle
from wei_multimodal.schema import DataSchema


class PredictionInputError(ValueError):
    """Raised when a named prediction payload violates the bundle schema."""


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = Field(min_length=1, max_length=32)
    feature_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pathology_features: dict[str, float] = Field(min_length=768, max_length=768)
    ct_features: dict[str, float] = Field(min_length=1409, max_length=1409)
    clinical: dict[str, int | float] = Field(min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    positive_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    predicted_class: Literal[0, 1]
    model_version: str
    warnings: list[str]


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_version: str
    bundle_format_version: str
    architecture: str
    schema_version: str
    feature_order_sha256: str
    pathology_feature_count: int
    ct_feature_count: int
    ct_group_counts: dict[str, int]
    clinical_policy: str
    clinical_features: list[str]
    allowed_categories: dict[str, list[str]]
    training_cohort: str
    member_count: int
    threshold: float
    integrity_verified: bool
    independent_test_claim: bool


class PredictionService:
    def __init__(self, bundle_directory: Path, *, device: str = "cpu") -> None:
        self._bundle = DeploymentBundle.load(bundle_directory, device=device)
        self._feature_order_sha256 = feature_order_sha256(self._bundle.schema)
        self._model_version = (bundle_directory / "manifest.sha256").read_text("ascii").strip()
        self.load_count = 1

    @property
    def schema(self) -> DataSchema:
        """Return the frozen input schema owned by the verified deployment bundle.

        MCP adapters need feature names and ordering for validation, but must not
        reach into the private deployment bundle or modify its preprocessing state.
        ``DataSchema`` is a frozen dataclass, so exposing this reference is read-only.
        """

        return self._bundle.schema

    def get_info(self) -> ModelInfo:
        schema = self._bundle.schema
        return ModelInfo(
            model_version=self._model_version,
            bundle_format_version=str(self._bundle.manifest["bundle_format_version"]),
            architecture=str(self._bundle.manifest["architecture"]),
            schema_version=schema.version,
            feature_order_sha256=self._feature_order_sha256,
            pathology_feature_count=len(schema.pathology_features),
            ct_feature_count=schema.ct_feature_count,
            ct_group_counts={
                "shape": len(schema.ct_shape),
                "original": len(schema.ct_original),
                "wavelet": len(schema.ct_wavelet),
                "transformed": len(schema.ct_transformed),
            },
            clinical_policy=str(self._bundle.manifest["clinical_policy"]),
            clinical_features=list(self._bundle.manifest["clinical_features"]),
            allowed_categories={
                name: sorted(mapping, key=lambda value: (float(value), value))
                for name, mapping in self._bundle.preprocessor.category_maps.items()
            },
            training_cohort=str(self._bundle.manifest["training_cohort"]),
            member_count=len(self._bundle.members),
            threshold=self._bundle.threshold,
            integrity_verified=True,
            independent_test_claim=bool(self._bundle.manifest["independent_test_claim"]),
        )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        schema = self._bundle.schema
        if request.schema_version != schema.version:
            raise PredictionInputError(
                f"schema_version must be {schema.version}, got {request.schema_version}"
            )
        if request.feature_order_sha256 != self._feature_order_sha256:
            raise PredictionInputError("feature_order_sha256 does not match the loaded model")
        _validate_exact_keys(
            "pathology_features",
            request.pathology_features,
            set(schema.pathology_features),
        )
        _validate_exact_keys(
            "ct_features",
            request.ct_features,
            set(schema.all_ct_features),
        )
        _validate_exact_keys(
            "clinical",
            request.clinical,
            {"age", "male", "Type", "T"},
        )
        clinical = self._validate_clinical(request.clinical)
        row = {
            **{
                f"pathology::{name}": request.pathology_features[name]
                for name in schema.pathology_features
            },
            **{f"ct::{name}": request.ct_features[name] for name in schema.all_ct_features},
            **clinical,
        }
        ordered_columns = [
            *schema.pathology_output_columns,
            *schema.ct_output_columns,
            "age",
            "male",
            "Type",
            "T",
        ]
        probability = float(
            self._bundle.predict_probabilities(pd.DataFrame([row], columns=ordered_columns))[0]
        )
        if not math.isfinite(probability):
            raise RuntimeError("model returned a non-finite probability")
        predicted_class: Literal[0, 1] = 1 if probability >= self._bundle.threshold else 0
        return PredictionResponse(
            positive_probability=probability,
            threshold=self._bundle.threshold,
            predicted_class=predicted_class,
            model_version=self._model_version,
            warnings=[
                "For research and development use only; this is not a clinical diagnosis.",
                "The deployment ensemble was retrained on all 460 cases and makes no "
                "independent-test claim.",
            ],
        )

    def _validate_clinical(
        self,
        clinical: dict[str, int | float],
    ) -> dict[str, int | float]:
        age = float(clinical["age"])
        male = float(clinical["male"])
        if not 0.0 <= age <= 120.0:
            raise PredictionInputError("clinical.age must be in [0, 120]")
        if male not in {0.0, 1.0}:
            raise PredictionInputError("clinical.male must be 0 or 1")
        validated: dict[str, int | float] = {"age": age, "male": int(male)}
        for field in ("Type", "T"):
            value = clinical[field]
            key = _category_key(value)
            if key not in self._bundle.preprocessor.category_maps[field]:
                allowed = ", ".join(
                    sorted(
                        self._bundle.preprocessor.category_maps[field],
                        key=lambda item: (float(item), item),
                    )
                )
                raise PredictionInputError(
                    f"clinical.{field} must be one of the trained categories: {allowed}"
                )
            validated[field] = value
        return validated


def _validate_exact_keys(
    field_name: str,
    values: Mapping[str, object],
    expected: set[str],
) -> None:
    actual = set(values)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {len(missing)} keys, first: {', '.join(missing[:3])}")
    if extra:
        details.append(f"extra {len(extra)} keys, first: {', '.join(extra[:3])}")
    raise PredictionInputError(f"{field_name} key set is invalid ({'; '.join(details)})")


def _category_key(value: int | float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, ".15g")

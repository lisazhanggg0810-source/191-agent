from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class DataContractError(ValueError):
    """Raised when an input table violates the locked data contract."""


@dataclass(frozen=True)
class DataSchema:
    version: str
    pathology_id_column: str
    radiomics_id_column: str
    pathology_label_source: str
    radiomics_label_column: str
    pathology_features: tuple[str, ...]
    ct_shape: tuple[str, ...]
    ct_original: tuple[str, ...]
    ct_wavelet: tuple[str, ...]
    ct_transformed: tuple[str, ...]
    clinical_columns: tuple[str, ...]

    @classmethod
    def from_columns(
        cls,
        pathology_columns: Iterable[object],
        radiomics_columns: Iterable[object],
    ) -> DataSchema:
        pathology = tuple(str(column) for column in pathology_columns)
        radiomics = tuple(str(column) for column in radiomics_columns)
        if len(pathology) < 2 or len(radiomics) < 6:
            raise DataContractError("input tables do not contain the required columns")

        expected_pathology = tuple(str(index) for index in range(768))
        if pathology[1:-1] != expected_pathology or pathology[-1] != "768":
            raise DataContractError(
                "pathology feature columns must be exactly 0..767 followed by label source 768"
            )

        clinical = ("male", "age", "Type", "T")
        if radiomics[-5:] != (*clinical, "label"):
            raise DataContractError("radiomics table must end with male, age, Type, T, label")

        ct_features = radiomics[1:-5]
        shape = tuple(column for column in ct_features if column.startswith("original_shape"))
        original = tuple(
            column
            for column in ct_features
            if column.startswith("original_") and not column.startswith("original_shape")
        )
        wavelet = tuple(column for column in ct_features if column.startswith("wavelet"))
        transformed_prefixes = (
            "exponential_",
            "gradient_",
            "lbp-2D_",
            "logarithm_",
            "square_",
            "squareroot_",
        )
        transformed = tuple(
            column for column in ct_features if column.startswith(transformed_prefixes)
        )
        recognized = set(shape) | set(original) | set(wavelet) | set(transformed)
        unsupported = tuple(column for column in ct_features if column not in recognized)
        if unsupported:
            preview = ", ".join(unsupported[:5])
            raise DataContractError(f"unsupported CT feature columns: {preview}")

        counts = (len(shape), len(original), len(wavelet), len(transformed))
        if counts != (14, 93, 744, 558):
            raise DataContractError(
                "CT channel counts must be shape/original/wavelet/transformed "
                f"14/93/744/558, got {counts}"
            )

        return cls(
            version="1.0.0",
            pathology_id_column=pathology[0],
            radiomics_id_column=radiomics[0],
            pathology_label_source="768",
            radiomics_label_column="label",
            pathology_features=expected_pathology,
            ct_shape=shape,
            ct_original=original,
            ct_wavelet=wavelet,
            ct_transformed=transformed,
            clinical_columns=clinical,
        )

    @property
    def all_ct_features(self) -> tuple[str, ...]:
        return self.ct_shape + self.ct_original + self.ct_wavelet + self.ct_transformed

    @property
    def ct_feature_count(self) -> int:
        return len(self.all_ct_features)

    @property
    def pathology_output_columns(self) -> tuple[str, ...]:
        return tuple(f"pathology::{column}" for column in self.pathology_features)

    @property
    def ct_output_columns(self) -> tuple[str, ...]:
        return tuple(f"ct::{column}" for column in self.all_ct_features)

    def ct_group_output_columns(self, group: str) -> tuple[str, ...]:
        source = {
            "shape": self.ct_shape,
            "original": self.ct_original,
            "wavelet": self.ct_wavelet,
            "transformed": self.ct_transformed,
        }.get(group)
        if source is None:
            raise KeyError(f"unknown CT group: {group}")
        return tuple(f"ct::{column}" for column in source)

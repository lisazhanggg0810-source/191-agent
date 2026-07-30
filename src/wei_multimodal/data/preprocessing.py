from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wei_multimodal.schema import DataContractError, DataSchema

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True)
class PreprocessedArrays:
    patient_ids: tuple[str, ...]
    pathology: Float32Array
    ct_shape: Float32Array
    ct_original: Float32Array
    ct_wavelet: Float32Array
    ct_transformed: Float32Array
    age: Float32Array
    male: Float32Array
    type_index: Int64Array
    t_stage_index: Int64Array
    labels: Int64Array | None

    _ARRAY_FIELDS: ClassVar[tuple[str, ...]] = (
        "pathology",
        "ct_shape",
        "ct_original",
        "ct_wavelet",
        "ct_transformed",
        "age",
        "male",
        "type_index",
        "t_stage_index",
        "labels",
    )

    def array_fields(self) -> tuple[str, ...]:
        return tuple(field for field in self._ARRAY_FIELDS if getattr(self, field) is not None)

    def __len__(self) -> int:
        return int(self.pathology.shape[0])


class FoldPreprocessor:
    def __init__(
        self,
        *,
        schema_version: str,
        group_columns: dict[str, tuple[str, ...]],
        means: dict[str, Float64Array],
        scales: dict[str, Float64Array],
        category_maps: dict[str, dict[str, int]],
        constant_columns: tuple[str, ...],
    ) -> None:
        self.schema_version = schema_version
        self.group_columns = group_columns
        self.means = means
        self.scales = scales
        self.category_maps = category_maps
        self.constant_columns = constant_columns

    @classmethod
    def fit(cls, frame: pd.DataFrame, schema: DataSchema) -> FoldPreprocessor:
        group_columns = cls._group_columns(schema)
        cls._validate_frame_columns(frame, group_columns)
        means: dict[str, Float64Array] = {}
        scales: dict[str, Float64Array] = {}
        constants: list[str] = []

        for group, columns in group_columns.items():
            values = cls._numeric_matrix(frame, columns, group)
            mean = values.mean(axis=0, dtype=np.float64)
            raw_scale = values.std(axis=0, dtype=np.float64)
            constant = raw_scale == 0.0
            constants.extend(column for column, flag in zip(columns, constant, strict=True) if flag)
            scale = raw_scale.copy()
            scale[constant] = 1.0
            means[group] = mean
            scales[group] = scale

        cls._validate_male(frame["male"])
        category_maps = {
            "Type": cls._fit_category_map(frame["Type"]),
            "T": cls._fit_category_map(frame["T"]),
        }
        return cls(
            schema_version=schema.version,
            group_columns=group_columns,
            means=means,
            scales=scales,
            category_maps=category_maps,
            constant_columns=tuple(constants),
        )

    def transform(self, frame: pd.DataFrame) -> PreprocessedArrays:
        self._validate_frame_columns(frame, self.group_columns)
        self._validate_male(frame["male"])
        transformed = {group: self._standardize(frame, group) for group in self.group_columns}
        type_index = self._encode_categories(frame["Type"], self.category_maps["Type"])
        t_stage_index = self._encode_categories(frame["T"], self.category_maps["T"])
        labels: Int64Array | None = None
        if "label" in frame.columns:
            numeric_labels = cast(
                NDArray[np.generic],
                pd.to_numeric(frame["label"], errors="raise").to_numpy(),
            )
            if not np.isin(numeric_labels, [0, 1]).all():
                raise DataContractError("labels must contain only 0 and 1")
            labels = numeric_labels.astype(np.int64, copy=False)

        patient_ids = (
            tuple(frame["patient_id"].astype(str)) if "patient_id" in frame.columns else ()
        )
        return PreprocessedArrays(
            patient_ids=patient_ids,
            pathology=transformed["pathology"],
            ct_shape=transformed["ct_shape"],
            ct_original=transformed["ct_original"],
            ct_wavelet=transformed["ct_wavelet"],
            ct_transformed=transformed["ct_transformed"],
            age=transformed["age"],
            male=cast(
                Float32Array,
                frame[["male"]].to_numpy(dtype=np.float32, copy=True),
            ),
            type_index=type_index,
            t_stage_index=t_stage_index,
            labels=labels,
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": self.schema_version,
            "clinical_policy": "aggressive",
            "clinical_features": ["age", "male", "Type", "T"],
            "group_columns": {
                group: list(columns) for group, columns in self.group_columns.items()
            },
            "category_maps": self.category_maps,
            "constant_columns": list(self.constant_columns),
        }
        (directory / "preprocessing.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        arrays: dict[str, Float64Array] = {}
        for group in self.group_columns:
            arrays[f"mean__{group}"] = self.means[group]
            arrays[f"scale__{group}"] = self.scales[group]
        # NumPy's typed ``allow_pickle`` keyword overlaps dynamic array names in
        # the current stubs; runtime ``savez_compressed`` accepts these arrays.
        np.savez_compressed(directory / "preprocessing.npz", **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, directory: Path, schema: DataSchema) -> FoldPreprocessor:
        metadata = json.loads((directory / "preprocessing.json").read_text("utf-8"))
        if metadata.get("schema_version") != schema.version:
            raise DataContractError("preprocessor schema version does not match input schema")
        expected_columns = cls._group_columns(schema)
        saved_columns = {
            group: tuple(columns) for group, columns in metadata.get("group_columns", {}).items()
        }
        if saved_columns != expected_columns:
            raise DataContractError("preprocessor feature order does not match input schema")

        means: dict[str, Float64Array] = {}
        scales: dict[str, Float64Array] = {}
        with np.load(directory / "preprocessing.npz", allow_pickle=False) as arrays:
            for group in expected_columns:
                means[group] = cast(Float64Array, arrays[f"mean__{group}"].copy())
                scales[group] = cast(Float64Array, arrays[f"scale__{group}"].copy())
        return cls(
            schema_version=schema.version,
            group_columns=expected_columns,
            means=means,
            scales=scales,
            category_maps={
                field: {str(key): int(value) for key, value in mapping.items()}
                for field, mapping in metadata["category_maps"].items()
            },
            constant_columns=tuple(metadata["constant_columns"]),
        )

    @staticmethod
    def _group_columns(schema: DataSchema) -> dict[str, tuple[str, ...]]:
        return {
            "pathology": schema.pathology_output_columns,
            "ct_shape": schema.ct_group_output_columns("shape"),
            "ct_original": schema.ct_group_output_columns("original"),
            "ct_wavelet": schema.ct_group_output_columns("wavelet"),
            "ct_transformed": schema.ct_group_output_columns("transformed"),
            "age": ("age",),
        }

    @staticmethod
    def _validate_frame_columns(
        frame: pd.DataFrame,
        group_columns: dict[str, tuple[str, ...]],
    ) -> None:
        required = {column for columns in group_columns.values() for column in columns} | {
            "male",
            "Type",
            "T",
        }
        actual = set(frame.columns)
        missing = sorted(required - actual)
        if missing:
            raise DataContractError(f"missing preprocessing columns: {', '.join(missing[:5])}")
        allowed = required | {"patient_id", "label"}
        unexpected = sorted(actual - allowed)
        if unexpected:
            raise DataContractError(
                f"unexpected preprocessing columns: {', '.join(unexpected[:5])}"
            )

    @staticmethod
    def _numeric_matrix(
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        group: str,
    ) -> Float64Array:
        numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="raise")
        values = cast(Float64Array, numeric.to_numpy(dtype=np.float64, copy=True))
        if not np.isfinite(values).all():
            raise DataContractError(f"{group} contains non-finite values")
        return values

    def _standardize(self, frame: pd.DataFrame, group: str) -> Float32Array:
        values = self._numeric_matrix(frame, self.group_columns[group], group)
        standardized = (values - self.means[group]) / self.scales[group]
        if not np.isfinite(standardized).all():
            raise DataContractError(f"{group} preprocessing produced non-finite values")
        return cast(Float32Array, standardized.astype(np.float32, copy=False))

    @staticmethod
    def _validate_male(series: pd.Series) -> None:
        numeric = pd.to_numeric(series, errors="raise")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise DataContractError("male must contain only 0 and 1")

    @classmethod
    def _fit_category_map(cls, series: pd.Series) -> dict[str, int]:
        categories = sorted({cls._category_key(value) for value in series})
        return {category: index for index, category in enumerate(categories, start=1)}

    @classmethod
    def _encode_categories(
        cls,
        series: pd.Series,
        mapping: dict[str, int],
    ) -> Int64Array:
        return cast(
            Int64Array,
            np.asarray(
                [mapping.get(cls._category_key(value), 0) for value in series],
                dtype=np.int64,
            ),
        )

    @staticmethod
    def _category_key(value: object) -> str:
        if pd.isna(value):
            return "<MISSING>"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return format(float(value), ".15g")
        return str(value).strip()

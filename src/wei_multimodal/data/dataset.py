from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch
from torch.utils.data import Dataset

from wei_multimodal.data.preprocessing import PreprocessedArrays
from wei_multimodal.schema import DataContractError


@dataclass(frozen=True)
class MultimodalSample:
    patient_id: str
    pathology: torch.Tensor
    ct_shape: torch.Tensor
    ct_original: torch.Tensor
    ct_wavelet: torch.Tensor
    ct_transformed: torch.Tensor
    age: torch.Tensor
    male: torch.Tensor
    type_index: torch.Tensor
    t_stage_index: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class MultimodalBatch:
    patient_ids: tuple[str, ...]
    pathology: torch.Tensor
    ct_shape: torch.Tensor
    ct_original: torch.Tensor
    ct_wavelet: torch.Tensor
    ct_transformed: torch.Tensor
    age: torch.Tensor
    male: torch.Tensor
    type_index: torch.Tensor
    t_stage_index: torch.Tensor
    target: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.target.shape[0])

    def to(self, device: torch.device, *, non_blocking: bool = False) -> MultimodalBatch:
        values: dict[str, object] = {"patient_ids": self.patient_ids}
        for field in fields(self):
            if field.name == "patient_ids":
                continue
            tensor = getattr(self, field.name)
            values[field.name] = tensor.to(device, non_blocking=non_blocking)
        return MultimodalBatch(**values)  # type: ignore[arg-type]


class MultimodalDataset(Dataset[MultimodalSample]):
    def __init__(self, arrays: PreprocessedArrays) -> None:
        if arrays.labels is None:
            raise DataContractError("training dataset requires labels")
        sample_count = len(arrays)
        if len(arrays.patient_ids) != sample_count:
            raise DataContractError("patient ID count does not match feature rows")
        for field_name in arrays.array_fields():
            value = getattr(arrays, field_name)
            if value.shape[0] != sample_count:
                raise DataContractError(f"{field_name} row count does not match pathology")

        self.patient_ids = arrays.patient_ids
        self.pathology = torch.from_numpy(arrays.pathology)
        self.ct_shape = torch.from_numpy(arrays.ct_shape)
        self.ct_original = torch.from_numpy(arrays.ct_original)
        self.ct_wavelet = torch.from_numpy(arrays.ct_wavelet)
        self.ct_transformed = torch.from_numpy(arrays.ct_transformed)
        self.age = torch.from_numpy(arrays.age)
        self.male = torch.from_numpy(arrays.male)
        self.type_index = torch.from_numpy(arrays.type_index)
        self.t_stage_index = torch.from_numpy(arrays.t_stage_index)
        self.target = torch.from_numpy(arrays.labels)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> MultimodalSample:
        return MultimodalSample(
            patient_id=self.patient_ids[index],
            pathology=self.pathology[index],
            ct_shape=self.ct_shape[index],
            ct_original=self.ct_original[index],
            ct_wavelet=self.ct_wavelet[index],
            ct_transformed=self.ct_transformed[index],
            age=self.age[index],
            male=self.male[index],
            type_index=self.type_index[index],
            t_stage_index=self.t_stage_index[index],
            target=self.target[index],
        )


def collate_multimodal(samples: Sequence[MultimodalSample]) -> MultimodalBatch:
    if not samples:
        raise DataContractError("cannot collate an empty batch")
    return MultimodalBatch(
        patient_ids=tuple(sample.patient_id for sample in samples),
        pathology=torch.stack([sample.pathology for sample in samples]),
        ct_shape=torch.stack([sample.ct_shape for sample in samples]),
        ct_original=torch.stack([sample.ct_original for sample in samples]),
        ct_wavelet=torch.stack([sample.ct_wavelet for sample in samples]),
        ct_transformed=torch.stack([sample.ct_transformed for sample in samples]),
        age=torch.stack([sample.age for sample in samples]),
        male=torch.stack([sample.male for sample in samples]),
        type_index=torch.stack([sample.type_index for sample in samples]),
        t_stage_index=torch.stack([sample.t_stage_index for sample in samples]),
        target=torch.stack([sample.target for sample in samples]),
    )

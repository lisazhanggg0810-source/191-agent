from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from wei_multimodal.artifacts.integrity import sha256_file
from wei_multimodal.data.dataset import MultimodalBatch
from wei_multimodal.data.preprocessing import FoldPreprocessor, PreprocessedArrays
from wei_multimodal.models.baselines import build_neural_model
from wei_multimodal.models.mcat_tabular import ModelOutput
from wei_multimodal.schema import DataContractError, DataSchema

FloatArray = NDArray[np.float64]
BUNDLE_FORMAT_VERSION = "1.0.0"
MEMBER_PAYLOADS = (
    "model_state.pt",
    "model_config.json",
    "preprocessing.npz",
    "preprocessing.json",
)


class BundleIntegrityError(DataContractError):
    """Raised when a model bundle fails structural or cryptographic validation."""


@dataclass(frozen=True)
class EvaluationBundleMember:
    seed: int
    fold: int
    model: nn.Module
    preprocessor: FoldPreprocessor
    model_config: dict[str, Any]
    device: torch.device

    def predict_logits(self, frame: pd.DataFrame) -> torch.Tensor:
        arrays = self.preprocessor.transform(frame)
        batch = preprocessed_to_batch(arrays).to(self.device)
        with torch.no_grad():
            output = cast(ModelOutput, self.model(batch))
        return output.logits.detach().cpu()


@dataclass(frozen=True)
class EvaluationBundle:
    directory: Path
    manifest: dict[str, Any]
    schema: DataSchema
    threshold: float
    members: tuple[EvaluationBundleMember, ...]

    @classmethod
    def load(cls, directory: Path, *, device: str | torch.device) -> EvaluationBundle:
        directory = directory.resolve()
        manifest_path = directory / "manifest.json"
        digest_path = directory / "manifest.sha256"
        if not manifest_path.is_file() or not digest_path.is_file():
            raise BundleIntegrityError("bundle manifest or manifest hash is missing")
        expected_manifest_hash = digest_path.read_text("ascii").strip()
        if sha256_file(manifest_path) != expected_manifest_hash:
            raise BundleIntegrityError("manifest hash mismatch")
        manifest = _read_json(manifest_path)
        if manifest.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
            raise BundleIntegrityError("unsupported bundle format version")
        if manifest.get("bundle_type") != "evaluation":
            raise BundleIntegrityError("bundle is not an evaluation bundle")

        payload_hashes = manifest.get("payload_sha256")
        if not isinstance(payload_hashes, dict) or not payload_hashes:
            raise BundleIntegrityError("payload hash index is missing")
        _verify_payloads(directory, payload_hashes)
        schema = schema_from_dict(_read_json(directory / "schema.json"))
        if schema.version != manifest.get("schema_version"):
            raise BundleIntegrityError("manifest schema version mismatch")
        if feature_order_sha256(schema) != manifest.get("feature_order_sha256"):
            raise BundleIntegrityError("schema field order mismatch")
        threshold_payload = _read_json(directory / "threshold.json")
        threshold = float(threshold_payload.get("value", math.nan))
        if not 0.0 <= threshold <= 1.0:
            raise BundleIntegrityError("bundle threshold is invalid")

        resolved_device = torch.device(device)
        members = tuple(
            _load_member(directory, item, schema, resolved_device)
            for item in manifest.get("members", [])
        )
        if len(members) != 15:
            raise BundleIntegrityError("evaluation bundle must contain exactly 15 members")
        return cls(
            directory=directory,
            manifest=manifest,
            schema=schema,
            threshold=threshold,
            members=members,
        )

    def predict_probabilities(self, frame: pd.DataFrame) -> FloatArray:
        if frame.empty:
            raise DataContractError("prediction frame must contain at least one row")
        member_probabilities = [
            torch.softmax(member.predict_logits(frame), dim=1)[:, 1] for member in self.members
        ]
        averaged = torch.stack(member_probabilities, dim=0).mean(dim=0)
        return cast(FloatArray, averaged.numpy().astype(np.float64, copy=False))


def package_evaluation_bundle(
    run_directory: Path,
    output_directory: Path,
    *,
    schema: DataSchema,
    source_git_sha: str,
    source_data_hashes: dict[str, str],
    environment: dict[str, str],
) -> Path:
    run_directory = run_directory.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise BundleIntegrityError("bundle output directory already exists")
    run_manifest = _read_json(run_directory / "run_manifest.json")
    selection = _validate_complete_run(run_directory, run_manifest)
    output_directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_directory.name}-",
        dir=output_directory.parent,
    ) as temporary:
        stage = Path(temporary) / output_directory.name
        stage.mkdir()
        schema_payload = schema_to_dict(schema)
        (stage / "schema.json").write_text(
            json.dumps(schema_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (stage / "threshold.json").write_text(
            json.dumps(
                {
                    "method": "oof_youden",
                    "source_cohort": "development_340",
                    "value": float(run_manifest["threshold"]),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        members: list[dict[str, Any]] = []
        for seed in tuple(int(value) for value in run_manifest["expected_seeds"]):
            for fold in range(5):
                relative = Path(f"seed_{seed}") / f"fold_{fold:02d}"
                source = run_directory / relative
                destination = stage / relative
                destination.mkdir(parents=True)
                for filename in MEMBER_PAYLOADS:
                    shutil.copy2(source / filename, destination / filename)
                members.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "directory": relative.as_posix(),
                    }
                )

        payload_hashes = {
            path.relative_to(stage).as_posix(): sha256_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "bundle_type": "evaluation",
            "schema_version": schema.version,
            "feature_order_sha256": feature_order_sha256(schema),
            "architecture": run_manifest["architecture"],
            "clinical_policy": run_manifest["clinical_policy"],
            "clinical_features": run_manifest["clinical_features"],
            "training_cohort": "development_340_oof",
            "source_git_sha": source_git_sha,
            "source_data_sha256": source_data_hashes,
            "environment": environment,
            "threshold": float(run_manifest["threshold"]),
            "model_selection": {
                "selected_model": selection["selected_model"],
                "selection_rule": selection["selection_rule"],
                "locked_test_used": selection.get("locked_test_used"),
                "neural_models": selection.get("neural_models", {}),
            },
            "members": members,
            "payload_sha256": payload_hashes,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (stage / "manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}\n",
            encoding="ascii",
        )
        os.replace(stage, output_directory)
    return output_directory


def preprocessed_to_batch(arrays: PreprocessedArrays) -> MultimodalBatch:
    row_count = len(arrays)
    labels = arrays.labels
    if labels is None:
        labels = np.zeros(row_count, dtype=np.int64)
    return MultimodalBatch(
        patient_ids=arrays.patient_ids,
        pathology=torch.from_numpy(arrays.pathology),
        ct_shape=torch.from_numpy(arrays.ct_shape),
        ct_original=torch.from_numpy(arrays.ct_original),
        ct_wavelet=torch.from_numpy(arrays.ct_wavelet),
        ct_transformed=torch.from_numpy(arrays.ct_transformed),
        age=torch.from_numpy(arrays.age),
        male=torch.from_numpy(arrays.male),
        type_index=torch.from_numpy(arrays.type_index),
        t_stage_index=torch.from_numpy(arrays.t_stage_index),
        target=torch.from_numpy(labels),
    )


def schema_to_dict(schema: DataSchema) -> dict[str, Any]:
    payload = asdict(schema)
    return {
        key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()
    }


def schema_from_dict(payload: dict[str, Any]) -> DataSchema:
    try:
        schema = DataSchema(
            version=str(payload["version"]),
            pathology_id_column=str(payload["pathology_id_column"]),
            radiomics_id_column=str(payload["radiomics_id_column"]),
            pathology_label_source=str(payload["pathology_label_source"]),
            radiomics_label_column=str(payload["radiomics_label_column"]),
            pathology_features=tuple(str(value) for value in payload["pathology_features"]),
            ct_shape=tuple(str(value) for value in payload["ct_shape"]),
            ct_original=tuple(str(value) for value in payload["ct_original"]),
            ct_wavelet=tuple(str(value) for value in payload["ct_wavelet"]),
            ct_transformed=tuple(str(value) for value in payload["ct_transformed"]),
            clinical_columns=tuple(str(value) for value in payload["clinical_columns"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BundleIntegrityError("schema payload is invalid") from error
    if len(schema.pathology_features) != 768:
        raise BundleIntegrityError("schema pathology field count is invalid")
    if (
        len(schema.ct_shape),
        len(schema.ct_original),
        len(schema.ct_wavelet),
        len(schema.ct_transformed),
    ) != (14, 93, 744, 558):
        raise BundleIntegrityError("schema CT field counts are invalid")
    if schema.clinical_columns != ("male", "age", "Type", "T"):
        raise BundleIntegrityError("schema clinical field order is invalid")
    return schema


def feature_order_sha256(schema: DataSchema) -> str:
    ordered = [
        *schema.pathology_output_columns,
        *schema.ct_output_columns,
        *schema.clinical_columns,
    ]
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _validate_complete_run(
    run_directory: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if manifest.get("incomplete") is not False:
        raise BundleIntegrityError("incomplete training run cannot be packaged")
    seeds = tuple(int(value) for value in manifest.get("expected_seeds", []))
    if seeds != (2024, 3407, 5280) or manifest.get("completed_members") != 15:
        raise BundleIntegrityError("training run does not contain the required 15 members")
    threshold = float(manifest.get("threshold", math.nan))
    if not 0.0 <= threshold <= 1.0:
        raise BundleIntegrityError("training run threshold is missing or invalid")
    selection_path = run_directory / "selection.json"
    if not selection_path.is_file():
        raise BundleIntegrityError("model selection record is missing")
    selection = _read_json(selection_path)
    if not selection.get("selection_rule"):
        raise BundleIntegrityError("model selection rule is missing")
    if selection.get("locked_test_used") is not False:
        raise BundleIntegrityError("model selection must not use the locked test cohort")
    if selection.get("selected_model") != manifest.get("architecture"):
        raise BundleIntegrityError("selected model does not match trained architecture")

    oof_path = run_directory / "oof_predictions.csv"
    if not oof_path.is_file():
        raise BundleIntegrityError("OOF prediction record is missing")
    oof = pd.read_csv(oof_path, usecols=["patient_id"])
    if len(oof) != 340 or oof["patient_id"].astype(str).nunique() != 340:
        raise BundleIntegrityError("OOF predictions must cover exactly 340 unique patients")

    for seed in seeds:
        for fold in range(5):
            member = run_directory / f"seed_{seed}" / f"fold_{fold:02d}"
            for filename in MEMBER_PAYLOADS:
                if not (member / filename).is_file():
                    raise BundleIntegrityError(
                        f"training member payload is missing: seed={seed} fold={fold} {filename}"
                    )
            config = _read_json(member / "model_config.json")
            if config.get("architecture") != manifest.get("architecture"):
                raise BundleIntegrityError("member architecture does not match run manifest")
            if config.get("clinical_policy") != "aggressive":
                raise BundleIntegrityError("member clinical policy is not aggressive")
    return selection


def _verify_payloads(directory: Path, payload_hashes: dict[str, Any]) -> None:
    for relative_name, expected_hash in payload_hashes.items():
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise BundleIntegrityError("payload index contains an unsafe path")
        path = directory.joinpath(*relative.parts)
        if not path.is_file():
            raise BundleIntegrityError(f"bundle payload is missing: {relative_name}")
        if sha256_file(path) != expected_hash:
            raise BundleIntegrityError(f"payload hash mismatch: {relative_name}")


def _load_member(
    directory: Path,
    item: Any,
    schema: DataSchema,
    device: torch.device,
) -> EvaluationBundleMember:
    if not isinstance(item, dict):
        raise BundleIntegrityError("member index is invalid")
    relative = PurePosixPath(str(item.get("directory", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise BundleIntegrityError("member index contains an unsafe path")
    member_directory = directory.joinpath(*relative.parts)
    config = _read_json(member_directory / "model_config.json")
    try:
        model = build_neural_model(
            str(config["architecture"]),
            type_vocab_size=int(config["type_vocab_size"]),
            t_stage_vocab_size=int(config["t_stage_vocab_size"]),
            hidden_dim=int(config["hidden_dim"]),
            num_heads=int(config["num_heads"]),
            dropout=float(config["dropout"]),
        )
        state = torch.load(
            member_directory / "model_state.pt",
            weights_only=True,
            map_location=device,
        )
        if not isinstance(state, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
        ):
            raise BundleIntegrityError("model state is not a tensor state_dict")
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        preprocessor = FoldPreprocessor.load(member_directory, schema)
    except BundleIntegrityError:
        raise
    except Exception as error:
        raise BundleIntegrityError("bundle member could not be loaded safely") from error
    return EvaluationBundleMember(
        seed=int(item["seed"]),
        fold=int(item["fold"]),
        model=model,
        preprocessor=preprocessor,
        model_config=config,
        device=device,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BundleIntegrityError(f"required JSON file is missing: {path.name}")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleIntegrityError(f"required JSON file is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise BundleIntegrityError(f"required JSON object is invalid: {path.name}")
    return value

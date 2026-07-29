from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from wei_multimodal.artifacts.bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleIntegrityError,
    EvaluationBundle,
    feature_order_sha256,
    preprocessed_to_batch,
    schema_from_dict,
    schema_to_dict,
)
from wei_multimodal.artifacts.integrity import sha256_file
from wei_multimodal.data.preprocessing import FoldPreprocessor
from wei_multimodal.models.baselines import build_neural_model
from wei_multimodal.models.mcat_tabular import ModelOutput
from wei_multimodal.schema import DataContractError, DataSchema

FloatArray = NDArray[np.float64]
DEFAULT_DEPLOYMENT_SEEDS = (2024, 3407, 5280, 7319, 9021)


@dataclass(frozen=True)
class DeploymentSpec:
    architecture: str
    fixed_epochs: int
    seeds: tuple[int, ...]
    threshold: float


@dataclass(frozen=True)
class DeploymentMemberArtifact:
    seed: int
    model_state: dict[str, torch.Tensor]
    model_config: dict[str, Any]


@dataclass(frozen=True)
class DeploymentBundleMember:
    seed: int
    model: nn.Module
    model_config: dict[str, Any]
    device: torch.device

    def predict_logits(self, batch: Any) -> torch.Tensor:
        with torch.no_grad():
            output = cast(ModelOutput, self.model(batch.to(self.device)))
        return output.logits.detach().cpu()


@dataclass(frozen=True)
class DeploymentBundle:
    directory: Path
    manifest: dict[str, Any]
    schema: DataSchema
    threshold: float
    preprocessor: FoldPreprocessor
    members: tuple[DeploymentBundleMember, ...]

    @classmethod
    def load(cls, directory: Path, *, device: str | torch.device) -> DeploymentBundle:
        directory = directory.resolve()
        manifest_path = directory / "manifest.json"
        digest_path = directory / "manifest.sha256"
        if not manifest_path.is_file() or not digest_path.is_file():
            raise BundleIntegrityError("deployment manifest or manifest hash is missing")
        if sha256_file(manifest_path) != digest_path.read_text("ascii").strip():
            raise BundleIntegrityError("deployment manifest hash mismatch")
        manifest = _read_json(manifest_path)
        if manifest.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
            raise BundleIntegrityError("unsupported deployment bundle format version")
        if manifest.get("bundle_type") != "deployment":
            raise BundleIntegrityError("bundle is not a deployment bundle")
        if manifest.get("training_cohort") != "all_460":
            raise BundleIntegrityError("deployment training cohort is not all_460")
        if manifest.get("independent_test_claim") is not False:
            raise BundleIntegrityError("deployment bundle makes an invalid test claim")
        _verify_payloads(directory, manifest.get("payload_sha256"))
        schema = schema_from_dict(_read_json(directory / "schema.json"))
        if feature_order_sha256(schema) != manifest.get("feature_order_sha256"):
            raise BundleIntegrityError("deployment schema field order mismatch")
        threshold_payload = _read_json(directory / "threshold.json")
        threshold = float(threshold_payload.get("value", math.nan))
        if not 0.0 <= threshold <= 1.0:
            raise BundleIntegrityError("deployment threshold is invalid")
        preprocessor = FoldPreprocessor.load(directory, schema)
        resolved_device = torch.device(device)
        members = tuple(
            _load_member(directory, item, resolved_device) for item in manifest.get("members", [])
        )
        if tuple(member.seed for member in members) != DEFAULT_DEPLOYMENT_SEEDS:
            raise BundleIntegrityError("deployment bundle seed list is invalid")
        return cls(
            directory=directory,
            manifest=manifest,
            schema=schema,
            threshold=threshold,
            preprocessor=preprocessor,
            members=members,
        )

    def predict_probabilities(self, frame: pd.DataFrame) -> FloatArray:
        if frame.empty:
            raise DataContractError("prediction frame must contain at least one row")
        batch = preprocessed_to_batch(self.preprocessor.transform(frame))
        probabilities = [
            torch.softmax(member.predict_logits(batch), dim=1)[:, 1] for member in self.members
        ]
        averaged = torch.stack(probabilities, dim=0).mean(dim=0)
        return cast(FloatArray, averaged.numpy().astype(np.float64, copy=False))


def derive_deployment_spec(evaluation: EvaluationBundle) -> DeploymentSpec:
    selection = evaluation.manifest.get("model_selection", {})
    architecture = str(evaluation.manifest.get("architecture", ""))
    if selection.get("selected_model") != architecture:
        raise BundleIntegrityError("evaluation selection does not match its architecture")
    if selection.get("locked_test_used") is not False:
        raise BundleIntegrityError("deployment architecture selection used locked test data")
    best_epochs = [int(member.model_config.get("best_epoch", 0)) for member in evaluation.members]
    if len(best_epochs) != 15 or any(epoch < 1 for epoch in best_epochs):
        raise BundleIntegrityError("evaluation bundle does not contain 15 valid best epochs")
    fixed_epochs = int(median(best_epochs))
    return DeploymentSpec(
        architecture=architecture,
        fixed_epochs=fixed_epochs,
        seeds=DEFAULT_DEPLOYMENT_SEEDS,
        threshold=evaluation.threshold,
    )


def package_deployment_bundle(
    output_directory: Path,
    *,
    schema: DataSchema,
    preprocessor: FoldPreprocessor,
    artifacts: tuple[DeploymentMemberArtifact, ...],
    architecture: str,
    threshold: float,
    evaluation_manifest_sha256: str,
    source_git_sha: str,
    source_data_hashes: dict[str, str],
    environment: dict[str, str],
) -> Path:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise BundleIntegrityError("deployment output directory already exists")
    if tuple(artifact.seed for artifact in artifacts) != DEFAULT_DEPLOYMENT_SEEDS:
        raise BundleIntegrityError("deployment artifacts do not match the fixed seed list")
    if not 0.0 <= threshold <= 1.0:
        raise BundleIntegrityError("deployment threshold is invalid")
    for artifact in artifacts:
        if artifact.model_config.get("architecture") != architecture:
            raise BundleIntegrityError("deployment member architecture mismatch")
        if artifact.model_config.get("clinical_policy") != "aggressive":
            raise BundleIntegrityError("deployment member clinical policy is not aggressive")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_directory.name}-",
        dir=output_directory.parent,
    ) as temporary:
        stage = Path(temporary) / output_directory.name
        stage.mkdir()
        (stage / "schema.json").write_text(
            json.dumps(schema_to_dict(schema), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (stage / "threshold.json").write_text(
            json.dumps(
                {
                    "method": "oof_youden",
                    "source": "evaluation_bundle",
                    "source_manifest_sha256": evaluation_manifest_sha256,
                    "value": float(threshold),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        preprocessor.save(stage)
        members: list[dict[str, Any]] = []
        for artifact in artifacts:
            relative = Path(f"seed_{artifact.seed}")
            member_directory = stage / relative
            member_directory.mkdir()
            torch.save(artifact.model_state, member_directory / "model_state.pt")
            (member_directory / "model_config.json").write_text(
                json.dumps(artifact.model_config, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            members.append({"seed": artifact.seed, "directory": relative.as_posix()})
        payload_hashes = {
            path.relative_to(stage).as_posix(): sha256_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "bundle_type": "deployment",
            "schema_version": schema.version,
            "feature_order_sha256": feature_order_sha256(schema),
            "architecture": architecture,
            "clinical_policy": "aggressive",
            "clinical_features": ["age", "male", "Type", "T"],
            "training_cohort": "all_460",
            "independent_test_claim": False,
            "threshold": float(threshold),
            "threshold_source_manifest_sha256": evaluation_manifest_sha256,
            "source_git_sha": source_git_sha,
            "source_data_sha256": source_data_hashes,
            "environment": environment,
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


def _load_member(
    directory: Path,
    item: Any,
    device: torch.device,
) -> DeploymentBundleMember:
    if not isinstance(item, dict):
        raise BundleIntegrityError("deployment member index is invalid")
    relative = PurePosixPath(str(item.get("directory", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise BundleIntegrityError("deployment member path is unsafe")
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
            raise BundleIntegrityError("deployment state is not a tensor state_dict")
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
    except BundleIntegrityError:
        raise
    except Exception as error:
        raise BundleIntegrityError("deployment member could not be loaded safely") from error
    return DeploymentBundleMember(
        seed=int(item["seed"]),
        model=model,
        model_config=config,
        device=device,
    )


def _verify_payloads(directory: Path, payload_hashes: Any) -> None:
    if not isinstance(payload_hashes, dict) or not payload_hashes:
        raise BundleIntegrityError("deployment payload hash index is missing")
    for relative_name, expected_hash in payload_hashes.items():
        relative = PurePosixPath(str(relative_name))
        if relative.is_absolute() or ".." in relative.parts:
            raise BundleIntegrityError("deployment payload path is unsafe")
        path = directory.joinpath(*relative.parts)
        if not path.is_file():
            raise BundleIntegrityError(f"deployment payload is missing: {relative_name}")
        if sha256_file(path) != expected_hash:
            raise BundleIntegrityError(f"deployment payload hash mismatch: {relative_name}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BundleIntegrityError(f"deployment JSON is missing: {path.name}")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleIntegrityError(f"deployment JSON is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise BundleIntegrityError(f"deployment JSON object is invalid: {path.name}")
    return value

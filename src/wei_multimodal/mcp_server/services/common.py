"""Core dependencies and deterministic helpers shared by the six MCP services."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from wei_multimodal.mcp_server.artifact_store import ArtifactMetadata, ArtifactStore
from wei_multimodal.mcp_server.case_repository import CaseRepository
from wei_multimodal.mcp_server.contracts import ArtifactRef
from wei_multimodal.mcp_server.errors import ContractError, ErrorCode
from wei_multimodal.service.prediction import PredictionService


@dataclass(slots=True, frozen=True)
class RuntimeDependencies:
    """Long-lived objects shared by every tool call.

    The application creates this object once during its lifespan.  In particular,
    ``prediction_service`` must not be reconstructed for individual requests.
    """

    prediction_service: PredictionService
    artifact_store: ArtifactStore
    case_repository: CaseRepository
    dicom_radiomics_mode: Literal["off"] = "off"
    tool_timeout_seconds: float = 120.0
    tool_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(2),
        compare=False,
        repr=False,
    )

    @classmethod
    def create(
        cls,
        *,
        prediction_service: PredictionService,
        artifact_store: ArtifactStore,
        case_repository: CaseRepository,
        tool_timeout_seconds: float,
        max_concurrency: int,
    ) -> RuntimeDependencies:
        """Build a lifespan runtime with validated timeout and concurrency limits."""

        if tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        return cls(
            prediction_service=prediction_service,
            artifact_store=artifact_store,
            case_repository=case_repository,
            tool_timeout_seconds=tool_timeout_seconds,
            tool_semaphore=asyncio.Semaphore(max_concurrency),
        )


def artifact_ref(metadata: ArtifactMetadata) -> ArtifactRef:
    """Convert private store metadata to the public, path-free artifact contract."""

    return ArtifactRef.model_validate(metadata.to_public_dict())


def ordered_feature_sha256(names: Sequence[str]) -> str:
    """Hash locked source feature names in order, separated by a single newline."""

    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def validate_feature_mapping(
    values: Mapping[str, Any],
    expected_names: Sequence[str],
    *,
    field_name: str,
) -> dict[str, float]:
    """Validate an exact finite feature mapping and return it in model order.

    Key-set errors and non-numeric values are schema errors.  Numeric NaN/Infinity
    receive the dedicated finite-value error.  Boolean values are rejected even
    though Python treats ``bool`` as an ``int`` subclass.
    """

    expected = set(expected_names)
    actual = set(values)
    if actual != expected:
        raise ContractError(
            ErrorCode.FEATURE_SCHEMA_MISMATCH,
            message=f"{field_name} does not match the locked model schema.",
            field=field_name,
        )
    ordered: dict[str, float] = {}
    for name in expected_names:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(
                ErrorCode.FEATURE_SCHEMA_MISMATCH,
                message=f"{field_name} must contain only numeric feature values.",
                field=field_name,
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ContractError(
                ErrorCode.NON_FINITE_VALUE,
                message=f"{field_name} contains a non-finite value.",
                field=field_name,
            )
        ordered[name] = numeric
    return ordered


def require_passed_qc(
    runtime: RuntimeDependencies,
    *,
    artifact_id: str,
    trace_id: str,
    case_binding_sha256: str,
    case_ref: str,
) -> dict[str, Any]:
    """Load a case-bound QC artifact and enforce its deterministic pass state."""

    stored = runtime.artifact_store.require_case_bound(
        artifact_id,
        trace_id=trace_id,
        expected_type="case_qc",
        case_binding_sha256=case_binding_sha256,
    )
    payload = stored.payload
    if (
        not isinstance(payload, dict)
        or payload.get("passed") is not True
        or payload.get("case_ref") != case_ref
    ):
        raise ContractError(
            ErrorCode.QC_REQUIRED_OR_FAILED,
            message="A passed quality-control artifact is required.",
            field="qc_artifact_id",
        )
    return payload


__all__ = [
    "RuntimeDependencies",
    "artifact_ref",
    "ordered_feature_sha256",
    "require_passed_qc",
    "validate_feature_mapping",
]

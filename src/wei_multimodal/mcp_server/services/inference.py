"""Hard-gated multimodal inference using the existing deployment service."""

from __future__ import annotations

from typing import Any, Literal

from wei_multimodal.mcp_server.contracts import PredictionData, PredictMultimodalRequest
from wei_multimodal.mcp_server.contracts.outputs import PerformanceReference
from wei_multimodal.mcp_server.errors import ContractError, ErrorCode
from wei_multimodal.service.prediction import (
    PredictionInputError,
    PredictionRequest,
)

from .common import (
    RuntimeDependencies,
    artifact_ref,
    ordered_feature_sha256,
    require_passed_qc,
    validate_feature_mapping,
)


def _validated_modality_payload(
    payload: Any,
    *,
    expected_names: tuple[str, ...],
    expected_modality_hash: str,
    expected_model_hash: str,
    field_name: str,
) -> dict[str, float]:
    """Revalidate a stored feature artifact immediately before prediction."""

    if not isinstance(payload, dict):
        raise ContractError(
            ErrorCode.MODEL_INCOMPATIBLE,
            message="Feature artifact is incompatible with the loaded model.",
            field=field_name,
        )
    if (
        payload.get("feature_order_sha256") != expected_modality_hash
        or payload.get("model_feature_order_sha256") != expected_model_hash
        or payload.get("source_type") != "precomputed_features"
    ):
        raise ContractError(
            ErrorCode.MODEL_INCOMPATIBLE,
            message="Feature artifact fingerprint is incompatible with the loaded model.",
            field=field_name,
        )
    raw_values = payload.get("values")
    if not isinstance(raw_values, dict):
        raise ContractError(
            ErrorCode.MODEL_INCOMPATIBLE,
            message="Feature artifact payload is incompatible with the loaded model.",
            field=field_name,
        )
    try:
        return validate_feature_mapping(raw_values, expected_names, field_name=field_name)
    except ContractError as exc:
        raise ContractError(
            ErrorCode.MODEL_INCOMPATIBLE,
            message="Feature artifact values are incompatible with the loaded model.",
            field=field_name,
        ) from exc


def _proximity(distance: float) -> Literal["near_threshold", "intermediate", "far_from_threshold"]:
    """Classify absolute threshold distance without making confidence claims."""

    if distance <= 0.05:
        return "near_threshold"
    if distance <= 0.15:
        return "intermediate"
    return "far_from_threshold"


def predict_multimodal(
    request: PredictMultimodalRequest,
    runtime: RuntimeDependencies,
) -> PredictionData:
    """Run the five-member ensemble after repeating all compatibility gates.

    The returned probability, threshold and class come directly from
    :class:`PredictionService`; this layer only adds transparent threshold-distance
    and review metadata.
    """

    case = runtime.case_repository.load(request.case_ref)
    qc_payload = require_passed_qc(
        runtime,
        artifact_id=request.input.qc_artifact_id,
        trace_id=str(request.trace_id),
        case_binding_sha256=case.case_binding_sha256,
        case_ref=request.case_ref,
    )
    ct_artifact = runtime.artifact_store.require_case_bound(
        request.input.ct_artifact_id,
        trace_id=str(request.trace_id),
        expected_type="ct_features",
        case_binding_sha256=case.case_binding_sha256,
    )
    pathology_artifact = runtime.artifact_store.require_case_bound(
        request.input.pathology_artifact_id,
        trace_id=str(request.trace_id),
        expected_type="pathology_features",
        case_binding_sha256=case.case_binding_sha256,
    )
    info = runtime.prediction_service.get_info()
    schema = runtime.prediction_service.schema
    if (
        ct_artifact.metadata.model_schema_version != info.schema_version
        or pathology_artifact.metadata.model_schema_version != info.schema_version
    ):
        raise ContractError(
            ErrorCode.MODEL_INCOMPATIBLE,
            message="Feature artifact schema version is incompatible with the loaded model.",
            field="input",
        )
    ct_hash = ordered_feature_sha256(schema.all_ct_features)
    pathology_hash = ordered_feature_sha256(schema.pathology_features)
    ct_values = _validated_modality_payload(
        ct_artifact.payload,
        expected_names=schema.all_ct_features,
        expected_modality_hash=ct_hash,
        expected_model_hash=info.feature_order_sha256,
        field_name="ct_features",
    )
    pathology_values = _validated_modality_payload(
        pathology_artifact.payload,
        expected_names=schema.pathology_features,
        expected_modality_hash=pathology_hash,
        expected_model_hash=info.feature_order_sha256,
        field_name="pathology_features",
    )
    try:
        result = runtime.prediction_service.predict(
            PredictionRequest(
                schema_version=info.schema_version,
                feature_order_sha256=info.feature_order_sha256,
                pathology_features=pathology_values,
                ct_features=ct_values,
                clinical=request.input.clinical.model_dump(),
            )
        )
    except PredictionInputError as exc:
        # Contract validation already enforces age/male.  Remaining predictor input
        # failures are trained-category mismatches and receive the dedicated code.
        raise ContractError(
            ErrorCode.INVALID_CLINICAL_CATEGORY,
            message="Clinical category is not present in the trained vocabulary.",
            field="input.clinical",
        ) from exc
    margin = result.positive_probability - result.threshold
    distance = abs(margin)
    qc_warnings = qc_payload.get("quality_warnings", [])
    has_qc_warning = isinstance(qc_warnings, list) and bool(qc_warnings)
    review_reasons: list[str] = []
    if distance <= 0.05:
        review_reasons.append("NEAR_DEPLOYMENT_THRESHOLD")
    if has_qc_warning:
        review_reasons.append("INPUT_QC_WARNING")
    elevated = bool(review_reasons)
    review_priority: Literal["routine", "elevated"] = "elevated" if elevated else "routine"
    fallback_used: Literal[False] = False
    fallback_reason = None
    ct_source_used: Literal["precomputed"] = "precomputed"
    prediction_payload = {
        "positive_probability": result.positive_probability,
        "threshold": result.threshold,
        "predicted_class": result.predicted_class,
        "decision_margin": margin,
        "absolute_threshold_distance": distance,
        "decision_proximity": _proximity(distance),
        "human_review_required": True,
        "review_priority": review_priority,
        "review_reasons": review_reasons,
        "member_count": 5,
        "ct_source_used": ct_source_used,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "model_version": result.model_version,
        "independent_test_claim": False,
        "performance_reference": {"metric": "oof_roc_auc", "value": 0.7749},
    }
    metadata = runtime.artifact_store.put(
        artifact_type="prediction",
        trace_id=str(request.trace_id),
        case_binding_sha256=case.case_binding_sha256,
        parent_artifact_ids=(
            request.input.qc_artifact_id,
            request.input.ct_artifact_id,
            request.input.pathology_artifact_id,
        ),
        model_schema_version=info.schema_version,
        payload=prediction_payload,
    )
    return PredictionData(
        artifact=artifact_ref(metadata),
        positive_probability=result.positive_probability,
        threshold=result.threshold,
        predicted_class=result.predicted_class,
        decision_margin=margin,
        absolute_threshold_distance=distance,
        decision_proximity=_proximity(distance),
        human_review_required=True,
        review_priority=review_priority,
        review_reasons=review_reasons,
        member_count=5,
        ct_source_used=ct_source_used,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        model_version=result.model_version,
        independent_test_claim=False,
        performance_reference=PerformanceReference(metric="oof_roc_auc", value=0.7749),
    )


__all__ = ["predict_multimodal"]

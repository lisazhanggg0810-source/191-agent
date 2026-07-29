"""Validation and retention of approved precomputed CT radiomics features."""

from __future__ import annotations

from wei_multimodal.mcp_server.contracts import CTFeatureData, PrepareCTRequest
from wei_multimodal.mcp_server.contracts.outputs import CompatibilityData, CTGroupCounts

from .common import (
    RuntimeDependencies,
    artifact_ref,
    ordered_feature_sha256,
    require_passed_qc,
    validate_feature_mapping,
)


def prepare_ct_features(
    request: PrepareCTRequest,
    runtime: RuntimeDependencies,
) -> CTFeatureData:
    """Validate and retain the approved 1409-dimensional feature vector."""

    case = runtime.case_repository.load(request.case_ref)
    require_passed_qc(
        runtime,
        artifact_id=request.input.qc_artifact_id,
        trace_id=str(request.trace_id),
        case_binding_sha256=case.case_binding_sha256,
        case_ref=request.case_ref,
    )
    schema = runtime.prediction_service.schema
    info = runtime.prediction_service.get_info()
    values = validate_feature_mapping(
        runtime.case_repository.read_ct_features(case),
        schema.all_ct_features,
        field_name="ct_features",
    )
    feature_hash = ordered_feature_sha256(schema.all_ct_features)
    metadata = runtime.artifact_store.put(
        artifact_type="ct_features",
        trace_id=str(request.trace_id),
        case_binding_sha256=case.case_binding_sha256,
        parent_artifact_ids=(request.input.qc_artifact_id,),
        model_schema_version=info.schema_version,
        payload={
            "values": values,
            "feature_order_sha256": feature_hash,
            "model_feature_order_sha256": info.feature_order_sha256,
            "source_type": "precomputed_features",
        },
    )
    return CTFeatureData(
        artifact=artifact_ref(metadata),
        input_mode="precomputed",
        extraction_performed=False,
        source_type="precomputed_features",
        feature_count=1409,
        group_counts=CTGroupCounts(shape=14, original=93, wavelet=744, transformed=558),
        ct_feature_order_sha256=feature_hash,
        compatibility=CompatibilityData(
            status="validated",
            model_compatible=True,
            decision="allow_prediction",
            basis="approved_precomputed_case_package",
            blocking_reasons=[],
        ),
    )


__all__ = ["prepare_ct_features"]

"""Preparation of precomputed 768-dimensional patient-level pathology features."""

from __future__ import annotations

from wei_multimodal.mcp_server.contracts import PathologyFeatureData, PreparePathologyRequest

from .common import (
    RuntimeDependencies,
    artifact_ref,
    ordered_feature_sha256,
    require_passed_qc,
    validate_feature_mapping,
)


def prepare_pathology_features(
    request: PreparePathologyRequest,
    runtime: RuntimeDependencies,
) -> PathologyFeatureData:
    """Validate and retain the locked patient-level pathology feature space.

    This function never opens WSI files, creates patches, runs a neural encoder or
    claims that a different 768-dimensional embedding is compatible.
    """

    case = runtime.case_repository.load(request.case_ref)
    require_passed_qc(
        runtime,
        artifact_id=request.input.qc_artifact_id,
        trace_id=str(request.trace_id),
        case_binding_sha256=case.case_binding_sha256,
        case_ref=request.case_ref,
    )
    schema = runtime.prediction_service.schema
    raw = runtime.case_repository.read_pathology_features(case)
    values = validate_feature_mapping(
        raw,
        schema.pathology_features,
        field_name="pathology_features",
    )
    feature_hash = ordered_feature_sha256(schema.pathology_features)
    metadata = runtime.artifact_store.put(
        artifact_type="pathology_features",
        trace_id=str(request.trace_id),
        case_binding_sha256=case.case_binding_sha256,
        parent_artifact_ids=(request.input.qc_artifact_id,),
        model_schema_version=schema.version,
        payload={
            "values": values,
            "feature_order_sha256": feature_hash,
            "model_feature_order_sha256": (
                runtime.prediction_service.get_info().feature_order_sha256
            ),
            "source_type": "precomputed_features",
        },
    )
    return PathologyFeatureData(
        artifact=artifact_ref(metadata),
        input_mode="precomputed",
        extraction_performed=False,
        source_type="precomputed_features",
        feature_count=768,
        pathology_feature_order_sha256=feature_hash,
        heatmap_status="not_available_in_v1",
    )


__all__ = ["prepare_pathology_features"]

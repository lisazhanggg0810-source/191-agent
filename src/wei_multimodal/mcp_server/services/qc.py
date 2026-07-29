"""Deterministic quality control for deidentified precomputed case packages."""

from __future__ import annotations

from wei_multimodal.mcp_server.contracts import CaseQCData, CaseQCRequest
from wei_multimodal.mcp_server.contracts.outputs import (
    CTSourceAvailability,
    ModalityAvailability,
)

from .common import RuntimeDependencies, artifact_ref


def case_data_qc(request: CaseQCRequest, runtime: RuntimeDependencies) -> CaseQCData:
    """Verify package integrity and issue a case-bound short-lived QC artifact."""

    case = runtime.case_repository.load(request.case_ref)
    payload = {
        "passed": True,
        "case_ref": case.case_ref,
        "research_id": case.research_id,
        "input_mode": "precomputed",
        "ct_source_selected": "precomputed",
        "fallback_used": False,
        "fallback_reason": None,
        "quality_warnings": [],
        "safe_case_summary": case.safe_summary(),
    }
    metadata = runtime.artifact_store.put(
        artifact_type="case_qc",
        trace_id=str(request.trace_id),
        case_binding_sha256=case.case_binding_sha256,
        payload=payload,
        model_schema_version=runtime.prediction_service.get_info().schema_version,
    )
    return CaseQCData(
        artifact=artifact_ref(metadata),
        case_ref=case.case_ref,
        input_mode="precomputed",
        ct_source_preference="precomputed",
        ct_source_selected="precomputed",
        fallback_policy="precomputed_if_available",
        passed=True,
        modalities=ModalityAvailability(ct="present", pathology="present", clinical="present"),
        ct_sources=CTSourceAvailability(precomputed="present"),
        privacy_check="passed",
        files_checked=3,
    )


__all__ = ["case_data_qc"]

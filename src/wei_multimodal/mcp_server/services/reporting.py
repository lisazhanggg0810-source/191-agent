"""Deterministic, escaped research-assistance report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from wei_multimodal.mcp_server.contracts import GenerateReportRequest, ReportData
from wei_multimodal.mcp_server.errors import ContractError, ErrorCode

from .common import RuntimeDependencies, artifact_ref, require_passed_qc

SAFETY_STATEMENT = "科研辅助评估，不替代病理诊断和临床最终决策。"
ReportSection = Literal[
    "case_summary",
    "input_quality",
    "model_score",
    "limitations",
    "expert_review",
    "safety_statement",
]
REPORT_SECTIONS: tuple[ReportSection, ...] = (
    "case_summary",
    "input_quality",
    "model_score",
    "limitations",
    "expert_review",
    "safety_statement",
)
_TEMPLATE_ENVIRONMENT = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"),
    autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
    undefined=StrictUndefined,
)
_REPORT_TEMPLATE = _TEMPLATE_ENVIRONMENT.get_template("report.html.j2")


def _require_prediction_payload(payload: Any) -> dict[str, Any]:
    """Accept only the deterministic payload shape emitted by inference service."""

    required = {
        "positive_probability",
        "threshold",
        "predicted_class",
        "decision_margin",
        "absolute_threshold_distance",
        "decision_proximity",
        "review_priority",
        "model_version",
        "ct_source_used",
        "fallback_used",
        "fallback_reason",
        "independent_test_claim",
        "performance_reference",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ContractError(
            ErrorCode.INVALID_STAGE_ORDER,
            message="A valid prediction artifact is required before report generation.",
            field="prediction_artifact_id",
        )
    return payload


def _render_html(case_ref: str, prediction: dict[str, Any]) -> str:
    """Render fixed report sections with escaped dynamic values and no scripts."""

    fallback_text = "已披露输入回退。" if bool(prediction["fallback_used"]) else "未使用输入回退。"
    return _REPORT_TEMPLATE.render(
        case_ref=case_ref,
        probability=float(prediction["positive_probability"]),
        threshold=float(prediction["threshold"]),
        predicted_class=int(prediction["predicted_class"]),
        model_version=str(prediction["model_version"]),
        proximity=str(prediction["decision_proximity"]),
        fallback_text=fallback_text,
        safety_statement=SAFETY_STATEMENT,
    )


def generate_report(
    request: GenerateReportRequest,
    runtime: RuntimeDependencies,
) -> ReportData:
    """Generate an immutable HTML report from QC and prediction artifacts only."""

    case = runtime.case_repository.load(request.case_ref)
    require_passed_qc(
        runtime,
        artifact_id=request.input.qc_artifact_id,
        trace_id=str(request.trace_id),
        case_binding_sha256=case.case_binding_sha256,
        case_ref=request.case_ref,
    )
    stored = runtime.artifact_store.require_case_bound(
        request.input.prediction_artifact_id,
        trace_id=str(request.trace_id),
        expected_type="prediction",
        case_binding_sha256=case.case_binding_sha256,
    )
    prediction = _require_prediction_payload(stored.payload)
    html = _render_html(request.case_ref, prediction)
    info = runtime.prediction_service.get_info()
    metadata = runtime.artifact_store.put(
        artifact_type="report",
        trace_id=str(request.trace_id),
        case_binding_sha256=case.case_binding_sha256,
        parent_artifact_ids=(
            request.input.qc_artifact_id,
            request.input.prediction_artifact_id,
        ),
        model_schema_version=info.schema_version,
        media_type="text/html; charset=utf-8",
        payload=html,
    )
    fallback_used = bool(prediction["fallback_used"])
    ct_source_used_value = prediction.get("ct_source_used", "precomputed")
    return ReportData(
        artifact=artifact_ref(metadata),
        report_format="html",
        report_resource_available=False,
        sections=REPORT_SECTIONS,
        heatmap_status="not_available_in_v1",
        feature_attribution_status="not_available_in_v1",
        ct_source_used=ct_source_used_value,
        fallback_disclosed=fallback_used,
        safety_statement=SAFETY_STATEMENT,
    )


__all__ = ["REPORT_SECTIONS", "SAFETY_STATEMENT", "generate_report"]

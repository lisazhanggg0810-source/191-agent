"""FastMCP wrapper for five-member multimodal prediction."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import Context
from pydantic import UUID4

from wei_multimodal.mcp_server.contracts import (
    PredictionData,
    PredictMultimodalRequest,
    PredictMultimodalResponse,
    WarningPayload,
)
from wei_multimodal.mcp_server.contracts.inputs import CaseRef, PredictMultimodalInput
from wei_multimodal.mcp_server.errors import ErrorCode
from wei_multimodal.mcp_server.services import RuntimeDependencies, predict_multimodal

from .common import execute_tool, runtime_from_context


def _prediction_warnings(data: PredictionData) -> list[WarningPayload]:
    """Promote deterministic review reasons to structured transport warnings."""

    warnings: list[WarningPayload] = []
    if "NEAR_DEPLOYMENT_THRESHOLD" in data.review_reasons:
        warnings.append(
            WarningPayload(
                code="NEAR_DEPLOYMENT_THRESHOLD",
                message="Prediction is within 0.05 of the deployment threshold.",
                field="data.absolute_threshold_distance",
                review_required=True,
            )
        )
    if "INPUT_QC_WARNING" in data.review_reasons:
        warnings.append(
            WarningPayload(
                code="INPUT_QC_WARNING",
                message="Input quality control produced a review warning.",
                field="input.qc_artifact_id",
                review_required=True,
            )
        )
    return warnings


async def prediction_tool(
    contract_version: Literal["1.1.0"],
    request_id: UUID4,
    trace_id: UUID4,
    case_ref: CaseRef,
    input: PredictMultimodalInput,
    ctx: Context[Any, RuntimeDependencies, Any],
) -> PredictMultimodalResponse:
    """Run hard-gated ensemble inference and disclose threshold distance."""

    runtime = runtime_from_context(ctx)
    request = PredictMultimodalRequest(
        contract_version=contract_version,
        request_id=request_id,
        trace_id=trace_id,
        case_ref=case_ref,
        input=input,
    )
    return await execute_tool(
        request=request,
        runtime=runtime,
        tool_name="crc_lnm_predict_multimodal",
        response_type=PredictMultimodalResponse,
        operation=lambda: predict_multimodal(request, runtime),
        internal_error=ErrorCode.INFERENCE_FAILURE,
        warning_builder=_prediction_warnings,
    )


__all__ = ["prediction_tool"]

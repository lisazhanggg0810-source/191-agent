"""FastMCP wrapper for precomputed CT feature preparation."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import Context
from pydantic import UUID4

from wei_multimodal.mcp_server.contracts import PrepareCTRequest, PrepareCTResponse
from wei_multimodal.mcp_server.contracts.inputs import CaseRef, PrepareCTInput
from wei_multimodal.mcp_server.errors import ErrorCode
from wei_multimodal.mcp_server.services import RuntimeDependencies, prepare_ct_features

from .common import execute_tool, runtime_from_context


async def ct_features_tool(
    contract_version: Literal["1.1.0"],
    request_id: UUID4,
    trace_id: UUID4,
    case_ref: CaseRef,
    input: PrepareCTInput,
    ctx: Context[Any, RuntimeDependencies, Any],
) -> PrepareCTResponse:
    """Validate and retain exactly 1409 named raw CT radiomics features."""

    runtime = runtime_from_context(ctx)
    request = PrepareCTRequest(
        contract_version=contract_version,
        request_id=request_id,
        trace_id=trace_id,
        case_ref=case_ref,
        input=input,
    )
    return await execute_tool(
        request=request,
        runtime=runtime,
        tool_name="crc_lnm_prepare_ct_features",
        response_type=PrepareCTResponse,
        operation=lambda: prepare_ct_features(request, runtime),
        internal_error=ErrorCode.SERVICE_UNAVAILABLE,
    )


__all__ = ["ct_features_tool"]

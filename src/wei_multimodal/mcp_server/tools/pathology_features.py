"""FastMCP wrapper for patient-level pathology feature preparation."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import Context
from pydantic import UUID4

from wei_multimodal.mcp_server.contracts import (
    PreparePathologyRequest,
    PreparePathologyResponse,
)
from wei_multimodal.mcp_server.contracts.inputs import CaseRef, PreparePathologyInput
from wei_multimodal.mcp_server.errors import ErrorCode
from wei_multimodal.mcp_server.services import RuntimeDependencies, prepare_pathology_features

from .common import execute_tool, runtime_from_context


async def pathology_features_tool(
    contract_version: Literal["1.1.0"],
    request_id: UUID4,
    trace_id: UUID4,
    case_ref: CaseRef,
    input: PreparePathologyInput,
    ctx: Context[Any, RuntimeDependencies, Any],
) -> PreparePathologyResponse:
    """Validate and retain the locked 768-dimensional pathology embedding."""

    runtime = runtime_from_context(ctx)
    request = PreparePathologyRequest(
        contract_version=contract_version,
        request_id=request_id,
        trace_id=trace_id,
        case_ref=case_ref,
        input=input,
    )
    return await execute_tool(
        request=request,
        runtime=runtime,
        tool_name="crc_lnm_prepare_pathology_features",
        response_type=PreparePathologyResponse,
        operation=lambda: prepare_pathology_features(request, runtime),
        internal_error=ErrorCode.SERVICE_UNAVAILABLE,
    )


__all__ = ["pathology_features_tool"]

"""FastMCP wrapper for read-only model metadata."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import Context
from pydantic import UUID4

from wei_multimodal.mcp_server.contracts import GetModelInfoRequest, GetModelInfoResponse
from wei_multimodal.mcp_server.contracts.inputs import EmptyInput
from wei_multimodal.mcp_server.errors import ErrorCode
from wei_multimodal.mcp_server.services import RuntimeDependencies, get_model_info

from .common import execute_tool, runtime_from_context


async def model_info_tool(
    contract_version: Literal["1.1.0"],
    request_id: UUID4,
    trace_id: UUID4,
    input: EmptyInput,
    ctx: Context[Any, RuntimeDependencies, Any],
) -> GetModelInfoResponse:
    """Return locked model version, dimensions, hashes, members and threshold."""

    runtime = runtime_from_context(ctx)
    request = GetModelInfoRequest(
        contract_version=contract_version,
        request_id=request_id,
        trace_id=trace_id,
        input=input,
    )
    return await execute_tool(
        request=request,
        runtime=runtime,
        tool_name="crc_lnm_get_model_info",
        response_type=GetModelInfoResponse,
        operation=lambda: get_model_info(runtime),
        internal_error=ErrorCode.SERVICE_UNAVAILABLE,
    )


__all__ = ["model_info_tool"]

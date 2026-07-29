"""FastMCP wrapper for deterministic case quality control."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import Context
from pydantic import UUID4

from wei_multimodal.mcp_server.contracts import CaseQCRequest, CaseQCResponse
from wei_multimodal.mcp_server.contracts.inputs import CaseQCInput, CaseRef
from wei_multimodal.mcp_server.errors import ErrorCode
from wei_multimodal.mcp_server.services import RuntimeDependencies, case_data_qc

from .common import execute_tool, runtime_from_context


async def case_qc_tool(
    contract_version: Literal["1.1.0"],
    request_id: UUID4,
    trace_id: UUID4,
    case_ref: CaseRef,
    input: CaseQCInput,
    ctx: Context[Any, RuntimeDependencies, Any],
) -> CaseQCResponse:
    """Validate package integrity, privacy and required modalities before prediction."""

    runtime = runtime_from_context(ctx)
    request = CaseQCRequest(
        contract_version=contract_version,
        request_id=request_id,
        trace_id=trace_id,
        case_ref=case_ref,
        input=input,
    )
    return await execute_tool(
        request=request,
        runtime=runtime,
        tool_name="crc_lnm_case_data_qc",
        response_type=CaseQCResponse,
        operation=lambda: case_data_qc(request, runtime),
        internal_error=ErrorCode.SERVICE_UNAVAILABLE,
    )


__all__ = ["case_qc_tool"]
